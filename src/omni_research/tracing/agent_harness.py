"""A small tool-using agent that runs against a real omni server and records agent traces.

Purpose: produce *observed* application-level traces (sessions, steps, model requests, tool
calls, dependencies, user think time) for the replay engine, with tasks whose answers can be
checked automatically. All task content is synthetic and generated here from a seed.

Protocol (prompted, so it works whether or not the server implements tool calling): each model
reply must be one JSON object, either ``{"tool": NAME, "args": {...}}`` or ``{"final": TEXT}``.
Tools run locally and deterministically. Optionally a tool call also sleeps for a *simulated*
remote latency (recorded as ``simulated_latency_s``), standing in for a web API; that part of
the trace is synthetic and labelled so.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import operator
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..replay.backends import Backend, ModelRequest, RequestResult
from .collector import SpanHandle, TraceCollector
from .schema import EV_COMPLETE, EV_FIRST_AUDIO, EV_FIRST_TEXT, EV_REQUEST_SENT, SpanKind, new_id

# ------------------------------------------------------------------------------------------
# Tools

_OPS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def safe_eval(expr: str) -> float:
    """Arithmetic only (numbers, + - * / // % **, parentheses); anything else raises ValueError."""
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 16:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element {type(node).__name__}")
    if len(expr) > 200:
        raise ValueError("expression too long")
    return ev(ast.parse(expr.replace("^", "**").replace("×", "*"), mode="eval"))


UNIT_FACTORS = {("miles", "km"): 1.609344, ("km", "miles"): 1 / 1.609344, ("kg", "lb"): 2.2046226218,
                ("lb", "kg"): 1 / 2.2046226218, ("inches", "cm"): 2.54, ("cm", "inches"): 1 / 2.54}


def make_knowledge_base(seed: int) -> dict[str, dict[str, int]]:
    """Fictional warehouse inventory (synthetic): item -> {warehouse: count}."""
    rng = random.Random(seed)
    items = ["copper wire", "steel bolts", "glass panels", "rubber seals", "oak planks", "solar cells"]
    return {it: {w: rng.randint(10, 400) for w in ("north", "south", "east")} for it in items}


@dataclass
class ToolBox:
    kb: dict[str, dict[str, int]]
    simulated_latency: tuple[float, float] | None = None  # lognormal (median_s, sigma); None = no sleep
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def describe(self) -> str:
        return (
            'calculator(expression: str) -> number. Arithmetic only, e.g. "17*23+4".\n'
            'convert(value: number, from_unit: str, to_unit: str) -> number. Units: miles, km, kg, lb, inches, cm.\n'
            "inventory(item: str, warehouse: str) -> integer count. Warehouses: north, south, east. "
            f"Items: {', '.join(sorted(self.kb))}."
        )

    def call(self, name: str, args: dict[str, Any]) -> Any:
        if name == "calculator":
            return round(safe_eval(str(args["expression"])), 6)
        if name == "convert":
            key = (str(args["from_unit"]).lower(), str(args["to_unit"]).lower())
            if key not in UNIT_FACTORS:
                raise ValueError(f"unknown conversion {key}")
            return round(float(args["value"]) * UNIT_FACTORS[key], 4)
        if name == "inventory":
            item, wh = str(args["item"]).lower(), str(args["warehouse"]).lower()
            if item not in self.kb or wh not in self.kb[item]:
                raise ValueError(f"no inventory record for {item!r} in {wh!r}")
            return self.kb[item][wh]
        raise ValueError(f"unknown tool {name!r}")

    def latency(self) -> float:
        if not self.simulated_latency:
            return 0.0
        median, sigma = self.simulated_latency
        return median * math.exp(sigma * self.rng.gauss(0, 1))


# ------------------------------------------------------------------------------------------
# Tasks


@dataclass
class AgentTask:
    task_id: str
    kind: str
    question: str
    answer: float
    tolerance: float = 0.01  # relative
    audio_path: str | None = None  # spoken version of the question, when the input is audio
    video_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def make_tasks(n: int, seed: int, kb: dict[str, dict[str, int]]) -> list[AgentTask]:
    rng = random.Random(seed)
    out: list[AgentTask] = []
    for i in range(n):
        kind = ["arith", "convert", "inventory_sum"][i % 3]
        if kind == "arith":
            a, b, c = rng.randint(11, 99), rng.randint(11, 99), rng.randint(2, 30)
            q = f"A crate holds {a} boxes and each box holds {b} parts. After {c} parts are removed, how many parts remain?"
            ans = a * b - c
        elif kind == "convert":
            v = rng.randint(5, 250)
            u = rng.choice([("miles", "km"), ("kg", "lb"), ("inches", "cm")])
            q = f"Convert {v} {u[0]} to {u[1]}. Give the number."
            ans = round(v * UNIT_FACTORS[u], 4)
        else:
            item = rng.choice(sorted(kb))
            w1, w2 = rng.sample(["north", "south", "east"], 2)
            q = f"How many {item} are stored in the {w1} and {w2} warehouses combined?"
            ans = kb[item][w1] + kb[item][w2]
        out.append(AgentTask(task_id=f"task-{seed}-{i:04d}", kind=kind, question=q, answer=float(ans)))
    return out


def check_answer(text: str, task: AgentTask) -> bool:
    nums = [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text or "")]
    return any(abs(x - task.answer) <= max(1e-6, abs(task.answer) * task.tolerance) for x in nums)


# ------------------------------------------------------------------------------------------
# Agent loop

SYSTEM_PROMPT = """You are a careful assistant that solves the user's task with tools.
Each reply must contain only JSON, in one of two forms:
  1. One or more tool calls, each {{"tool": "<name>", "args": {{...}}}}. Several calls may be listed together when
     they do not depend on each other. Do not include a final answer in the same reply as tool calls.
  2. When you know the answer: {{"final": "<short answer containing the number>"}}
Tools:
{tools}
Always use a tool for arithmetic, conversions and inventory counts. Examples:
  {{"tool": "calculator", "args": {{"expression": "12*7+3"}}}}
  {{"tool": "convert", "args": {{"value": 10, "from_unit": "kg", "to_unit": "lb"}}}}
  {{"tool": "inventory", "args": {{"item": "oak planks", "warehouse": "north"}}}} {{"tool": "inventory", "args": {{"item": "oak planks", "warehouse": "east"}}}}
After the tool results arrive, reply with {{"final": ...}} or with more tool calls."""


class _Pairs(dict):
    """dict that also keeps every (key, value) pair, so duplicate keys are not lost."""

    pairs: list[tuple[str, Any]]


def _pairs_hook(pairs: list[tuple[str, Any]]) -> _Pairs:
    d = _Pairs(pairs)
    d.pairs = pairs
    return d


def _json_objects(text: str) -> list[_Pairs]:
    """All top-level JSON objects embedded in a reply, left to right."""
    dec = json.JSONDecoder(object_pairs_hook=_pairs_hook)
    out: list[_Pairs] = []
    i = 0
    while True:
        j = text.find("{", i)
        if j < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end


def _normalize_args(tool: str, args: Any) -> dict[str, Any]:
    """Accept the argument shapes models actually produce for our three tools."""
    if isinstance(args, dict):
        return dict(args)
    if isinstance(args, str):
        a = args.strip()
        if a.startswith("{"):
            try:
                parsed = json.loads(a)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        if tool == "calculator":
            return {"expression": a}
        if tool == "inventory" and "," in a:
            item, wh = (x.strip() for x in a.rsplit(",", 1))
            return {"item": item, "warehouse": wh}
    raise ValueError(f"arguments for {tool} must be a JSON object, got {type(args).__name__}")


def parse_actions(text: str, tool_names: tuple[str, ...] = ("calculator", "convert", "inventory")) -> dict[str, Any]:
    """Tool calls and/or a final answer from a model reply.

    Accepts {"tool": NAME, "args": {...}} objects, the shorthand {NAME: {...}} (duplicate keys
    kept, so two calls in one object both count), and {"final": TEXT}. Returns
    {"calls": [(name, raw_args), ...], "final": str | None}.
    """
    calls: list[tuple[str, Any]] = []
    final = None
    for obj in _json_objects(text or ""):
        if "tool" in obj:
            calls.append((str(obj["tool"]), obj.get("args", {})))
            continue
        if "final" in obj and final is None:
            final = str(obj["final"])
        for k, v in obj.pairs:
            if k in tool_names:
                calls.append((k, v))
    return {"calls": calls, "final": final}


def parse_action(text: str) -> dict[str, Any] | None:
    """First action only (kept for callers that handle one action at a time)."""
    a = parse_actions(text)
    if a["calls"]:
        name, args = a["calls"][0]
        return {"tool": name, "args": args}
    if a["final"] is not None:
        return {"final": a["final"]}
    return None


@dataclass
class AgentConfig:
    max_steps: int = 5
    max_tokens: int = 192
    speak_final: bool = False  # final answer as text+audio (voice agent)
    final_max_tokens: int = 96
    input_modality: str = "text"  # text | audio
    think_time_s: tuple[float, float] | None = None  # synthetic user think time before the task (lognormal median, sigma)
    temperature: float = 0.0
    request_timeout_s: float = 300.0


def _span_request(parent: SpanHandle, name: str, req: ModelRequest, step_deps: list[str]) -> SpanHandle:
    attrs: dict[str, Any] = {
        "input_modalities": req.input_modalities,
        "output_modalities": req.output_modalities,
        "endpoint": "chat.completions",
        "max_tokens": req.max_tokens,
        "input_text_chars": sum(len(m["content"]) for m in req.messages if isinstance(m.get("content"), str)),
        "audio_input_s": req.metadata.get("audio_input_s"),
        "n_messages": len(req.messages),
    }
    if parent._c.header.content_policy == "public":
        # Synthetic task content only: stored so a replay can resend the exact request.
        attrs["request_body"] = {"messages": req.messages, "audios": req.audios, "videos": req.videos,
                                 "input_modalities": req.input_modalities, "output_modalities": req.output_modalities,
                                 "max_tokens": req.max_tokens, "temperature": req.temperature}
    return parent.child(SpanKind.MODEL_REQUEST, name, request_id=req.request_id, depends_on=step_deps, attributes=attrs)


def _close_request(h: SpanHandle, res: RequestResult) -> None:
    if res.sent_ns:
        h.event(EV_REQUEST_SENT, time_ns=res.sent_ns)
    if res.first_text_ns:
        h.event(EV_FIRST_TEXT, time_ns=res.first_text_ns)
    if res.first_audio_ns:
        h.event(EV_FIRST_AUDIO, time_ns=res.first_audio_ns)
    h.event(EV_COMPLETE, time_ns=res.done_ns or h._c.now(), status=res.status)
    status = {"ok": "ok", "timeout": "timeout", "cancelled": "cancelled"}.get(res.status, "error")
    h.end(status, completion_tokens=res.completion_tokens, prompt_tokens=res.prompt_tokens,
          output_text_chars=res.text_chars, output_audio_s=res.output_audio_s, error=res.error,
          ttft_s=None if not (res.first_text_ns and res.sent_ns) else (res.first_text_ns - res.sent_ns) / 1e9)


async def run_agent_session(
    backend: Backend,
    collector: TraceCollector,
    task: AgentTask,
    tools: ToolBox,
    cfg: AgentConfig,
    rng: random.Random,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    sid = f"sess-{task.task_id}-{new_id()[:8]}"
    outcome: dict[str, Any] = {"session_id": sid, "task_id": task.task_id, "kind": task.kind, "steps": 0,
                               "tool_calls": 0, "success": False, "error": None}
    with collector.span(SpanKind.SESSION, "agent_session", session_id=sid,
                        attributes={"task_kind": task.kind, "input_modality": cfg.input_modality}) as sess:
        last: list[str] = []
        if cfg.think_time_s:
            median, sigma = cfg.think_time_s
            think = median * math.exp(sigma * rng.gauss(0, 1))
            with sess.child(SpanKind.WAIT, "user_think", attributes={"reason": "user_think", "synthetic": True,
                                                                    "distribution": f"lognormal(median={median},sigma={sigma})"}) as w:
                await asyncio.sleep(think)
            last = [w.span_id]
        with sess.child(SpanKind.INTERACTION, "user_turn", interaction_id=f"it-{new_id()[:8]}") as turn:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools.describe())},
                {"role": "user", "content": task.question if cfg.input_modality == "text"
                 else "The task is in the attached audio. Solve it."},
            ]
            audios = [task.audio_path] if cfg.input_modality == "audio" and task.audio_path else None
            final_text = None
            for step in range(cfg.max_steps):
                outcome["steps"] = step + 1
                with turn.child(SpanKind.AGENT_STEP, f"step-{step}", step_id=f"{sid}-s{step}", depends_on=last) as st:
                    rid = f"agent-{sid}-s{step}-{new_id()[:6]}"
                    req = ModelRequest(rid, list(messages), [cfg.input_modality] + (["text"] if cfg.input_modality != "text" else []),
                                       ["text"], max_tokens=cfg.max_tokens, audios=audios if step == 0 else None,
                                       temperature=cfg.temperature, timeout_s=cfg.request_timeout_s,
                                       metadata={"session_id": sid, "step": step, "task_id": task.task_id})
                    if req.audios is None and "audio" in req.input_modalities:
                        req.input_modalities = ["text"]
                    rh = _span_request(st, "plan", req, last)
                    res = await backend.run(req)
                    _close_request(rh, res)
                    results.append(res.to_dict() | {"phase": "agent", "session_id": sid, "step": step, "role": "plan"})
                    last = [rh.span_id]
                    if res.status != "ok":
                        outcome["error"] = f"model request {res.status}: {res.error}"
                        st.end("error", reason="model_request_failed")
                        break
                    text = res.text or ""
                    acts = parse_actions(text)
                    messages.append({"role": "assistant", "content": text})
                    if not acts["calls"] and acts["final"] is None:
                        st.set(parse_error=True)
                        messages.append({"role": "user", "content": "Reply with JSON tool calls or a final answer as instructed."})
                        continue
                    if not acts["calls"]:
                        final_text = acts["final"]
                        break
                    if acts["final"] is not None:
                        st.set(premature_final_ignored=True)  # a guess written next to tool calls
                    outcome["tool_calls"] += len(acts["calls"])
                    st.set(parallel_tool_calls=len(acts["calls"]))

                    async def run_tool(name: str, raw_args: Any, deps: list[str]) -> tuple[str, str]:
                        with st.child(SpanKind.TOOL_CALL, name, depends_on=deps,
                                      attributes={"tool_name": name, "args_bytes": len(json.dumps(raw_args, default=str))}) as tc:
                            lat = tools.latency()
                            t0 = time.monotonic()
                            try:
                                value = tools.call(name, _normalize_args(name, raw_args))
                                observation = {"tool": name, "result": value}
                                tc.set(tool_status="ok")
                            except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError) as exc:
                                observation = {"tool": name, "error": str(exc)}
                                tc.set(tool_status="error", tool_error=str(exc)[:200])
                            tc.set(local_exec_s=time.monotonic() - t0)
                            if lat:
                                await asyncio.sleep(lat)
                                tc.set(simulated_latency_s=lat, synthetic_latency=True)
                        return tc.span_id, json.dumps(observation)

                    deps = list(last)
                    results_ = await asyncio.gather(*(run_tool(n, a, deps) for n, a in acts["calls"]))
                    last = [sid_ for sid_, _ in results_]
                    messages.append({"role": "user", "content": "Tool results: " + "; ".join(obs for _, obs in results_)})
            if final_text is not None:
                outcome["success"] = check_answer(final_text, task)
                if cfg.speak_final:
                    with turn.child(SpanKind.AGENT_STEP, "speak", step_id=f"{sid}-speak", depends_on=last) as st:
                        rid = f"agent-{sid}-speak-{new_id()[:6]}"
                        req = ModelRequest(rid, [{"role": "user", "content": f"Say this answer briefly: {final_text}"}],
                                           ["text"], ["text", "audio"], max_tokens=cfg.final_max_tokens,
                                           temperature=cfg.temperature, timeout_s=cfg.request_timeout_s,
                                           metadata={"session_id": sid, "step": "speak", "task_id": task.task_id})
                        rh = _span_request(st, "speak", req, last)
                        res = await backend.run(req)
                        _close_request(rh, res)
                        results.append(res.to_dict() | {"phase": "agent", "session_id": sid, "step": "speak", "role": "speak"})
                        outcome["speech_s"] = res.output_audio_s
                        if res.status != "ok":
                            outcome["error"] = f"speech request {res.status}: {res.error}"
            elif outcome["error"] is None:
                outcome["error"] = "no final answer within max_steps"
        sess.set(success=outcome["success"], steps=outcome["steps"], tool_calls=outcome["tool_calls"])
    return outcome


async def run_agents(backend: Backend, collector: TraceCollector, tasks: list[AgentTask], tools: ToolBox,
                     cfg: AgentConfig, concurrency: int, seed: int = 0, results: list[dict[str, Any]] | None = None,
                     arrival_rate_per_s: float | None = None) -> list[dict[str, Any]]:
    """Run sessions with at most ``concurrency`` active; optional Poisson session arrivals."""
    sem = asyncio.Semaphore(concurrency)
    rng = random.Random(seed)
    results = results if results is not None else []
    t0 = time.monotonic()
    starts = []
    t = 0.0
    for _ in tasks:
        starts.append(t)
        if arrival_rate_per_s:
            t += rng.expovariate(arrival_rate_per_s)

    async def one(task: AgentTask, at: float, r: random.Random) -> dict[str, Any]:
        delay = t0 + at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        async with sem:
            return await run_agent_session(backend, collector, task, tools, cfg, r, results)

    return list(await asyncio.gather(*(one(tk, s, random.Random(rng.random())) for tk, s in zip(tasks, starts))))
