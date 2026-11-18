from pathlib import Path

from omni_research.orchestration.placement import Host, Task, calibrate, load_manifest, place

REPO = Path(__file__).resolve().parents[1]


def hosts():
    return {
        "mac": Host("mac", 10, 32, None, "arm64", False, link_gbps={"spark": 0.2, "slurm": 0.1}),
        "spark": Host("spark", 20, 121, "GB10", "arm64", True, setup_min=0.5, link_gbps={"mac": 0.2}),
        "slurm": Host("slurm", 24, 300, "RTX A6000", "x86_64", True, always_available=False, setup_min=15, link_gbps={"mac": 0.1}),
    }


def test_formal_benchmarks_only_on_slurm():
    t = place(Task("b", "A6000 sweep", category="benchmark", requires_gpu=True, requires_a6000=True), hosts())
    assert t.preferred_host == "slurm" and t.fallback_host is None
    t = place(Task("m", "multi-gpu", requires_gpu=True, multi_gpu=True), hosts())
    assert t.preferred_host == "slurm"


def test_gpu_dev_prefers_spark_and_respects_arm64():
    assert place(Task("s", "smoke", requires_gpu=True, arm64_ok=None), hosts()).fallback_host == "slurm"
    assert place(Task("s", "smoke", requires_gpu=True, arm64_ok=True), hosts()).preferred_host == "spark"
    assert place(Task("x", "x86 only", requires_cuda=True, arm64_ok=False), hosts()).preferred_host == "slurm"


def test_short_cpu_task_stays_on_mac_long_one_moves_to_spark():
    short = place(Task("q", "tiny", estimated_duration_minutes=0.2), hosts())
    assert short.preferred_host == "mac"
    long_ = place(Task("a", "analysis", estimated_cpu_cores=16, estimated_duration_minutes=30, input_host="spark",
                       input_size_gb=2), hosts())
    assert long_.preferred_host == "spark"


def test_no_cpu_work_on_host_running_a_benchmark():
    h = hosts()
    h["spark"].running_benchmark = True
    t = place(Task("a", "analysis", estimated_cpu_cores=8, estimated_duration_minutes=30, input_host="spark"), h)
    assert t.preferred_host == "mac"


def test_calibration_scales_estimates():
    one = [{"category": "cpu", "host": "spark", "estimated_duration_minutes": 10, "actual_duration_minutes": 30}]
    assert calibrate(one) == {}  # a single run is not enough evidence
    cal = calibrate(one * 3)
    assert cal == {"cpu@spark": 3.0}
    from omni_research.orchestration.placement import total_minutes

    task = Task("a", "x", estimated_duration_minutes=10, input_host="spark")
    assert total_minutes(task, hosts()["spark"], hosts(), cal) > total_minutes(task, hosts()["spark"], hosts(), {})


def test_pinned_task_stays_on_its_host():
    t = place(Task("pr", "draft PR", pinned_host="mac", pin_reason="GitHub 凭据只在 Mac 上"), hosts())
    assert t.preferred_host == "mac" and "GitHub" in t.reason


def test_repository_manifest_loads_and_places():
    h, tasks, history = load_manifest(REPO / "configs" / "orchestration" / "task_placement.yaml")
    placed = {t.task_id: place(t, h, calibrate(history)) for t in tasks}
    assert placed["a6000-plan-job409712"].preferred_host == "slurm"
    assert placed["spark-minicpm-smoke"].preferred_host == "spark"
    assert all(t.reason for t in placed.values())
