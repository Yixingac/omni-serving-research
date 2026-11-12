"""The topology parser of scripts/slurm/gpu_p2p_check.py (the measurements need GPUs and are not tested)."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("gpu_p2p_check", Path(__file__).resolve().parents[1] / "scripts/slurm/gpu_p2p_check.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

TOPO = """\tGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tNV4\tPIX\tSYS\tNODE\t0-31\t0\t\tN/A
GPU1\tNV4\t X \tPIX\tSYS\tNODE\t0-31\t0\t\tN/A
GPU2\tPIX\tPIX\t X \tNV4\tSYS\t32-63\t1\t\tN/A
GPU3\tSYS\tSYS\tNV4\t X \tSYS\t32-63\t1\t\tN/A
NIC0\tNODE\tNODE\tSYS\tSYS\t X

Legend:

  X    = Self
  NV#  = Connection traversing a bonded set of # NVLinks
"""


def test_parse_topo_pairs_and_links():
    links = mod.parse_topo(TOPO)
    assert links == {(0, 1): "NV4", (0, 2): "PIX", (0, 3): "SYS", (1, 2): "PIX", (1, 3): "SYS", (2, 3): "NV4"}


def test_parse_topo_empty():
    assert mod.parse_topo("No devices were found") == {}
