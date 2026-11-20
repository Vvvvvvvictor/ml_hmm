"""One strict XRootD reader for preflight, fits, and every plot family."""

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, sleep
import uproot
import config
from until import nominal_weight_scale
from Jet_DNN.jet_io import jet_tree

READ_STEP = 200_000
SCAN_STEP = 20_000
IO_WORKERS = 4


def input_url(path):
    path = str(path)
    if path.startswith("/eos/home-"):
        p = Path(path).parts
        path = "/eos/user/" + p[2][5:] + "/" + "/".join(p[3:])
    return "root://eosuser.cern.ch/" + path if path.startswith("/eos/user/") else path


@contextmanager
def open_root(path):
    # Retry only opening. Restarting an iterator after yielding would count events twice.
    for attempt in range(3):
        try:
            root = uproot.open(
                input_url(path),
                object_cache=None,
                array_cache=None,
                timeout=120,
                num_workers=IO_WORKERS,
            )
            break
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            sleep(2**attempt)
    try:
        yield root
    finally:
        root.close()


def input_batches(
    path,
    region,
    data,
    branches,
    *,
    compute_step=READ_STEP,
    entry_start=None,
    entry_stop=None,
    statistics=None,
):
    if compute_step <= 0:
        raise ValueError("compute_step must be positive")
    stats = statistics if statistics is not None else {}
    for name in ("open_s", "read_s", "entries", "read_batches"):
        stats.setdefault(name, 0)
    started = perf_counter()
    with open_root(path) as root:
        scale = nominal_weight_scale(root, Path(path), region, data)
        tree = jet_tree(root[config.TREE])
        stats["open_s"] += perf_counter() - started
        batches = iter(
            tree.iterate(
                list(dict.fromkeys(branches)),
                step_size=READ_STEP,
                library="ak",
                entry_start=entry_start,
                entry_stop=entry_stop,
            )
        )
        while True:
            started = perf_counter()
            try:
                arrays = next(batches)
            except StopIteration:
                stats["read_s"] += perf_counter() - started
                break
            stats["read_s"] += perf_counter() - started
            stats["entries"] += len(arrays)
            stats["read_batches"] += 1
            for start in range(0, len(arrays), compute_step):
                yield arrays[start : start + compute_step], scale
