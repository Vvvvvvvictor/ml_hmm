"""Read published EOS ROOT files without Python XRootD dependencies."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uproot

@contextmanager
def open_published_root(path):
    path = os.fspath(path)
    if not path.startswith(("/eos/", "root://")):
        with uproot.open(path) as source:
            yield source
        return
    executable = shutil.which("xrdcp")
    if executable is None:
        raise RuntimeError(f"xrdcp is required to validate published ROOT file: {path}")
    url = f"root://eosuser.cern.ch/{path}" if path.startswith("/eos/") else path
    scratch = os.environ.get("_CONDOR_SCRATCH_DIR")
    with tempfile.TemporaryDirectory(prefix="dnn_readback_", dir=scratch) as temporary:
        local = str(Path(temporary) / "published.root")
        result = subprocess.run([executable, "-f", url, local], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"xrdcp readback failed for {url} (exit {result.returncode})\n{result.stdout}\n{result.stderr}")
        with uproot.open(local) as source:
            yield source
