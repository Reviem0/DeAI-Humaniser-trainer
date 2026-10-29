"""Console log for every run: everything printed goes to ``runs/<run>/train.log`` too.

``start(run_dir)`` tees stdout and stderr (prints, warnings, library loggers,
an uncaught traceback) into the log, with a time on every line. Each launch
appends a header with the command, git commit and start time, so a resumed run
keeps one log of all its sessions. Progress bars redraw with ``\\r``; only the
state each line ends in is logged.
"""

from __future__ import annotations

import atexit
import io
import logging
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

LOG_NAME = "train.log"


class _Tee(io.TextIOBase):
    """Writes to the console stream as before and, line by line, to the log."""

    def __init__(self, stream, log):
        self.stream, self.log, self._line = stream, log, ""

    def write(self, s: str) -> int:
        self.stream.write(s)
        if self.log.closed:  # after exit, a logging handler may still hold this tee
            return len(s)
        self._line += s
        *done, self._line = self._line.split("\n")
        for line in done:
            line = line.rsplit("\r", 1)[-1]  # a redrawn progress bar: keep its last state
            self.log.write(f"{datetime.now():%H:%M:%S} {line}\n")
        if done:
            self.log.flush()
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        if not self.log.closed:
            self.log.flush()

    def isatty(self) -> bool:  # tqdm and transformers check this
        return self.stream.isatty()

    def __getattr__(self, name):  # encoding, fileno, ...
        return getattr(self.stream, name)


def _git_commit(root: Path) -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{rev}{' (uncommitted changes)' if dirty else ''}" if rev else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def start(run_dir: Path) -> Path:
    """Tee this process's output into ``run_dir/train.log`` until it exits."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / LOG_NAME
    log = path.open("a", encoding="utf-8")
    t0, began = time.time(), datetime.now()
    log.write(f"\n===== {began:%Y-%m-%d %H:%M:%S} {' '.join([Path(sys.executable).name, *sys.argv])}\n"
              f"===== git {_git_commit(Path(__file__).resolve().parents[1])}\n")
    log.flush()

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, log), _Tee(old_err, log)
    # Loggers that already hold the old stream (set up at import) are moved over too.
    for logger in [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]:
        for h in getattr(logger, "handlers", []):
            if type(h) is logging.StreamHandler and h.stream in (old_out, old_err):
                h.setStream(sys.stdout if h.stream is old_out else sys.stderr)

    hook = sys.excepthook

    def log_exception(kind, value, tb):
        log.write("".join(traceback.format_exception(kind, value, tb)))
        log.write(f"===== failed after {(time.time() - t0) / 60:.1f} min\n")
        log.flush()
        hook(kind, value, tb)

    def finish():
        if not log.closed:
            sys.stdout.flush()
            sys.stdout, sys.stderr = old_out, old_err  # Python flushes them again at shutdown
            log.write(f"===== exited {datetime.now():%Y-%m-%d %H:%M:%S}, "
                      f"{(time.time() - t0) / 60:.1f} min\n")
            log.close()

    sys.excepthook = log_exception
    atexit.register(finish)
    print(f"Log: {path}", flush=True)
    return path
