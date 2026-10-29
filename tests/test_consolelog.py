"""deai.consolelog: run output is teed into runs/<run>/train.log."""

import io
import re
import subprocess
import sys
from pathlib import Path

from deai.consolelog import LOG_NAME, _Tee

ROOT = Path(__file__).resolve().parents[1]


def test_tee_keeps_console_output_and_collapses_progress_bars():
    console, log = io.StringIO(), io.StringIO()
    tee = _Tee(console, log)
    tee.write("step 1\rstep 2\rstep 3\nnext ")
    tee.write("line\npartial")
    assert console.getvalue() == "step 1\rstep 2\rstep 3\nnext line\npartial"
    lines = [re.sub(r"^\d\d:\d\d:\d\d ", "", ln) for ln in log.getvalue().splitlines()]
    assert lines == ["step 3", "next line"]  # an unfinished line waits for its newline


def test_start_logs_prints_loggers_and_traceback(tmp_path):
    script = (
        "import logging, sys\n"
        "logging.basicConfig()\n"  # a handler bound to stderr before the tee starts
        "from deai import consolelog\n"
        f"consolelog.start(__import__('pathlib').Path(r'{tmp_path}'))\n"
        "print('hello from print')\n"
        "logging.getLogger('x').warning('hello from logging')\n"
        "raise RuntimeError('boom')\n"
    )
    for _ in range(2):  # a resume appends to the same log
        r = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 1 and "hello from print" in r.stdout
    text = (tmp_path / LOG_NAME).read_text(encoding="utf-8")
    assert text.count("===== git ") == 2
    assert "hello from print" in text and "hello from logging" in text
    assert "RuntimeError: boom" in text and "===== failed after" in text
