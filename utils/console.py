"""Shared Windows-console fix: cmd.exe/PowerShell's default codepage can't
encode the emoji and non-ASCII punctuation this project prints (agent avatars,
"->" arrows, etc.), which raises UnicodeEncodeError and crashes the process
mid-run on Windows. Reconfiguring stdout/stderr to UTF-8 avoids that on any
platform where the streams support it.
"""

import sys


def fix_windows_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
