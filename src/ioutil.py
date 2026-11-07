"""Robust file writes. The project lives in a OneDrive-synced folder, where the sync client can briefly lock a file
while Python rewrites it (PermissionError / OSError, leaving a 0-byte file). Write to a temp file, then replace,
retrying on lock errors."""
from __future__ import annotations
import os
import time


def retry(fn, what: str, tries: int = 8, wait: float = 1.5):
    for i in range(tries):
        try:
            return fn()
        except OSError as e:  # PermissionError is a subclass
            if i == tries - 1:
                raise
            print(f"  [io] {what}: {type(e).__name__} ({e}); retry {i+1}/{tries-1}", flush=True)
            time.sleep(wait * (i + 1))


def safe_to_csv(df, path: str, **kw):
    tmp = path + ".tmp"
    retry(lambda: df.to_csv(tmp, index=False, **kw), f"write {tmp}")
    retry(lambda: os.replace(tmp, path), f"replace {path}")


def safe_append_csv(df, path: str):
    header = not os.path.exists(path)
    retry(lambda: df.to_csv(path, mode="a", header=header, index=False), f"append {path}")
