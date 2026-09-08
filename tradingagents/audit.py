"""Opt-in, before-write sanitized research evidence. No client/env dumps."""
import base64
import contextvars
import functools
import hashlib
import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, quote_plus

_active = contextvars.ContextVar("audit_trace", default=None)
_parent = contextvars.ContextVar("audit_parent", default=None)
_PRIVATE = re.compile(
    r"authorization|cookie|api.?key|password|secret|access.?token|refresh.?token|"
    r"reasoning|thinking|signature|encrypted", re.I
)
_QUERY = re.compile(
    r"(?i)([?&](?:api[_-]?key|key|token|access_token|signature|sig|"
    r"x-amz-[^=&#\s]+|x-goog-[^=&#\s]+|password|secret)=)[^&#\s]+"
)


class Redactor:
    def __init__(self, secrets):
        forms = set()
        for secret in secrets:
            if not secret:
                continue
            forms.update((secret, quote(secret, safe=""), quote_plus(secret),
                          base64.b64encode(secret.encode()).decode(),
                          json.dumps(secret)[1:-1]))
        self.forms = sorted(forms, key=len, reverse=True)

    def text(self, text):
        for secret in self.forms:
            text = text.replace(secret, "[REDACTED]")
        text = _QUERY.sub(r"\1[REDACTED]", text)
        text = re.sub(r"(?i)\bBearer\s+[^\s\"',}]+", "Bearer [REDACTED]", text)
        text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED]", text)
        return text

    def clean(self, value, depth=0):
        if depth > 60:
            raise ValueError("audit nesting limit")
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            if _PRIVATE.search(str(value.get("type", ""))):
                return {"type": "omitted"}
            return {self.text(str(k)): ("[REDACTED]" if _PRIVATE.search(str(k))
                    else self.clean(v, depth + 1)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(v, depth + 1) for v in value]
        # Message fields only. Never serialize arbitrary provider additional_kwargs.
        if hasattr(value, "content") and hasattr(value, "type"):
            fields = {k: getattr(value, k, None) for k in
                      ("type", "content", "name", "id", "tool_call_id",
                       "tool_calls", "usage_metadata")}
            return self.clean(fields, depth + 1)
        return {"omitted_type": type(value).__name__}


class Trace:
    def __init__(self, directory, secrets=()):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "events.jsonl"
        self.redactor = Redactor(secrets)
        self.lock = threading.Lock()
        self.sequence = 0
        self.failed = False
        self.run_id = str(uuid.uuid4())

    def emit(self, kind, **fields):
        try:
            with self.lock:
                self.sequence += 1
                record = dict(kind=kind, sequence=self.sequence, run_id=self.run_id,
                              timestamp=datetime.now(timezone.utc).isoformat(),
                              monotonic=time.monotonic(), **fields)
                line = json.dumps(self.redactor.clean(record), ensure_ascii=False,
                                  allow_nan=False)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                self.path.chmod(0o600)
        except Exception:
            self.failed = True
            raise

    def write(self, name, value):
        if Path(name).name != name:
            raise ValueError("audit filename must be a basename")
        safe = self.redactor.clean(value)
        data = safe if isinstance(safe, str) else json.dumps(safe, ensure_ascii=False, indent=2)
        path = self.directory / name
        path.write_text(data, encoding="utf-8")
        path.chmod(0o600)

    def verify(self):
        if self.failed:
            raise ValueError("audit capture failed")
        for path in self.directory.iterdir():
            if not path.is_file() or path.is_symlink():
                raise ValueError("unexpected audit artifact")
            text = path.read_text(encoding="utf-8")
            if any(s in text for s in self.redactor.forms):
                raise ValueError("audit secret scan failed")
            if re.search(r"\bsk-[A-Za-z0-9_-]{12,}", text):
                raise ValueError("audit credential pattern detected")
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.directory.iterdir() if p.is_file()}


@contextmanager
def trace_scope(trace):
    token = _active.set(trace)
    try:
        yield trace
    finally:
        _active.reset(token)


def event(kind, **fields):
    trace = _active.get()
    if trace:
        trace.emit(kind, parent_id=_parent.get(), **fields)


def observe(name):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            trace = _active.get()
            if trace is None:
                return fn(*args, **kwargs)
            span = str(uuid.uuid4())
            trace.emit("data.start", name=name, span_id=span,
                       parent_id=_parent.get(), args=args, kwargs=kwargs)
            token = _parent.set(span)
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                trace.emit("data.end", name=name, span_id=span,
                           duration_s=time.monotonic() - start, result=result)
                return result
            except Exception as exc:
                trace.emit("data.error", name=name, span_id=span,
                           duration_s=time.monotonic() - start,
                           error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                _parent.reset(token)
        return wrapped
    return decorate
