"""LangChain callbacks with explicit payload allowlists."""
import time

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.audit import _parent


class AuditCallback(BaseCallbackHandler):
    raise_error = True
    run_inline = True

    def __init__(self, trace):
        self.trace = trace
        self.starts = {}
        self.parent_tokens = {}

    def start(self, kind, run_id, parent_run_id=None, **fields):
        key = str(run_id)
        with self.trace.lock:
            self.starts[key] = time.monotonic()
        self.trace.emit(kind + ".start", span_id=key,
                        parent_id=str(parent_run_id) if parent_run_id else None, **fields)

    def end(self, kind, run_id, **fields):
        with self.trace.lock:
            start = self.starts.pop(str(run_id), None)
        self.trace.emit(kind, span_id=str(run_id),
                        duration_s=time.monotonic() - start if start is not None else None,
                        **fields)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        metadata = kwargs.get("metadata") or {}
        self.start("chain", run_id, parent_run_id, name=kwargs.get("name"),
                   node=metadata.get("langgraph_node"), inputs=inputs,
                   step=metadata.get("langgraph_step"))
        self.parent_tokens[str(run_id)] = _parent.set(str(run_id))

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self.end("chain.end", run_id, outputs=outputs)
        token = self.parent_tokens.pop(str(run_id), None)
        if token is not None:
            _parent.reset(token)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self.end("chain.error", run_id, error_type=type(error).__name__, error=str(error))
        token = self.parent_tokens.pop(str(run_id), None)
        if token is not None:
            _parent.reset(token)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        params = kwargs.get("invocation_params") or {}
        allowed = ("model", "model_name", "temperature", "max_tokens", "tools",
                   "tool_choice", "response_format")
        self.start("llm", run_id, parent_run_id, messages=messages,
                   parameters={k: params[k] for k in allowed if k in params})

    def on_llm_end(self, response, *, run_id, **kwargs):
        generations = [[getattr(g, "message", None) or g.text for g in batch]
                       for batch in response.generations]
        out = response.llm_output or {}
        self.end("llm.end", run_id, generations=generations,
                 usage=out.get("token_usage"), model=out.get("model_name"))

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.end("llm.error", run_id, error_type=type(error).__name__, error=str(error))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self.start("tool", run_id, parent_run_id,
                   name=(serialized or {}).get("name"),
                   inputs=kwargs.get("inputs") if kwargs.get("inputs") is not None else input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self.end("tool.end", run_id, output=output)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.end("tool.error", run_id, error_type=type(error).__name__, error=str(error))

    def on_retry(self, retry_state, *, run_id, **kwargs):
        self.trace.emit("retry", span_id=str(run_id),
                        attempt=getattr(retry_state, "attempt_number", None))
