import json
import uuid

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tradingagents.audit import Trace, observe, trace_scope
from tradingagents.audit_callbacks import AuditCallback


def test_real_graph_model_tool_and_direct_span(tmp_path):
    trace = Trace(tmp_path, ["synthetic-secret"])
    cb = AuditCallback(trace)
    model = FakeListChatModel(responses=["Hold"])

    @observe("prefetch")
    def prefetch():
        return "price=7161"

    @tool
    def quote(symbol: str) -> str:
        """Read a synthetic quote without network access."""
        return prefetch()

    class State(TypedDict):
        result: str

    def analyst(state):
        data = quote.invoke({"symbol": "MTARTECH.NS"})
        return {"result": model.invoke(data).content}

    graph = StateGraph(State)
    graph.add_node("Analyst", analyst)
    graph.add_edge(START, "Analyst")
    graph.add_edge("Analyst", END)
    graph = graph.compile()
    baseline = graph.invoke({"result": ""})
    with trace_scope(trace):
        observed = graph.invoke({"result": ""}, config={"callbacks": [cb]})
    assert observed == baseline == {"result": "Hold"}
    events = [json.loads(x) for x in trace.path.read_text().splitlines()]
    kinds = {x["kind"] for x in events}
    assert {"chain.start", "chain.end", "llm.start", "llm.end",
            "tool.start", "tool.end", "data.start", "data.end"} <= kinds
    assert any(e.get("node") == "Analyst" for e in events)
    assert sum(e["kind"] == "llm.start" for e in events) == 1
    trace.verify()


def test_reasoning_never_serialized_but_usage_is(tmp_path):
    trace = Trace(tmp_path, ["synthetic-secret"])
    cb = AuditCallback(trace)
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [[AIMessage(content="visible")]], run_id=run_id,
                           invocation_params={"api_key": "synthetic-secret",
                                              "model": "fake", "tools": []})
    message = AIMessage(content="Hold", additional_kwargs={"reasoning_content": "hidden"},
                        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
    cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)
    output = trace.path.read_text()
    assert "hidden" not in output and "synthetic-secret" not in output
    assert '"total_tokens": 12' in output


def test_vendor_and_structured_fallback_emit(tmp_path, monkeypatch):
    from tradingagents.agents.utils.structured import invoke_structured_or_freetext
    from tradingagents.dataflows import interface
    monkeypatch.setattr(interface, "get_vendor", lambda *a: "yfinance")
    monkeypatch.setitem(interface.VENDOR_METHODS["get_stock_data"], "yfinance", lambda *a: "OHLCV")
    trace = Trace(tmp_path)
    class Broken:
        def invoke(self, prompt):
            raise ValueError("malformed schema")
    with trace_scope(trace):
        assert interface.route_to_vendor("get_stock_data", "MTARTECH.NS", "a", "b") == "OHLCV"
        assert invoke_structured_or_freetext(Broken(), FakeListChatModel(responses=["Hold"]),
                                            "test", str, "Judge") == "Hold"
    output = trace.path.read_text()
    assert "vendor.yfinance.get_stock_data" in output
    assert "vendor.route" in output
    assert "structured.fallback" in output
    assert "malformed schema" in output
