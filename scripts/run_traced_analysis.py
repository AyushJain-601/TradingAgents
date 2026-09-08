"""Single baseline research run; uploads are gated on sanitized artifact validation."""
import contextlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

from tradingagents.audit import Trace, trace_scope


def main():
    # No untrusted ticker/model interpolation: approved experiment is fixed.
    request = json.loads(Path("analysis-requests/trace.json").read_text())
    expected = {"ticker": "MTARTECH.NS", "analysis_date": "2026-09-07",
                "model": "deepseek-v4-flash"}
    if any(request.get(k) != v for k, v in expected.items()):
        raise SystemExit("Trace request differs from approved baseline")
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("Missing DeepSeek credential")
    output = Path("trace-output")
    if output.exists():
        raise SystemExit("Trace output already exists; refusing stale artifacts")
    secrets = [v for k, v in os.environ.items()
               if v and any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    trace = Trace(output, secrets)
    success = False
    with tempfile.TemporaryDirectory(prefix="tradingagents-audit-") as runtime:
        # Console output is discarded, never staged for upload or inspected for secrets.
        with (open(os.devnull, "w") as quiet,
              contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet),
              trace_scope(trace)):
                try:
                    from tradingagents.audit_callbacks import AuditCallback
                    from tradingagents.default_config import DEFAULT_CONFIG
                    from tradingagents.graph.trading_graph import TradingAgentsGraph

                    config = DEFAULT_CONFIG.copy()
                    config.update(llm_provider="deepseek", backend_url="https://api.deepseek.com",
                                  deep_think_llm=expected["model"], quick_think_llm=expected["model"],
                                  max_debate_rounds=1, max_risk_discuss_rounds=1,
                                  llm_max_retries=2, max_tokens=8192, checkpoint_enabled=False,
                                  audit_trace_enabled=True, results_dir=runtime + "/results",
                                  data_cache_dir=runtime + "/cache")
                    manifest = {"request": request, "config": config,
                                "commit": os.environ.get("GITHUB_SHA"),
                                "python": platform.python_version(),
                                "packages": {d.metadata["Name"]: d.version
                                             for d in importlib.metadata.distributions()},
                                "limitations": [
                                        "Application-visible tool/vendor outputs; not raw HTTP wire capture",
                                        "SDK-internal retries may not emit LangChain retry callbacks",
                                        "Model reasoning and client credentials deliberately excluded",
                                        "Live data may differ from baseline; not point-in-time replay"]}
                    trace.write("manifest.json", manifest)
                    trace.emit("run.start", request=request)
                    callback = AuditCallback(trace)
                    graph = TradingAgentsGraph(debug=False, config=config, callbacks=[callback])
                    state, decision = graph.propagate(expected["ticker"], expected["analysis_date"])
                    trace.write("final-state.json", state)
                    sections = [f'# {expected["ticker"]} traced run', f"Decision: {decision}"]
                    for field in ("market_report", "sentiment_report", "news_report",
                                  "fundamentals_report", "investment_plan",
                                  "trader_investment_plan", "final_trade_decision"):
                        sections.append(f"## {field}\n{state.get(field, '')}")
                    trace.write("report.md", "\n\n".join(sections))
                    trace.emit("run.end", decision=decision, unresolved_callback_spans=len(callback.starts))
                    success = True
                except Exception as exc:
                    trace.emit("run.error", error_type=type(exc).__name__)
        # A failure artifact is useful, but only if capture and secret scan succeeded.
        trace.write("status.json", {"analysis_success": success})
        hashes = trace.verify()
        trace.write("checksums.json", hashes)
        trace.verify()
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("artifact_ready=true\n")
    print("Sanitized trace ready. Analysis " + ("completed." if success else "failed; inspect trace."))
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # No exception text/traceback can leak outside the guarded capture path.
        raise SystemExit("Trace runner failed; artifact publication withheld.") from None
