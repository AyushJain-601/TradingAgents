import base64
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote


class AuditTests(unittest.TestCase):
    def api(self):
        from tradingagents.audit import Redactor, Trace, observe, trace_scope
        return Redactor, Trace, observe, trace_scope

    def test_feature_exists(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec("tradingagents.audit"))

    def test_redacts_nested_and_encoded_secrets_without_corrupting_numbers(self):
        Redactor, *_ = self.api()
        secret = "synthetic-secret/a+b=123"
        r = Redactor([secret])
        data = {"price": 7161, "Authorization": "Bearer other",
                "rows": [secret, quote(secret, safe=""),
                         base64.b64encode(secret.encode()).decode()],
                "url": "https://example.com/feed?apikey=unknown&symbol=MTARTECH.NS"}
        cleaned = r.clean(data)
        result = json.dumps(cleaned)
        self.assertEqual(cleaned["price"], 7161)
        for value in [secret, quote(secret, safe=""),
                      base64.b64encode(secret.encode()).decode(), "unknown", "Bearer other"]:
            self.assertNotIn(value, result)
        self.assertIn("MTARTECH.NS", result)

    def test_message_reasoning_excluded_and_unknown_repr_never_used(self):
        Redactor, *_ = self.api()
        class Dangerous:
            def __repr__(self):
                raise AssertionError("repr must not run")
        result = Redactor([]).clean({
            "reasoning_content": "hidden-thought", "thinking": "hidden-thought",
            "content": [{"type": "reasoning", "text": "hidden-thought"},
                        {"type": "text", "text": "Hold"}], "object": Dangerous()})
        self.assertNotIn("hidden-thought", json.dumps(result))
        self.assertIn("Hold", json.dumps(result))

    def test_spans_preserve_result_and_error_and_parentage(self):
        _, Trace, observe, scope = self.api()
        with tempfile.TemporaryDirectory() as d:
            trace = Trace(Path(d), ["test-secret"])
            @observe("inner")
            def inner():
                return {"price": 7161}
            @observe("outer")
            def outer():
                return inner()
            @observe("failure")
            def fail():
                raise ValueError("test-secret")
            with scope(trace):
                self.assertEqual(outer(), {"price": 7161})
                with self.assertRaises(ValueError):
                    fail()
            events = [json.loads(x) for x in trace.path.read_text().splitlines()]
            starts = {e["name"]: e for e in events if e["kind"] == "data.start"}
            self.assertEqual(starts["inner"]["parent_id"], starts["outer"]["span_id"])
            self.assertTrue(any(e["kind"] == "data.error" for e in events))
            self.assertNotIn("test-secret", trace.path.read_text())
            self.assertIn("[REDACTED]", trace.path.read_text())

    def test_parallel_writer_and_publication_scan(self):
        _, Trace, *_ = self.api()
        with tempfile.TemporaryDirectory() as d:
            trace = Trace(Path(d), ["sentinel-secret"])
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda i: trace.emit("test", value=i), range(50)))
            self.assertEqual(len(trace.path.read_text().splitlines()), 50)
            trace.verify()
            (Path(d) / "leak.txt").write_text("sentinel-secret")
            with self.assertRaises(ValueError):
                trace.verify()


if __name__ == "__main__":
    unittest.main()
