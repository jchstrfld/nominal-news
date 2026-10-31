"""Offline summary recovery checks; do not import the executable pipeline."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import summaries_cache


class Choice(dict):
    @property
    def message(self):
        return self["message"]


def response(content="Headline: Court sets hearing\nSummary: The court set a hearing.",
             finish="stop", refusal=None):
    return types.SimpleNamespace(choices=[Choice(
        finish_reason=finish, message={"content": content, "refusal": refusal})])


class SummaryRecoveryChecks(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")))
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory(dir=ROOT / "tests")))
        self.cache_file = self.folder / "cache.json"
        self.output = self.folder / "edition.json"
        self.output.write_text('[{"summary":"previous edition"}]', encoding="utf-8")
        self.before = self.output.read_bytes()
        self.api = Mock(return_value=response())
        self.source = (ROOT / "summarize_grouped_topics.py").read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)
        functions = {"format_article", "truncate_prompt", "summarize_cluster",
                     "extract_headline", "extract_body", "get_or_create_summary"}
        constants = {"SUMM_MODEL", "MAX_SUMMARY_OUTPUT_TOKENS", "MAX_TOKENS",
                     "MAX_ARTICLES_PER_CLUSTER", "PROMPT_VERSION", "MIN_ARTICLES", "MAX_CLUSTERS"}
        nodes = [n for n in self.tree.body if
                 (isinstance(n, ast.FunctionDef) and n.name in functions) or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
        self.ns = dict(re=re, json=json, os=os, OUTPUT_FILE=str(self.output),
            openai=types.SimpleNamespace(ChatCompletion=types.SimpleNamespace(create=self.api)),
            ENCODING=types.SimpleNamespace(encode=list, decode="".join),
            select_central_articles=lambda articles, k: articles[:k],
            make_summ_key=summaries_cache.make_summ_key,
            get_cached_summary=summaries_cache.get_cached_summary,
            put_cached_summary=summaries_cache.put_cached_summary,
            save_summ_cache=lambda cache: summaries_cache.save_summ_cache(cache, str(self.cache_file)))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "summary_functions", "exec"), self.ns)
        self.articles = [{"url": "https://example.org/hearing", "title": "Hearing", "description": "A hearing was set."}]

    def test_bounded_supported_request_preserves_evidence_prompt(self):
        text = self.ns["summarize_cluster"](self.articles)
        self.assertEqual("Court sets hearing", self.ns["extract_headline"](text))
        args = self.api.call_args.kwargs
        self.assertEqual("gpt-5.6-sol", args["model"])
        self.assertEqual("none", args["reasoning_effort"])
        self.assertEqual(600, args["max_completion_tokens"])
        self.assertEqual(0, args["temperature"])
        self.assertIn("Use only facts explicitly stated", args["messages"][1]["content"])

    def test_provider_failure_propagates_without_cache_or_edition_write(self):
        self.api.side_effect = RuntimeError("provider unavailable")
        cache = {}
        with self.assertRaisesRegex(RuntimeError, "saved edition was not replaced"):
            self.ns["get_or_create_summary"](self.articles, cache)
        self.api.assert_called_once()
        self.assertEqual({}, cache)
        self.assertFalse(self.cache_file.exists())
        self.assertEqual(self.before, self.output.read_bytes())

    def test_refusal_truncation_empty_or_malformed_output_is_not_cached(self):
        for value in [response(finish="length"), response(refusal="refused"), response(None),
                      response(""), response("Headline: Only a headline"),
                      response("Headline: A headline\nSummary: ")]:
            with self.subTest(value=value):
                self.api.return_value = value
                with self.assertRaises(RuntimeError):
                    self.ns["get_or_create_summary"](self.articles, {})
                self.assertFalse(self.cache_file.exists())
                self.assertEqual(self.before, self.output.read_bytes())

    def test_completed_summary_survives_later_failure_and_is_reused(self):
        second = [{**self.articles[0], "url": "https://example.org/second"}]
        cache = {"old-model-entry": {"headline": "Old", "summary": "Preserved"}}
        self.ns["get_or_create_summary"](self.articles, cache)
        saved = json.loads(self.cache_file.read_text(encoding="utf-8"))
        self.api.side_effect = RuntimeError("later failure")
        with self.assertRaises(RuntimeError):
            self.ns["get_or_create_summary"](second, cache)
        self.assertEqual(saved, json.loads(self.cache_file.read_text(encoding="utf-8")))
        self.assertEqual(self.before, self.output.read_bytes())
        with contextlib.redirect_stdout(io.StringIO()):
            self.ns["get_or_create_summary"](self.articles, saved)
        self.assertEqual(2, self.api.call_count)
        self.assertIn("old-model-entry", saved)

    def test_empty_edition_is_rejected_before_atomic_replacement(self):
        start = next(i for i, n in enumerate(self.tree.body) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.operand, ast.Name)
                     and n.test.operand.id == "summaries")
        self.ns["summaries"] = []
        with self.assertRaisesRegex(RuntimeError, "No summaries produced"):
            exec(compile(ast.Module(body=self.tree.body[start:], type_ignores=[]), "summary_publish", "exec"), self.ns)
        self.assertEqual(self.before, self.output.read_bytes())
        self.assertFalse(Path(str(self.output) + ".tmp").exists())

    def test_empty_input_is_rejected_before_any_api_call(self):
        start = next(i for i, n in enumerate(self.tree.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "valid_clusters" for t in n.targets))
        end = next(i for i, n in enumerate(self.tree.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "summ_cache" for t in n.targets))
        self.ns["clusters"] = []
        with self.assertRaisesRegex(RuntimeError, "No eligible stories"):
            exec(compile(ast.Module(body=self.tree.body[start:end], type_ignores=[]), "summary_inputs", "exec"), self.ns)
        self.api.assert_not_called()
        self.assertEqual(self.before, self.output.read_bytes())


if __name__ == "__main__":
    unittest.main()
