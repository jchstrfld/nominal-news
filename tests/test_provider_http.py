"""Credential boundaries, using synthetic keys and offline HTTP adapters only."""
import ast
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import requests
from requests.adapters import BaseAdapter
from provider_http import ProviderRequestError, provider_get


class RecordingAdapter(BaseAdapter):
    def __init__(self, body=b'{"ok": true}', status=200, headers=None):
        self.body, self.status = body, status
        self.headers = headers or {}
        self.calls = []
        self.responses = []
        self.closed = False

    def send(self, request, **kwargs):
        self.calls.append((request, kwargs))
        r = requests.Response()
        r.status_code = self.status
        r.url, r.request = request.url, request
        r.headers.update({"Content-Type": "application/json", **self.headers})
        r.raw = io.BytesIO(self.body)
        self.responses.append(r)
        return r

    def close(self):
        self.closed = True


class ProviderHTTPTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        self.session = requests.Session()
        self.adapter = RecordingAdapter()
        self.session.mount("https://", self.adapter)
        self.enterContext(patch("provider_http.requests.Session", return_value=self.session))

    def test_credentials_stay_in_headers_and_environment_is_ignored(self):
        with patch.dict("os.environ", {"HTTPS_PROXY": "http://127.0.0.1:8888",
                                       "REQUESTS_CA_BUNDLE": "not-a-trust-store"}), \
             patch("requests.sessions.get_netrc_auth", side_effect=AssertionError("No netrc")):
            r = provider_get("newsapi", "https://newsapi.org/v2/everything",
                headers={"X-Api-Key": "synthetic-secret", "Cookie": "private",
                         "Proxy-Authorization": "private"}, params={"q": "an event"})
        self.assertEqual(r.json(), {"ok": True})
        request, kwargs = self.adapter.calls[0]
        self.assertEqual(request.headers["X-Api-Key"], "synthetic-secret")
        self.assertNotIn("synthetic-secret", request.url)
        self.assertNotIn("Cookie", request.headers)
        self.assertNotIn("Proxy-Authorization", request.headers)
        self.assertTrue(kwargs["verify"])
        self.assertEqual(kwargs["proxies"], {})
        self.assertEqual(kwargs["timeout"], 8)
        self.assertTrue(self.adapter.closed)
        self.assertTrue(r.raw.closed)

    def test_rejects_wrong_origin_path_and_embedded_credentials_before_send(self):
        for url in ("http://api.unsplash.com/photos/abc/download", "https://evil.example/photos/abc/download",
                    "https://api.unsplash.com.evil.example/photos/abc/download",
                    "https://api.unsplash.com@evil.example/photos/abc/download",
                    "https://api.unsplash.com:8443/photos/abc/download",
                    "https://api.unsplash.com./photos/abc/download",
                    "https://api.unsplash.com/photos/../private",
                    "https://api.unsplash.com/photos/%2e%2e/private",
                    "https://api.unsplash.com/photos/abc/download?client_id=private",
                    "https://api.unsplash.com/photos/abc/download#fragment",
                    "https://api.unsplash.com/\\evil.example/", " https://api.unsplash.com/photos/abc"):
            with self.subTest(url=url), self.assertRaises(ProviderRequestError) as error:
                provider_get("unsplash", url, headers={"Authorization": "Client-ID synthetic-secret"})
            self.assertEqual(str(error.exception), "invalid_provider_url")
        self.assertEqual(self.adapter.calls, [])

    def test_query_credentials_are_rejected_after_parameter_encoding(self):
        with self.assertRaises(ProviderRequestError):
            provider_get("newsapi", "https://newsapi.org/v2/everything", params={"apiKey": "synthetic-secret"})
        self.assertEqual(self.adapter.calls, [])

    def test_no_redirect_can_receive_credentials(self):
        for status in (301, 302, 303, 307, 308):
            for target in ("https://evil.example/", "http://api.unsplash.com/photos/abc", "/photos/def"):
                self.adapter.status = status
                self.adapter.headers = {"Location": target, "Set-Cookie": "private=1"}
                before = len(self.adapter.calls)
                with self.subTest(status=status, target=target), self.assertRaises(ProviderRequestError):
                    provider_get("unsplash", "https://api.unsplash.com/photos/abc/download?ixid=usage",
                                 headers={"Authorization": "Client-ID synthetic-secret"})
                self.assertEqual(len(self.adapter.calls), before + 1)

    def test_tracking_query_is_preserved_without_auto_retry(self):
        provider_get("unsplash", "https://api.unsplash.com/photos/abc-DEF_123/download?ixid=usage")
        self.assertTrue(self.adapter.calls[0][0].url.endswith("?ixid=usage"))
        self.assertEqual(len(self.adapter.calls), 1)

    def test_rejected_redirect_body_is_never_consumed(self):
        self.adapter.status = 302
        self.adapter.headers = {"Location": "https://evil.example/"}
        with patch.object(requests.Response, "iter_content", side_effect=AssertionError("Must not read redirect body")):
            with self.assertRaisesRegex(ProviderRequestError, "provider_response_rejected"):
                provider_get("unsplash", "https://api.unsplash.com/photos/example/download")
        self.assertTrue(self.adapter.responses[-1].raw.closed)

    def test_response_size_encoding_and_errors_are_safe(self):
        self.adapter.body = b"x" * 101
        with self.assertRaisesRegex(ProviderRequestError, "provider_response_too_large"):
            provider_get("newsapi", "https://newsapi.org/v2/everything", max_bytes=100)
        self.adapter.headers = {"Content-Encoding": "br"}
        with self.assertRaisesRegex(ProviderRequestError, "provider_encoding_rejected"):
            provider_get("newsapi", "https://newsapi.org/v2/everything")
        with patch.object(self.adapter, "send", side_effect=requests.ConnectionError("synthetic-secret")):
            with self.assertRaises(ProviderRequestError) as error:
                provider_get("newsapi", "https://newsapi.org/v2/everything")
        self.assertEqual(str(error.exception), "provider_request_failed")
        self.assertIsNone(error.exception.request)
        self.assertIsNone(error.exception.response)

    def test_news_fetch_mapping_unchanged_and_provider_error_body_not_logged(self):
        path = Path(__file__).resolve().parents[1] / "fetch_articles_by_date.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        ns = {"provider_get": provider_get, "NEWS_API_KEY": "synthetic-secret",
              "NEWSAPI_URL": "https://newsapi.org/v2/everything"}
        exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name == "fetch_newsapi"], []), str(path), "exec"), ns)
        article = {"title": "Title", "description": "Description", "url": "https://news.example/article",
                   "source": {"name": "Outlet"}, "publishedAt": "2026-11-06T12:00:00Z"}
        self.adapter.body = json.dumps({"status": "ok", "articles": [article]}).encode()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            rows = ns["fetch_newsapi"]("2026-11-06")
        self.assertEqual(rows, [{"title": "Title", "description": "Description", "url": article["url"],
                                "source": "Outlet", "published_at": article["publishedAt"],
                                "published_date": "2026-11-06"}])
        self.adapter.body = b'{"status":"error", "message":"synthetic-secret"}'
        with contextlib.redirect_stdout(output):
            self.assertEqual(ns["fetch_newsapi"]("2026-11-06"), [])
        self.assertNotIn("synthetic-secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
