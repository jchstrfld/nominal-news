"""Public-reader boundaries using synthetic inputs, without credentials or network."""
import json
import base64
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import quote, unquote

import main
from reader_security import ReaderSecurityHeaders, legacy_chips, safe_http_url, validate_edition_date


class Elements(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.text = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def render(topics, worth_knowing=None, market_overview=None):
    return main.templates.get_template("index.html").render(
        summaries=topics, requested_date="2028-02-29", edition_intro="Welcome.",
        edition_sign_off="See you next time.", market_overview=market_overview,
        worth_knowing=worth_knowing,
    )


def topic(**updates):
    value = {"topic_title": "Example event", "summary": "A short summary.",
             "num_sources": 1, "coverage_writeup_count": 1,
             "coverage_writeups": [{"url": "https://example.com/article",
                "title": "Source article", "source_name": "Example",
                "bias": "Center", "bias_slug": "center"}]}
    value.update(updates)
    return value


class ReaderOutputTests(unittest.TestCase):
    def test_csp_allows_only_the_trusted_script_and_preserves_inline_style_support(self):
        guard = ReaderSecurityHeaders(None, Path(main.__file__).parent / "templates" / "index.html")
        html = render([topic()])
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        self.assertIn("'sha256-" + digest + "'", guard.policy)
        self.assertNotIn("unsafe-inline", guard.policy.split("script-src ")[1].split(";")[0])
        self.assertIn("style-src 'self' 'unsafe-inline'", guard.policy)
        for rule in ("connect-src 'none'", "form-action 'none'", "base-uri 'none'", "frame-ancestors 'none'"):
            self.assertIn(rule, guard.policy)

    def test_dates_require_ascii_iso_format_and_valid_calendar_day(self):
        for valid in ("2028-02-29", "0001-01-01", "9999-12-31", "2026-11-06"):
            with self.subTest(valid=valid):
                self.assertEqual(validate_edition_date(valid), valid)
        for invalid in (None, "", "2026-2-9", "2026-02-29", "2026-04-31",
                        "0000-01-01", "2026-13-01", "2026-W45-5", "20261106",
                        "2026-11-06T00:00:00", "２０２６-１１-０６", " 2026-11-06",
                        "2026-11-06\n", "../private", "probe/../../private",
                        "..\\private", "%2e%2e%2fprivate", "x" * 10000):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_edition_date(invalid)

    def test_safe_urls_retain_real_links_and_queries(self):
        for url in ("https://example.com/a?x=1&y=2#part", "http://example.com/",
                    "HTTPS://EXAMPLE.COM:443/a", "https://éxample.org/story",
                    "https://[2001:db8::1]:8443/a", "https://example.com./",
                    'https://example.com/?quote="&word=reader\'s'):
            with self.subTest(url=url):
                self.assertEqual(safe_http_url(url), url)

    def test_unsafe_or_malformed_urls_are_not_links(self):
        for url in (None, 42, "", "javascript:alert(1)", "JaVaScRiPt:alert(1)",
                    "data:text/html,<script>alert(1)</script>", "vbscript:msgbox(1)",
                    "file:///private.txt", "//example.com", "/relative", "https:/host",
                    "https:///host", " https://example.com", "https://exam\nple.com",
                    "java\tscript:alert(1)", "https://example.com/\x00x",
                    "https://example.com/\x7fx", "https://example.com\\@evil.test",
                    "https://name:password@example.com", "https://name@example.com",
                    "https://example.com:0/", "https://example.com:65536/",
                    "https://example.com:bad/", "https://-bad.example", "https://a..b",
                    "https://%65xample.com", "https://[::1", "https://[fe80::1%25eth0]/"):
            with self.subTest(url=url):
                self.assertEqual(safe_http_url(url), "")

    def test_ordinary_text_and_link_attributes_are_escaped(self):
        attack = '<img src=x onerror="window.attack=1">'
        url = 'https://example.com/?a=" onmouseover="attack()&b=2'
        # Spaces make this URL invalid; a quoted but otherwise valid URL stays one attribute.
        self.assertEqual(safe_http_url(url), "")
        url = 'https://example.com/?a="onmouseover="attack()&b=2'
        html = render([topic(topic_title=attack, summary=attack, next_signal=attack,
            coverage_writeups=[{"url": url, "title": attack, "source_name": attack,
                                "bias": "Center", "bias_slug": attack}])])
        document = Elements(html)
        self.assertFalse(any(tag == "img" for tag, _ in document.tags))
        self.assertGreaterEqual("".join(document.text).count(attack), 4)
        self.assertIn(("a", {"class": "writeup-title", "href": url,
                             "target": "_blank", "rel": "noopener noreferrer"}), document.tags)
        self.assertFalse(any(k.startswith("on") for _, attrs in document.tags for k in attrs))

    def test_all_dynamic_link_locations_validate_urls(self):
        bad = "javascript:alert(1)"
        for image_source in ("commons", "unsplash"):
            with self.subTest(image_source=image_source):
                data = topic(image_url="https://example.com/photo.jpg", image_alt="Photo",
                    image_source=image_source, image_credit="Photo: Photographer / Unsplash",
                    image_source_url=bad, image_photographer_url=bad, image_provider_url=bad,
                    image_license="CC BY 3.0", image_license_url=bad,
                    coverage_writeups=[{"url": bad, "title": "Source article",
                        "source_name": "Example", "bias": "Unknown", "bias_slug": "unknown"}])
                doc = Elements(render([data], {"items": [{"headline": "More news",
                    "summary": "More context", "source_url": bad, "source_name": "Example"}]}))
                self.assertFalse(any(attrs.get("href") == bad for _, attrs in doc.tags))
                self.assertIn(("span", {"class": "writeup-title"}), doc.tags)
                self.assertIn(("span", {"class": "worth-knowing-source"}), doc.tags)
                self.assertIn("CC BY 3.0", "".join(doc.text))

    def test_credit_url_parts_and_valid_credits_keep_attribution(self):
        data = topic(image_url="https://example.com/photo.jpg", image_credit=
            "Photo: Photographer / https://example.com/original?a=1&b=2",
            image_source_url="https://example.com/source", image_license="Public domain",
            image_license_url="https://example.com/license")
        doc = Elements(render([data]))
        links = [attrs["href"] for tag, attrs in doc.tags if tag == "a" and "href" in attrs]
        for url in (data["image_source_url"], data["image_license_url"],
                    "https://example.com/original?a=1&b=2"):
            self.assertIn(url, links)
        self.assertIn("Public Domain", "".join(doc.text))
        data["image_credit"] = "Photo: https://name:secret@example.com/photo"
        doc = Elements(render([data]))
        self.assertFalse(any("name:secret" in a.get("href", "") for _, a in doc.tags))

    def test_unsafe_images_are_not_embedded(self):
        for url in ("data:image/svg+xml,<svg onload=alert(1)>", "javascript:alert(1)",
                    "/static/private.jpg", "//example.com/photo.jpg"):
            with self.subTest(url=url):
                self.assertFalse(any(tag == "img" for tag, _ in Elements(render([
                    topic(image_url=url)])).tags))

    def test_legacy_chips_keep_outlets_links_and_bias_colors(self):
        source = ('<a class="chip" href="https://example.com/?a=1&amp;b=2" '
                  'style="background-color: #275BF533; border-color: #275BF5;">'
                  'Example &amp; Co.</a>')
        chips = legacy_chips(source)
        self.assertEqual(chips, [{"url": "https://example.com/?a=1&b=2",
            "label": "Example & Co.", "style": "background-color: #275BF533; border-color: #275BF5;"}])
        doc = Elements(render([topic(coverage_writeups=[], html_chips=source)]))
        self.assertIn(("a", {"class": "chip", "href": chips[0]["url"],
            "style": chips[0]["style"], "target": "_blank", "rel": "noopener noreferrer"}), doc.tags)

    def test_legacy_html_never_becomes_trusted_markup(self):
        source = ('<script>window.attack=1</script><svg onload="attack()"></svg>'
                  '<a class="chip" href="java&#115;cript:attack()" onclick="attack()" '
                  'style="background:url(https://evil.test/x); border-color: #275BF5; '
                  'position:fixed;color:expression(attack())">'
                  '<img src=x onerror="attack()">Outlet &lt;script&gt;alert(1)&lt;/script&gt;</a>'
                  '<a class="chip" href="https://example.com/">Good outlet</a>')
        clean = Elements(render([topic(coverage_writeups=[], html_chips=source)]))
        baseline = Elements(render([topic(coverage_writeups=[], html_chips="")]))
        self.assertEqual(sum(t == "script" for t, _ in clean.tags),
                         sum(t == "script" for t, _ in baseline.tags))
        self.assertFalse(any(t in {"img", "svg", "iframe"} for t, _ in clean.tags))
        self.assertFalse(any(k.startswith("on") for _, attrs in clean.tags for k in attrs))
        chips = [(tag, attrs) for tag, attrs in clean.tags if attrs.get("class") == "chip"]
        self.assertEqual(chips[0], ("span", {"class": "chip", "style": "border-color: #275BF5;"}))
        self.assertEqual(chips[1][1]["href"], "https://example.com/")
        self.assertIn("Outlet <script>alert(1)</script>", "".join(clean.text))


class ReaderRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="reader-", dir=Path(__file__).parent)
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.base = patch.object(main, "BASE_DIR", self.root)
        self.base.start()
        self.addCleanup(self.base.stop)
        (self.root / "static" / "outputs").mkdir(parents=True)
        (self.root / "static" / "style.css").write_text("body { color: #123; }", encoding="utf-8")
        (self.root / "static" / "outputs" / "review.html").write_text("PRIVATE REVIEW", encoding="utf-8")

    async def request(self, path="/", query=""):
        """Exercise the real ASGI application without an HTTP client or network."""
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "GET", "scheme": "http",
                 "path": unquote(path), "raw_path": path.encode(), "root_path": "",
                 "query_string": query.encode(), "headers": [],
                 "client": ("127.0.0.1", 12345), "server": ("reader.test", 80)}
        await main.app(scope, receive, send)
        status = next(m["status"] for m in messages if m["type"] == "http.response.start")
        body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
        return status, body.decode()

    async def test_bad_dates_fail_before_any_edition_file_access(self):
        for value in ("", "2026-02-29", "2026-13-01", "../private", "probe/../../private",
                      "..\\private", "%2e%2e%2fprivate", "2026-11-06\x00", "x" * 10000):
            with self.subTest(value=value), patch.object(main, "load_summaries") as load:
                status, body = await self.request(query="date=" + quote(value, safe=""))
                self.assertEqual(status, 422)
                load.assert_not_called()
                self.assertNotIn("PRIVATE", body)
                self.assertNotIn(str(self.root), body)

    def test_all_loaders_reject_traversal_before_resolving_any_path(self):
        for loader in (main.load_summaries, main.load_market_overview, main.load_worth_knowing):
            with self.subTest(loader=loader.__name__), patch.object(Path, "resolve") as resolve:
                with self.assertRaises(ValueError):
                    loader("probe/../../private")
                resolve.assert_not_called()

    async def test_missing_edition_never_launches_pipeline_even_when_enabled(self):
        with patch.dict("os.environ", {"AUTO_RUN_PIPELINE": "1"}), \
                patch("subprocess.run", side_effect=AssertionError("Pipeline must not run")), \
                patch("subprocess.Popen", side_effect=AssertionError("Pipeline must not run")):
            status, body = await self.request(query="date=2028-02-29")
        self.assertEqual(status, 200)
        self.assertNotIn('id="topic-1"', body)
        self.assertEqual(list(self.root.glob("*.json")), [])

    async def test_latest_selects_valid_saved_edition_and_explicit_date_works(self):
        for name in ("2028-02-29", "2028-02-28", "9999-99-99"):
            (self.root / f"topic_summaries_{name}.json").write_text(json.dumps([topic()]), encoding="utf-8")
        status, body = await self.request()
        self.assertEqual(status, 200)
        self.assertIn('<time datetime="2028-02-29">', body)
        self.assertIn("Source article", body)
        status, body = await self.request(query="date=2028-02-28")
        self.assertEqual(status, 200)
        self.assertIn('<time datetime="2028-02-28">', body)

    async def test_only_known_stylesheet_is_public(self):
        status, body = await self.request("/static/style.css")
        self.assertEqual(status, 200)
        self.assertEqual(body, "body { color: #123; }")
        for path in ("/static/outputs/review.html", "/static/private.json", "/static/",
                     "/static/../.env", "/static/%2e%2e/.env", "/static/%2e%2e%5c.env",
                     "/static/style.css/../outputs/review.html", "/.env", "/.git/config",
                     "/outputs/review.html", "/topic_summaries_2028-02-29.json",
                     "/docs", "/redoc", "/openapi.json"):
            with self.subTest(path=path):
                status, body = await self.request(path)
                self.assertEqual(status, 404)
                self.assertNotIn("PRIVATE REVIEW", body)
        self.assertTrue((self.root / "static" / "outputs" / "review.html").is_file())

    async def test_headers_cover_errors_and_hsts_only_trusted_https(self):
        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 404, "headers": []})
        guard = ReaderSecurityHeaders(inner, Path(main.__file__).parent / "templates" / "index.html")
        for scheme in ("http", "https"):
            messages = []
            async def send(message):
                messages.append(message)
            await guard({"type": "http", "scheme": scheme,
                         "headers": [(b"x-forwarded-proto", b"https")]}, None, send)
            headers = dict(messages[0]["headers"])
            self.assertEqual(headers[b"x-content-type-options"], b"nosniff")
            self.assertEqual(headers[b"referrer-policy"], b"no-referrer")
            self.assertEqual(headers[b"x-frame-options"], b"DENY")
            self.assertEqual(b"strict-transport-security" in headers, scheme == "https")

    async def test_resolved_aliases_are_rejected_for_editions_and_stylesheet(self):
        # Simulate Windows junctions/symlinks without requiring symlink privileges.
        resolve = Path.resolve
        for filename in ("topic_summaries_2028-02-29.json", "market_overview_2028-02-29.json",
                         "worth_knowing_2028-02-29.json", "static/style.css"):
            target = self.root / filename

            def alias(path, *args, **kwargs):
                return self.root.parent / "private.json" if path == target else resolve(path, *args, **kwargs)

            with self.subTest(filename=filename), patch.object(Path, "resolve", alias):
                status, body = await self.request("/static/style.css" if filename.startswith("static/")
                                                  else "/", "date=2028-02-29")
                self.assertEqual(status, 404)
                self.assertNotIn("private.json", body)


if __name__ == "__main__":
    unittest.main()
