"""Offline network-boundary tests; every DNS answer and connection is mocked."""
import ast
from email.message import Message
import gzip
import io
from pathlib import Path
import socket
import ssl
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

import requests
import urllib3

import enrich_gdelt_discovery_articles as enrichment
import public_download as download


def answer(address, port=443):
    v6 = ":" in address
    return (socket.AF_INET6 if v6 else socket.AF_INET, socket.SOCK_STREAM,
            socket.IPPROTO_TCP, "", (address, port, 0, 0) if v6 else (address, port))


def response(body=b"ok", status=200, headers=None):
    return urllib3.HTTPResponse(body=io.BytesIO(body), status=status, reason="Test",
        headers=headers or {"Content-Type": "text/html; charset=utf-8"}, preload_content=False)


class PublicDownloadTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        self.dns = self.enterContext(patch.object(download.socket, "getaddrinfo", return_value=[answer("8.8.8.8")]))
        self.https = self.enterContext(patch.object(download.urllib3, "HTTPSConnectionPool"))
        self.http = self.enterContext(patch.object(download.urllib3, "HTTPConnectionPool"))
        self.pool = Mock()
        self.https.return_value = self.pool
        self.http.return_value = self.pool
        self.raw = response()
        self.pool.urlopen.return_value = self.raw

    def test_non_public_literals_stop_before_dns_or_connection(self):
        for address in ("127.0.0.1", "0.0.0.0", "10.2.3.4", "172.16.0.1", "192.168.1.1",
                        "169.254.169.254", "100.100.100.200", "168.63.129.16", "192.0.2.1",
                        "224.0.0.1", "255.255.255.255", "[::1]", "[::]", "[fe80::1]",
                        "[fd00:ec2::254]", "[::ffff:127.0.0.1]", "[ff02::1]",
                        "[64:ff9b::a9fe:a9fe]", "[2002:7f00:1::]", "[2001:db8::1]"):
            with self.subTest(address=address), self.assertRaises(download.UnsafeDownload):
                with download.public_get("http://" + address + "/"):
                    self.fail("Unsafe response yielded")
        self.dns.assert_not_called()
        self.https.assert_not_called()
        self.http.assert_not_called()

    def test_ambiguous_urls_credentials_and_non_web_ports_are_rejected(self):
        for url in ("file:///secret", "ftp://example.com/", "//example.com/", "https:/example.com",
                    "http://localhost/", "http://foo.localhost/", "http://intranet/",
                    "http://127.1/", "http://2130706433/", "http://0177.0.0.1/", "http://0x7f000001/",
                    "http://127%2e0%2e0%2e1/", "https://user:secret@example.com/",
                    "https://user@example.com/", "https://example.com\\@127.0.0.1/",
                    "https://example.com:0/", "https://example.com:8443/", "https://example.com:/",
                    "https://[fe80::1%25eth0]/", " https://example.com/", "https://exam\nple.com/",
                    "https://example.com/\r\nHost:localhost", "https://example.com/\x00", "https://[::1"):
            with self.subTest(url=url), self.assertRaises(download.UnsafeDownload):
                download.parse_destination(url)
        self.dns.assert_not_called()

    def test_idna_path_and_query_normalization_preserve_request_meaning(self):
        result = download.parse_destination("HTTPS://bücher.example:443/é?a=2&b=%2F#section")
        self.assertEqual(result.host, "xn--bcher-kva.example")
        self.assertEqual(result.target, "/%C3%A9?a=2&b=%2F")
        self.assertEqual(result.url, "https://xn--bcher-kva.example/%C3%A9?a=2&b=%2F")

    def test_dns_aliases_and_mixed_public_private_answers_fail_closed(self):
        for addresses in (["127.0.0.1"], ["8.8.8.8", "10.0.0.1"], ["8.8.8.8", "::1"],
                          ["8.8.8.8", "168.63.129.16"],
                          ["8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", "169.254.169.254"]):
            self.dns.return_value = [answer(a) for a in addresses]
            with self.subTest(addresses=addresses), self.assertRaises(download.UnsafeDownload):
                with download.public_get("https://news.example.com/"):
                    self.fail("Unsafe DNS answers yielded a response")
        self.https.assert_not_called()

    def test_dns_failure_or_empty_answers_do_not_connect(self):
        for value in ([], socket.gaierror("failed")):
            self.dns.side_effect = value if isinstance(value, Exception) else None
            self.dns.return_value = value
            with self.subTest(value=value), self.assertRaises(requests.ConnectionError):
                with download.public_get("https://news.example.com/"):
                    self.fail("No DNS answer")
        self.https.assert_not_called()

    def test_ipv6_scoped_dns_answer_is_blocked(self):
        self.dns.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP,
                                 "", ("2606:4700:4700::1111", 443, 0, 2))]
        with self.assertRaises(download.UnsafeDownload):
            with download.public_get("https://news.example.com/"):
                self.fail("Scoped address")
        self.https.assert_not_called()

    def test_https_uses_checked_ip_with_original_host_and_verified_tls(self):
        self.dns.side_effect = [[answer("8.8.8.8")], [answer("127.0.0.1")]]
        with download.public_get("https://news.example.com/story?q=1", timeout=(6, 12)) as result:
            self.assertEqual(result.url, "https://news.example.com/story?q=1")
            self.assertEqual(b"".join(result.iter_content(10)), b"ok")
        self.dns.assert_called_once()
        args = self.https.call_args
        self.assertEqual(args.args, ("8.8.8.8",))
        self.assertEqual(args.kwargs["server_hostname"], "news.example.com")
        self.assertEqual(args.kwargs["assert_hostname"], "news.example.com")
        self.assertEqual(args.kwargs["cert_reqs"], ssl.CERT_REQUIRED)
        self.assertEqual(args.kwargs["ca_certs"], requests.certs.where())
        call = self.pool.urlopen.call_args
        self.assertEqual(call.args, ("GET", "/story?q=1"))
        self.assertEqual(call.kwargs["headers"]["Host"], "news.example.com")
        self.assertIs(call.kwargs["redirect"], False)
        self.assertIs(call.kwargs["retries"], False)
        self.assertIs(call.kwargs["preload_content"], False)
        self.pool.close.assert_called_once()

    def test_public_ipv6_literal_has_bracketed_host_header(self):
        with download.public_get("https://[2606:4700:4700::1111]/"):
            pass
        self.dns.assert_not_called()
        self.assertEqual(self.https.call_args.args, ("2606:4700:4700::1111",))
        self.assertEqual(self.pool.urlopen.call_args.kwargs["headers"]["Host"], "[2606:4700:4700::1111]")

    def test_credentials_cookies_proxies_and_netrc_are_not_used(self):
        with patch.dict("os.environ", {"HTTPS_PROXY": "http://127.0.0.1:8080", "NETRC": "private"}), \
                patch.object(requests.Session, "request", side_effect=AssertionError("Session bypassed")), \
                download.public_get("https://news.example.com/", headers={
                    "User-Agent": "Reader", "Accept": "text/html", "Authorization": "secret",
                    "Cookie": "secret", "Proxy-Authorization": "secret", "X-Api-Key": "secret",
                    "Host": "localhost", "Referer": "https://private/"}):
            pass
        headers = self.pool.urlopen.call_args.kwargs["headers"]
        self.assertEqual(set(headers), {"User-Agent", "Accept", "Host", "Accept-Encoding"})
        self.assertNotIn("secret", str(headers))
        self.assertNotIn("_proxy", self.https.call_args.kwargs)

    def test_each_redirect_is_checked_and_no_cookie_is_forwarded(self):
        pools = [Mock(), Mock(), Mock()]
        self.https.side_effect = pools
        pools[0].urlopen.return_value = response(status=302, headers={"Location": "/moved", "Set-Cookie": "secret=1"})
        pools[1].urlopen.return_value = response(status=307, headers={"Location": "https://other.example.org/article"})
        pools[2].urlopen.return_value = response(b"final")
        self.dns.side_effect = [[answer("8.8.8.8")], [answer("8.8.4.4")], [answer("1.1.1.1")]]
        with download.public_get("https://news.example.com/start") as result:
            self.assertEqual(result.url, "https://other.example.org/article")
            self.assertEqual(result.content, b"final")
        self.assertEqual([c.args[0] for c in self.https.call_args_list], ["8.8.8.8", "8.8.4.4", "1.1.1.1"])
        for pool in pools:
            pool.close.assert_called_once()
            self.assertNotIn("Cookie", pool.urlopen.call_args.kwargs["headers"])

    def test_redirect_to_private_literal_credentials_or_downgrade_stops_before_connect(self):
        for location in ("http://169.254.169.254/", "https://127.0.0.1/", "//[::1]/",
                         "https://user:secret@example.com/", "file:///secret",
                         "http://public.example.com/", "https://example.com/\nsecret"):
            self.pool.reset_mock()
            self.pool.urlopen.return_value = response(status=302, headers={"Location": location})
            with self.subTest(location=location), self.assertRaises(download.UnsafeDownload):
                with download.public_get("https://news.example.com/"):
                    self.fail("Unsafe redirect")
            self.assertEqual(self.pool.urlopen.call_count, 1)
            self.pool.close.assert_called_once()

    def test_redirect_dns_rebinding_is_rejected(self):
        self.pool.urlopen.return_value = response(status=302, headers={"Location": "/next"})
        self.dns.side_effect = [[answer("8.8.8.8")], [answer("127.0.0.1")]]
        with self.assertRaises(download.UnsafeDownload):
            with download.public_get("https://news.example.com/"):
                self.fail("Rebound hostname")
        self.assertEqual(self.pool.urlopen.call_count, 1)

    def test_redirect_loops_and_long_chains_are_bounded(self):
        self.pool.urlopen.return_value = response(status=302, headers={"Location": "/"})
        with self.assertRaisesRegex(download.UnsafeDownload, "redirect_loop"):
            with download.public_get("https://news.example.com/"):
                pass
        self.assertEqual(self.pool.urlopen.call_count, 1)
        self.pool.reset_mock()
        self.pool.urlopen.side_effect = [response(status=302, headers={"Location": f"/next{i}"}) for i in range(6)]
        with self.assertRaisesRegex(download.UnsafeDownload, "redirect_limit"):
            with download.public_get("https://news.example.com/", max_redirects=100):
                pass
        self.assertEqual(self.pool.urlopen.call_count, 6)

    def test_redirect_bodies_are_closed_without_being_read(self):
        raw = Mock(status=302, headers=urllib3._collections.HTTPHeaderDict({"Location": "https://127.0.0.1/"}))
        raw.read.side_effect = AssertionError("Must not drain redirect")
        self.pool.urlopen.return_value = raw
        with self.assertRaises(download.UnsafeDownload):
            with download.public_get("https://news.example.com/"):
                pass
        raw.read.assert_not_called()
        raw.close.assert_called_once()

    def test_connect_fallback_stays_with_checked_ips_and_closes_failed_pool(self):
        self.dns.return_value = [answer("8.8.8.8"), answer("1.1.1.1")]
        first, second = Mock(), Mock()
        first.urlopen.side_effect = urllib3.exceptions.NewConnectionError(None, "unavailable")
        second.urlopen.return_value = response()
        self.https.side_effect = [first, second]
        with download.public_get("https://news.example.com/"):
            pass
        self.assertEqual([c.args[0] for c in self.https.call_args_list], ["8.8.8.8", "1.1.1.1"])
        first.close.assert_called_once()
        second.close.assert_called_once()
        self.dns.assert_called_once()

    def test_certificate_failure_does_not_retry_or_disable_verification(self):
        self.dns.return_value = [answer("8.8.8.8"), answer("1.1.1.1")]
        self.pool.urlopen.side_effect = urllib3.exceptions.SSLError("certificate mismatch")
        with self.assertRaises(requests.ConnectionError):
            with download.public_get("https://news.example.com/"):
                pass
        self.https.assert_called_once()
        self.pool.close.assert_called_once()

    def test_timeout_and_consumer_error_close_resources(self):
        self.pool.urlopen.side_effect = urllib3.exceptions.ReadTimeoutError(None, "/", "slow")
        with self.assertRaises(requests.Timeout):
            with download.public_get("https://news.example.com/"):
                pass
        self.pool.close.assert_called_once()
        self.pool.reset_mock(side_effect=True)
        self.pool.urlopen.return_value = response()
        with self.assertRaisesRegex(RuntimeError, "consumer"):
            with download.public_get("https://news.example.com/"):
                raise RuntimeError("consumer")
        self.pool.close.assert_called_once()

    def test_bounded_read_handles_gzip_and_refuses_unrequested_encodings(self):
        self.pool.urlopen.return_value = response(gzip.compress(b"x" * 5000), headers={
            "Content-Type": "text/html; charset=utf-8", "Content-Encoding": "gzip"})
        with download.public_get("https://news.example.com/") as result:
            self.assertEqual(enrichment.response_text_limited(result, 100), "x" * 100)
        for encoding in ("br", "zstd", "gzip, gzip", "unknown"):
            self.pool.urlopen.return_value = response(headers={"Content-Encoding": encoding})
            with self.subTest(encoding=encoding), self.assertRaises(download.UnsafeDownload):
                with download.public_get("https://news.example.com/"):
                    self.fail("Unexpected decoder")


class WireBoundaryTests(unittest.TestCase):
    def test_real_http_pool_connects_to_numeric_ip_not_a_second_hostname_lookup(self):
        # The real urllib3 -> http.client path runs against a fake socket only.
        addresses = []

        def dns(host, port, *args, **kwargs):
            addresses.append(host)
            if host not in {"news.example.com", "8.8.8.8"}:
                raise AssertionError("Unexpected destination")
            if addresses.count("news.example.com") > 1:
                return [answer("127.0.0.1", 80)]
            return [answer("8.8.8.8", 80)]

        sock = Mock()
        sock.makefile.return_value = io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nContent-Type: text/html\r\n\r\nsafe")
        with patch.object(socket, "getaddrinfo", side_effect=dns), patch.object(socket, "socket", return_value=sock):
            with download.public_get("http://news.example.com/article") as result:
                self.assertEqual(result.content, b"safe")
        self.assertEqual(addresses, ["news.example.com", "8.8.8.8"])
        sock.connect.assert_called_once_with(("8.8.8.8", 80))
        wire = b"".join(c.args[0] for c in sock.sendall.call_args_list)
        self.assertIn(b"GET /article HTTP/1.1\r\n", wire)
        self.assertIn(b"Host: news.example.com\r\n", wire)


class FetchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        self.enterContext(patch("socket.getaddrinfo", side_effect=AssertionError("No live DNS")))

    def fetch(self, raw, title="Example"):
        pool = Mock()
        with patch.object(download, "resolve_public", return_value=["8.8.8.8"]), \
                patch.object(download, "_open_pinned", return_value=(pool, raw)):
            session = Mock()
            result = enrichment.fetch_metadata(session, "https://example.com/article", title, 12, 10000)
            session.get.assert_not_called()
        pool.close.assert_called_once()
        return result

    def test_metadata_extraction_and_success_contract_unchanged(self):
        desc = "The council approved a new bridge following a public hearing on Thursday."
        html = f'<meta property="og:description" content="{desc}"><meta property="og:site_name" content="Example">'
        metadata, status, cacheable = self.fetch(response(html.encode()))
        self.assertEqual(metadata, {"description": desc, "site_name": "Example"})
        self.assertEqual((status, cacheable), ("ok", True))

    def test_existing_http_and_non_html_status_contracts_remain(self):
        for status, cacheable in ((404, True), (429, False), (500, False)):
            with self.subTest(status=status):
                self.assertEqual(self.fetch(response(status=status)), ({}, f"http_{status}", cacheable))
        self.assertEqual(self.fetch(response(headers={"Content-Type": "image/png"})), ({}, "not_html", True))

    def test_blocked_metadata_destination_is_empty_and_not_cached(self):
        with patch.object(enrichment, "public_get", side_effect=download.UnsafeDownload("non_public_address")):
            self.assertEqual(enrichment.fetch_metadata(Mock(), "http://private/", "Title", 12, 1000),
                             ({}, "blocked_destination", False))

    def signal_functions(self):
        # Load only I/O functions; never initialize models or load credentials.
        tree = ast.parse((Path(__file__).resolve().parents[1] / "generate_next_signals.py").read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {
            "_fetch_article_text", "_fetch_raw_document"}]
        ns = dict(Request=Request, Message=Message, public_get=download.public_get,
                  HTTPError=HTTPError, URLError=URLError, MAX_FETCH_BYTES=1000,
                  _canonical_url=lambda url: url, _safe_public_url=lambda url: True,
                  _extract_article_text=lambda body: body, _clean_text=lambda text: text)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "signal_fetches", "exec"), ns)
        return ns

    def test_optional_signal_fetches_use_safe_transport_and_keep_charset_and_cache(self):
        ns = self.signal_functions()
        for name in ("_fetch_article_text", "_fetch_raw_document"):
            cache = {}
            pool = Mock()
            raw = response("Café checkpoint".encode("iso-8859-1"), headers={"Content-Type": "text/html; charset=iso-8859-1"})
            args = ("https://example.com/story", cache) + (("raw",) if name == "_fetch_raw_document" else ())
            with patch.object(download, "resolve_public", return_value=["8.8.8.8"]), \
                    patch.object(download, "_open_pinned", return_value=(pool, raw)) as opened:
                text, info, dirty = ns[name](*args)
                self.assertEqual(text, "Café checkpoint")
                self.assertTrue(dirty)
                self.assertEqual(info["status"], "ok")
                self.assertEqual(ns[name](*args), (text, info, False))
                opened.assert_called_once()
            pool.close.assert_called_once()

    def test_optional_signal_fetches_withhold_unsafe_evidence(self):
        ns = self.signal_functions()
        for name in ("_fetch_article_text", "_fetch_raw_document"):
            args = ("http://127.0.0.1/", {}) + (("raw",) if name == "_fetch_raw_document" else ())
            text, info, _ = ns[name](*args)
            self.assertEqual(text, "")
            self.assertEqual(info["status"], "fetch_failed")


if __name__ == "__main__":
    unittest.main()
