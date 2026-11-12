"""Bounded credentialed GETs to the two supported news/photo providers.

Never follow a redirect, use environment proxies/netrc, or expose provider
response bodies and request details in errors. This is not an OpenAI client.
"""
import re
from urllib.parse import parse_qsl, urlsplit

import requests


class ProviderRequestError(requests.RequestException):
    """Safe to log: contains only a fixed reason, never a request or response."""


def _validate(provider, url):
    if (not isinstance(url, str) or len(url) > 16384
            or re.search(r"[\x00-\x20\x7f\\]", url)):
        raise ProviderRequestError("invalid_provider_url")
    try:
        parsed = urlsplit(url)
        host = {"newsapi": "newsapi.org", "unsplash": "api.unsplash.com"}[provider]
        if (parsed.scheme != "https" or parsed.netloc not in {host, host + ":443"}
                or parsed.fragment or parsed.username is not None):
            raise ValueError
        if provider == "newsapi":
            if parsed.path != "/v2/everything":
                raise ValueError
        elif (parsed.path != "/search/photos" and not re.fullmatch(
                r"/photos/[A-Za-z0-9_-]+(?:/download)?", parsed.path)):
            raise ValueError
        if any(key.lower() in {"apikey", "api_key", "client_id", "access_token", "token"}
               for key, _ in parse_qsl(parsed.query)):
            raise ValueError
    except (KeyError, ValueError):
        raise ProviderRequestError("invalid_provider_url") from None


def provider_get(provider, url, *, headers=None, params=None, timeout=8,
                 max_bytes=4_000_000):
    """Return a buffered Requests response, with the socket already closed.

    TLS verifies the fixed provider hostname. Only explicit caller credentials
    are used. Every 3xx is rejected, including same-origin redirects.
    """
    _validate(provider, url)
    allowed = {"accept", "accept-version", "user-agent"}
    allowed.add("x-api-key" if provider == "newsapi" else "authorization")
    headers = {k: v for k, v in (headers or {}).items() if k.lower() in allowed}
    headers["Accept-Encoding"] = "gzip, deflate"
    try:
        with requests.Session() as session:
            session.trust_env = False
            prepared = session.prepare_request(requests.Request("GET", url, headers=headers, params=params))
            _validate(provider, prepared.url)
            # Session.send prepares a redirect even with allow_redirects=False,
            # consuming its body first. The adapter never follows/prepares one,
            # so reject a 3xx before reading any of its body.
            with session.get_adapter(prepared.url).send(
                    prepared, timeout=timeout, stream=True, verify=True,
                    cert=None, proxies={}) as response:
                if not 200 <= response.status_code < 300:
                    raise ProviderRequestError("provider_response_rejected")
                encoding = response.headers.get("Content-Encoding", "identity").lower().strip()
                if encoding not in {"identity", "gzip", "deflate"}:
                    raise ProviderRequestError("provider_encoding_rejected")
                chunks = []
                size = 0
                for chunk in response.iter_content(chunk_size=16384):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ProviderRequestError("provider_response_too_large")
                    chunks.append(chunk)
                response._content = b"".join(chunks)
                response._content_consumed = True
                response.raw.close()
                return response
    except ProviderRequestError:
        raise
    except Exception:
        # Requests/urllib3 exceptions may include URLs or reflected secrets.
        raise ProviderRequestError("provider_request_failed") from None
