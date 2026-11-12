"""Anonymous public-page downloads with a checked, pinned network destination.

No proxy, session credentials, cookies, automatic redirects or automatic retries.
This protects article fetches, not authenticated provider APIs or browser links.
Deployment must also restrict worker egress and use the pinned dependencies.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import ipaddress
import re
import socket
import ssl
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import idna
import requests
import urllib3


MAX_REDIRECTS = 5
MAX_CONNECT_ADDRESSES = 4
_TRANSLATION_NETWORKS = tuple(map(ipaddress.ip_network, (
    "64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32",
)))
# Azure's host-service virtual IP is special despite being classified as global.
_HOST_SERVICE_IP = ipaddress.ip_address("168.63.129.16")


class UnsafeDownload(requests.RequestException):
    """A generic reason only; never include a URL, credentials or DNS answers."""


@dataclass(frozen=True)
class Destination:
    url: str
    scheme: str
    host: str
    port: int
    target: str
    authority: str


def public_ip(value):
    """Reject non-public addresses, including IPv4 embedded in IPv6."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise UnsafeDownload("invalid_address") from None
    if ("%" in str(address) or not address.is_global or address.is_multicast
            or address.is_reserved or address == _HOST_SERVICE_IP):
        raise UnsafeDownload("non_public_address")
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            public_ip(address.ipv4_mapped)
        if any(address in network for network in _TRANSLATION_NETWORKS):
            raise UnsafeDownload("translated_address")
    return address


def parse_destination(url: str) -> Destination:
    # Reject ambiguous input before urllib parsing can strip control characters.
    if (not isinstance(url, str) or len(url) > 16384
            or re.search(r"[\x00-\x20\x7f\\]", url)
            or not re.match(r"https?://", url, re.I)):
        raise UnsafeDownload("invalid_url")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if (not host or parsed.username is not None or parsed.password is not None
                or "%" in host or parsed.netloc.endswith(":")):
            raise ValueError
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if port != (443 if parsed.scheme == "https" else 80):
            raise UnsafeDownload("non_web_port")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            host = idna.encode(host, uts46=True, std3_rules=True).decode("ascii").lower().rstrip(".")
            if ("." not in host or host.endswith(".localhost") or host == "localhost"
                    or not re.search(r"[a-z]", host.rsplit(".", 1)[-1])):
                raise ValueError
            authority = host
        else:
            public_ip(address)
            host = str(address)
            authority = f"[{host}]" if address.version == 6 else host
        # Preserve valid escaping and query order; HTTP request targets are ASCII.
        path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
        query = quote(parsed.query, safe="/%?:@!$&'()*+,;=-._~")
        target = path + ("?" + query if query else "")
        normalized = urlunsplit((parsed.scheme, authority, path, query, ""))
        return Destination(normalized, parsed.scheme, host, port, target, authority)
    except UnsafeDownload:
        raise
    except (ValueError, UnicodeError):
        raise UnsafeDownload("invalid_url") from None


def resolve_public(destination: Destination) -> list[str]:
    try:
        address = ipaddress.ip_address(destination.host)
    except ValueError:
        try:
            answers = socket.getaddrinfo(destination.host, destination.port,
                                         type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
        except OSError:
            raise requests.ConnectionError("dns_unavailable") from None
        addresses = []
        for family, kind, protocol, _, sockaddr in answers:
            if (family not in {socket.AF_INET, socket.AF_INET6}
                    or kind != socket.SOCK_STREAM or protocol != socket.IPPROTO_TCP
                    or family == socket.AF_INET6 and sockaddr[3] != 0):
                raise UnsafeDownload("invalid_dns_answer")
            address = public_ip(sockaddr[0])
            if str(address) not in addresses:
                addresses.append(str(address))
        if not addresses:
            raise requests.ConnectionError("dns_unavailable")
        # Validate ALL answers, even those beyond the connection-attempt bound.
        return addresses[:MAX_CONNECT_ADDRESSES]
    return [str(public_ip(address))]


def _open_pinned(destination, addresses, headers, timeout):
    for index, address in enumerate(addresses):
        kwargs = dict(port=destination.port, timeout=urllib3.Timeout(connect=timeout[0], read=timeout[1]),
                      retries=False, maxsize=1)
        if destination.scheme == "https":
            pool = urllib3.HTTPSConnectionPool(address, server_hostname=destination.host,
                assert_hostname=destination.host, cert_reqs=ssl.CERT_REQUIRED,
                ca_certs=requests.certs.where(), **kwargs)
        else:
            pool = urllib3.HTTPConnectionPool(address, **kwargs)
        try:
            raw = pool.urlopen("GET", destination.target, headers=headers, redirect=False,
                               retries=False, preload_content=False, decode_content=False,
                               assert_same_host=False)
            return pool, raw
        except (urllib3.exceptions.NewConnectionError, urllib3.exceptions.ConnectTimeoutError):
            pool.close()
            if index + 1 == len(addresses):
                raise requests.ConnectionError("connect_failed") from None
        except urllib3.exceptions.TimeoutError:
            pool.close()
            raise requests.Timeout("read_timeout") from None
        except Exception:
            pool.close()
            raise requests.ConnectionError("download_failed") from None


@contextmanager
def public_get(url: str, *, headers=None, timeout=(6, 12), max_redirects=MAX_REDIRECTS):
    """Yield a streamed Requests-compatible response, closing every hop afterward.

    The pool connects to a numeric, checked IP; original Host/SNI/certificate
    identity are kept. Redirect targets repeat all validation before connecting.
    Only public-page headers are accepted, never auth, cookies or proxy settings.
    Existing callers remain responsible for their bounded content reads.
    """
    public_headers = {k: v for k, v in (headers or {}).items()
                      if k.lower() in {"user-agent", "accept", "accept-language"}}
    public_headers["Accept-Encoding"] = "gzip, deflate"
    destination = parse_destination(url)
    seen = set()
    for hop in range(min(max_redirects, MAX_REDIRECTS) + 1):
        if destination.url in seen:
            raise UnsafeDownload("redirect_loop")
        seen.add(destination.url)
        addresses = resolve_public(destination)
        pool, raw = _open_pinned(destination, addresses,
                                 {**public_headers, "Host": destination.authority}, timeout)
        response = None
        try:
            if raw.status in {301, 302, 303, 307, 308} and raw.headers.get("location"):
                if hop >= min(max_redirects, MAX_REDIRECTS):
                    raise UnsafeDownload("redirect_limit")
                location = raw.headers["location"]
                if re.search(r"[\x00-\x20\x7f\\]", location):
                    raise UnsafeDownload("invalid_redirect")
                target = parse_destination(urljoin(destination.url, location))
                if destination.scheme == "https" and target.scheme != "https":
                    raise UnsafeDownload("insecure_redirect")
                destination = target
                continue
            # Refuse unexpected encodings; never invoke optional Brotli/Zstd code.
            encoding = (raw.headers.get("content-encoding") or "identity").lower().strip()
            if encoding not in {"identity", "gzip", "deflate"}:
                raise UnsafeDownload("unsupported_encoding")
            response = requests.Response()
            response.status_code = raw.status
            response.headers = requests.structures.CaseInsensitiveDict(raw.headers)
            response.encoding = requests.utils.get_encoding_from_headers(response.headers)
            response.raw = raw
            response.url = destination.url
            response.reason = raw.reason
            yield response
            return
        finally:
            # Do not drain redirect/error bodies: they can be huge or compressed.
            try:
                if response is not None:
                    response.close()
                else:
                    raw.close()
            finally:
                pool.close()
