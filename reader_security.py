"""Output-boundary checks for the public reader; no pipeline or network access."""
from datetime import date
import base64
import hashlib
from html.parser import HTMLParser
from ipaddress import ip_address
import re
from urllib.parse import urlsplit


class ReaderSecurityHeaders:
    """Browser protections; allow only the fixed template's inline script.

    Hash trusted template source, never rendered story content. Inline styles
    remain needed for the validated crop/coverage values and legacy chips.
    """
    def __init__(self, app, template_path):
        self.app = app
        scripts = re.findall(r"<script>(.*?)</script>", template_path.read_text(encoding="utf-8"), re.S)
        if any("{{" in script or "{%" in script for script in scripts):
            raise ValueError("Reader scripts must not interpolate edition data")
        hashes = ["'sha256-" + base64.b64encode(hashlib.sha256(script.encode()).digest()).decode() + "'"
                  for script in scripts]
        self.policy = ("default-src 'none'; script-src " + (" ".join(hashes) or "'none'") +
            "; style-src 'self' 'unsafe-inline'; img-src https: http:; font-src 'self'; "
            "connect-src 'none'; object-src 'none'; base-uri 'none'; "
            "form-action 'none'; frame-ancestors 'none'")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def secured(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"content-security-policy", self.policy.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
                ])
                # Only a genuinely HTTPS ASGI connection; do not trust arbitrary
                # forwarded headers. The deployment proxy must be configured.
                if scope.get("scheme") == "https":
                    headers.append((b"strict-transport-security", b"max-age=31536000"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, secured)


def validate_edition_date(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("Use a valid date in YYYY-MM-DD format.")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError("Use a valid date in YYYY-MM-DD format.") from None
    return value


def safe_http_url(value) -> str:
    """Allow absolute HTTP(S) links, not executable schemes or embedded credentials.

    This is browser-output validation, not a server-side download/SSRF policy.
    Return a plain string so the template still applies HTML attribute escaping.
    """
    if not isinstance(value, str) or re.search(r"[\x00-\x20\x7f\\]", value):
        return ""
    if not re.match(r"https?://", value, re.IGNORECASE):
        return ""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        if (parsed.scheme.lower() not in {"http", "https"} or not host
                or parsed.username is not None or parsed.password is not None
                or parsed.port is not None and not 1 <= parsed.port <= 65535):
            return ""
        if ":" in host:
            # Validate IPv6 literals, excluding scoped/local interface syntax.
            if "%" in host:
                return ""
            ip_address(host)
        else:
            ascii_host = host.encode("idna").decode("ascii").rstrip(".")
            if len(ascii_host) > 253 or not all(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                for label in ascii_host.split(".")
            ):
                return ""
    except (ValueError, UnicodeError):
        return ""
    return str(value)


def _chip_colors(style: str) -> str:
    """Preserve legacy bias colors without accepting arbitrary CSS."""
    colors = {}
    for declaration in style.split(";"):
        name, separator, value = declaration.partition(":")
        name, value = name.strip().lower(), value.strip()
        if (separator and name in {"background-color", "border-color"}
                and re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})", value)):
            colors[name] = value
    return " ".join(f"{name}: {value};" for name, value in colors.items())


class _LegacyChips(HTMLParser):
    """Extract data from old chips; never return any source HTML as trusted markup."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.chips = []
        self.current = None
        self.text = []

    def _finish(self):
        if self.current is not None:
            self.current["label"] = "".join(self.text).strip() or "Source"
            self.chips.append(self.current)
        self.current, self.text = None, []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        self._finish()
        values = dict(attrs)
        if "chip" in (values.get("class") or "").split():
            self.current = {"url": safe_http_url(values.get("href")),
                            "style": _chip_colors(values.get("style") or "")}

    def handle_endtag(self, tag):
        if tag == "a":
            self._finish()

    def handle_data(self, data):
        if self.current is not None:
            self.text.append(data)


def legacy_chips(value) -> list[dict]:
    if not isinstance(value, str):
        return []
    parser = _LegacyChips()
    parser.feed(value)
    parser.close()
    parser._finish()
    return parser.chips
