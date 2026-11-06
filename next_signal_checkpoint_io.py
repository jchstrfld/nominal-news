"""Opt-in checkpoint I/O: durable budget, exact receipts, one HTTP attempt."""
import json
import os
from pathlib import Path
from decimal import Decimal
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from next_signal_checkpoint_review import Budget, CONTRACT_VERSION, MODEL, ReviewError, _digest


def _read(path, default):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default
    if not isinstance(value, dict):
        raise ReviewError('Checkpoint state must be a JSON object')
    return value


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _PersistentBudget(Budget):
    def __init__(self, path, context, max_calls, max_usd):
        super().__init__(max_calls=max_calls, max_usd=max_usd)
        self.path, self.context, self.pending = Path(path), context, None
        state = _read(self.path, {})
        if state:
            if state.get('context') != context:
                raise ReviewError('Checkpoint ledger belongs to a different edition or contract')
            calls, amount = state.get('calls'), state.get('accounted_usd')
            try:
                amount = Decimal(amount)
            except (TypeError, ValueError, ArithmeticError):
                raise ReviewError('Invalid checkpoint spending record') from None
            if (type(calls) is not int or calls < 0 or not amount.is_finite() or amount < 0
                    or type(state.get('halted')) is not bool or 'pending' not in state):
                raise ReviewError('Invalid checkpoint spending record')
            self.calls, self.accounted_usd = calls, amount
            self.pending, self.halted = state['pending'], state['halted']
            if self.pending is not None or self.halted:
                raise ReviewError('Previous checkpoint attempt is unfinished or failed; inspect its ledger before any new calls')
            if self.calls > self.max_calls or self.accounted_usd > self.max_usd:
                raise ReviewError('Existing checkpoint usage exceeds the supplied limit')

    def flush(self):
        _save(self.path, {'context': self.context, 'calls': self.calls,
                         'accounted_usd': str(self.accounted_usd),
                         'max_calls': self.max_calls, 'max_usd': str(self.max_usd),
                         'halted': self.halted, 'pending': self.pending})

    def reserve(self, request):
        if self.pending is not None:
            raise ReviewError('An earlier checkpoint request has not been saved completely')
        amount = super().reserve(request)
        self.pending = {'request_digest': _digest(request), 'reservation_usd': str(amount),
                        'sent': False, 'settled': False}
        self.flush()  # Durable reservation before the network can be reached.
        return amount

    def settle(self, reservation, usage):
        try:
            super().settle(reservation, usage)
            self.pending['settled'] = True
        finally:
            self.flush()

    def saved_receipt(self, value):
        if (self.pending is None or not self.pending['settled']
                or self.pending['request_digest'] != value.get('request_digest')):
            raise ReviewError('Receipt does not match the reserved checkpoint request')
        self.pending = None
        self.flush()


class _ReceiptCache(dict):
    def __init__(self, path, budget):
        self.path, self.budget = Path(path), budget
        super().__init__(_read(path, {}))

    def __setitem__(self, key, value):
        updated = dict(self)
        updated[key] = value
        _save(self.path, updated)
        super().__setitem__(key, value)
        self.budget.saved_receipt(value)


class CheckpointSession:
    """One explicit live session. Never logs credentials or retries failures.

    A process lock protects the shared receipt cache and cumulative ledger.
    A failed/unfinished request keeps its reservation and blocks an automatic
    restart. Limits are supplied by the caller; zero spending is the default.
    """
    def __init__(self, *, edition, cache_file, ledger_file, max_calls, max_usd, api_key):
        self._locks = []
        self.budget = None
        if not isinstance(api_key, str) or not api_key.strip():
            raise ReviewError('A configured API key is required for explicit live checkpoint review')
        self._key = api_key
        if Path(cache_file).resolve() == Path(ledger_file).resolve():
            raise ReviewError('Checkpoint receipts and spending ledger require different paths')
        try:
            for target in sorted({str(Path(cache_file).resolve()), str(Path(ledger_file).resolve())}):
                lock_path = Path(target + '.lock')
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                handle = lock_path.open('a+b')
                self._locks.append(handle)
                if lock_path.stat().st_size == 0:
                    handle.write(b'0')
                    handle.flush()
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            context = {'edition': edition, 'model': MODEL, 'contract': CONTRACT_VERSION}
            self.budget = _PersistentBudget(ledger_file, context, max_calls, max_usd)
            self.cache = _ReceiptCache(cache_file, self.budget)
            self._opener = build_opener(_NoRedirect())
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        try:
            if self.budget is not None and (self.budget.calls or self.budget.path.exists()):
                self.budget.flush()
        finally:
            for handle in self._locks:
                handle.close()  # OS releases the lock, including on crashes.
            self._locks = []
            self._key = None

    def send(self, request):
        pending = self.budget.pending
        if (pending is None or pending['sent'] or pending['request_digest'] != _digest(request)
                or set(request) != {'model','reasoning_effort','max_completion_tokens','messages','response_format'}):
            raise ReviewError('Unreserved or repeated checkpoint request')
        pending['sent'] = True
        self.budget.flush()
        # Standard service tier keeps the configured token rates applicable.
        # It changes billing/latency, not the model's prompt or review contract.
        body = json.dumps({**request, 'service_tier': 'default'}, ensure_ascii=False, allow_nan=False).encode('utf-8')
        req = Request('https://api.openai.com/v1/chat/completions', data=body, method='POST',
                      headers={'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json'})
        try:
            with self._opener.open(req, timeout=45) as response:
                data = response.read(1_000_001)
                if len(data) > 1_000_000:
                    raise ReviewError('Checkpoint response exceeded the size limit')
            raw = json.loads(data)
        except HTTPError as exc:
            raise ReviewError(f'Checkpoint request failed with HTTP {exc.code}; no retry') from None
        except (URLError, OSError, ValueError):
            raise ReviewError('Checkpoint transport failed; reservation retained and no retry') from None
        if not isinstance(raw, dict) or raw.get('service_tier') not in (None, 'default'):
            raise ReviewError('Unexpected checkpoint response or service tier')
        choices = raw.get('choices')
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ReviewError('Invalid checkpoint response choices')
        choice = choices[0]
        message = choice.get('message')
        if not isinstance(message, dict):
            raise ReviewError('Invalid checkpoint response message')
        try:
            parsed = json.loads(message.get('content'))
        except (TypeError, ValueError):
            parsed = None  # Preserve usage; the reviewer rejects this response.
        return {'response': parsed, 'usage': raw.get('usage'),
                'finish_reason': choice.get('finish_reason'), 'refusal': message.get('refusal')}
