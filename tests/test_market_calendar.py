"""Offline calendar failure/selection checks; no pipeline run or paid calls."""
import ast
import contextlib
from datetime import UTC, date, datetime, timedelta
import html
import io
import json
from pathlib import Path
import re
import tempfile
import types
from typing import Any, Dict, List, Optional, Tuple
import unittest
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = {
    '_target_date', '_unfold_ics', '_parse_ics_date', '_fetch_bls_events',
    '_fetch_bea_events', '_fetch_fed_events', '_official_calendar_events',
    '_calendar_status', '_normalize_event_title', '_next_signal', 'main',
}


def load_calendar():
    path = ROOT / 'generate_market_overview.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    scope = dict(UTC=UTC, date=date, datetime=datetime, timedelta=timedelta,
                 html=html, re=re, Any=Any, Dict=Dict, List=List, Optional=Optional,
                 Tuple=Tuple, requests=requests, BASE_DIR=ROOT,
                 BLS_ICS_URL='https://www.bls.gov/schedule/news_release/bls.ics',
                 BEA_RELEASE_JSON_URL='https://apps.bea.gov/API/signup/release_dates.json',
                 FED_MONETARY_POLICY_URL='https://www.federalreserve.gov/monetarypolicy.htm',
                 REQUEST_HEADERS={'User-Agent': 'NominalNews/1.0 market-economy-overview'})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


class MarketCalendarChecks(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')))
        self.get = self.enterContext(patch('requests.get', side_effect=AssertionError('Unexpected HTTP call')))
        self.ns = load_calendar()

    def schedules(self, year=2026):
        return {
            'BLS': [{'date': date(year, 11, 6), 'title': 'Employment Situation', 'source': 'BLS'},
                    {'date': date(year, 11, 10), 'title': 'Consumer Price Index', 'source': 'BLS'}],
            'BEA': [{'date': date(year, 11, 25), 'title': 'Gross Domestic Product', 'source': 'BEA'},
                    {'date': date(year, 11, 25), 'title': 'Personal Income and Outlays', 'source': 'BEA'}],
            'Federal Reserve': [{'date': date(year, 12, 9), 'title': 'Federal Reserve rate decision', 'source': 'Federal Reserve'}],
        }

    def select(self, day, schedules):
        status = self.ns['_calendar_status'](day, schedules)
        rows = [r for group in schedules.values() for r in (group or [])]
        return self.ns['_next_signal'](day, rows, calendar_status=status), status

    def responses(self, *, bls_failure=False, bls_text=None):
        bls = Mock(text=bls_text if bls_text is not None else (
            'BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nDTSTART;TZID=America/New_York:20261106T083000\r\n'
            'SUMMARY:Employment \r\n Situation\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n'))
        if bls_failure:
            bls.raise_for_status.side_effect = requests.HTTPError('403')
        bea = Mock()
        bea.json.return_value = {
            'Gross Domestic Product': {'release_dates': ['2026-11-25T08:30:00']},
            'Personal Income and Outlays': {'release_dates': ['2026-11-25T08:30:00']},
        }
        fed = Mock(text='<p>Dec. 8-9 FOMC Meeting</p>')
        by_url = {self.ns['BLS_ICS_URL']: bls, self.ns['BEA_RELEASE_JSON_URL']: bea,
                  self.ns['FED_MONETARY_POLICY_URL']: fed}
        self.get.side_effect = lambda url, **kwargs: by_url[url]
        return by_url

    def test_oct29_bls_403_cannot_promote_nov25_as_next(self):
        self.responses(bls_failure=True)
        rows, status = self.ns['_official_calendar_events']('2026-10-29')
        self.assertTrue(any(r['date'] == date(2026, 11, 25) for r in rows))
        self.assertEqual('unavailable_or_unusable', status['sources']['BLS'])
        self.assertIsNone(self.ns['_next_signal']('2026-10-29', rows, calendar_status=status))
        self.assertEqual(3, self.get.call_count)  # No new retry/call loop.

    def test_healthy_fetches_keep_the_earliest_report_and_feed_parsers(self):
        self.responses()
        rows, status = self.ns['_official_calendar_events']('2026-10-29')
        result = self.ns['_next_signal']('2026-10-29', rows, calendar_status=status)
        self.assertTrue(status['complete'])
        self.assertEqual({'date': '2026-11-06', 'date_label': 'Nov 6', 'label': 'Jobs Report', 'source': 'BLS'}, result)

    def test_any_missing_agency_withholds_across_years(self):
        for year in (2026, 2027, 2031):
            day = f'{year}-10-29'
            for source in self.schedules(year):
                for missing in ([], None):
                    with self.subTest(year=year, source=source, missing=missing):
                        schedules = self.schedules(year)
                        schedules[source] = missing
                        result, status = self.select(day, schedules)
                        self.assertIsNone(result)
                        self.assertFalse(status['complete'])
        self.get.assert_not_called()

    def test_expired_or_irrelevant_calendar_does_not_establish_coverage(self):
        for rows in ([{'date': date(2026, 10, 29), 'title': 'Employment Situation'}],
                     [{'date': date(2026, 9, 6), 'title': 'Employment Situation'}],
                     [{'date': date(2026, 11, 3), 'title': 'Unrelated release'}]):
            schedules = self.schedules()
            schedules['BLS'] = rows
            result, status = self.select('2026-10-29', schedules)
            self.assertIsNone(result)
            self.assertEqual('no_usable_future_events', status['sources']['BLS'])

    def test_success_status_with_html_or_invalid_calendar_is_unusable(self):
        for text in ('<html>Access denied</html>', 'BEGIN:VCALENDAR\nEND:VCALENDAR',
                     'BEGIN:VEVENT\nDTSTART:20261340\nSUMMARY:Employment Situation\nEND:VEVENT'):
            with self.subTest(text=text):
                self.responses(bls_text=text)
                rows, status = self.ns['_official_calendar_events']('2026-10-29')
                self.assertFalse(status['complete'])
                self.assertIsNone(self.ns['_next_signal']('2026-10-29', rows, calendar_status=status))

    def test_http_timeout_and_invalid_json_fail_closed(self):
        for url in ('BLS_ICS_URL', 'BEA_RELEASE_JSON_URL', 'FED_MONETARY_POLICY_URL'):
            with self.subTest(url=url):
                responses = self.responses()
                responses[self.ns[url]].raise_for_status.side_effect = requests.Timeout()
                rows, status = self.ns['_official_calendar_events']('2026-10-29')
                self.assertIsNone(self.ns['_next_signal']('2026-10-29', rows, calendar_status=status))
        responses = self.responses()
        responses[self.ns['BEA_RELEASE_JSON_URL']].json.side_effect = ValueError('Invalid JSON')
        _, status = self.ns['_official_calendar_events']('2026-10-29')
        self.assertFalse(status['complete'])

    def test_status_must_cover_every_agency_and_the_same_edition(self):
        schedules = self.schedules()
        rows = [r for group in schedules.values() for r in group]
        for status in ({}, {'complete': True},
                       {'date': '2026-10-29', 'complete': True, 'sources': {'BEA': 'ok'}},
                       self.ns['_calendar_status']('2026-10-28', schedules)):
            self.assertIsNone(self.ns['_next_signal']('2026-10-29', rows, calendar_status=status))

    def test_year_boundary_same_day_grouping_and_30_day_horizon(self):
        schedules = {source: [{**row, 'date': row['date'].replace(year=2032, month=1)} for row in rows]
                     for source, rows in self.schedules(2031).items()}
        result, status = self.select('2031-12-29', schedules)
        self.assertEqual('2032-01-06', result['date'])
        schedules = self.schedules()
        schedules['BLS'] = [{**r, 'date': r['date'].replace(month=12)} for r in schedules['BLS']]
        result, _ = self.select('2026-10-29', schedules)
        self.assertEqual('GDP + Consumer Spending & Inflation', result['label'])
        result, status = self.select('2026-09-01', schedules)
        self.assertTrue(status['complete'])
        self.assertIsNone(result)

    def test_daily_market_entrypoint_uses_guard_without_changing_narrative(self):
        self.responses(bls_failure=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as folder:
            base = Path(folder)
            (base / 'articles_with_bias_2026-10-29.json').write_text('[]', encoding='utf-8')
            write = Mock()
            self.ns.update(BASE_DIR=base, _parse_args=lambda: types.SimpleNamespace(date='2026-10-29'),
                _read_json=lambda path: [], _write_json=write,
                _build_market_context=lambda day: ({}, 'flat', {'average_pct_change': 0, 'average_abs_pct_change': 0}, None),
                _build_inflation_signal=lambda day: {}, _build_jobs_signal=lambda day: {},
                _build_rate_signal=lambda day: ({}, []), _build_secondary_context=lambda day: {},
                _semantic_evidence=lambda articles: {}, _what_changed=lambda **kw: 'Original change text',
                _synthesize_explainer=lambda **kw: ('Original explanation', 'Original implications'),
                _market_state_label=lambda state: 'Steady', _human_date_with_weekday=lambda day: '')
            with contextlib.redirect_stdout(io.StringIO()):
                self.ns['main']()
            out = write.call_args.args[1]
            self.assertIsNone(out['next_signal'])
            self.assertFalse(out['calendar_status']['complete'])
            self.assertEqual('Original explanation', out['what_happened'])
            self.assertEqual('Original implications', out['what_it_means'])
            self.assertEqual('Original change text', out['what_changed'])


if __name__ == '__main__':
    unittest.main()
