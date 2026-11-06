"""Offline market-period checks. Synthetic prices, no models or HTTP calls."""
import ast
from collections import Counter, defaultdict
import contextlib
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
import io
import json
import os
from pathlib import Path
import tempfile
import types
from typing import Any, Dict, Iterable, List, Optional, Tuple
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_market(path=None):
    """Load pure definitions without importing or running the daily generator."""
    path = Path(path) if path else ROOT / 'generate_market_overview.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(UTC=UTC, date=date, datetime=datetime, timedelta=timedelta, np=np,
              Any=Any, Dict=Dict, Iterable=Iterable, List=List, Optional=Optional,
              Tuple=Tuple, Path=Path, Counter=Counter, defaultdict=defaultdict,
              json=json, os=os, BASE_DIR=ROOT,
              MARKET_FLAT_THRESHOLD=.15, MARKET_OVERALL_FLAT_THRESHOLD=.20,
              MARKET_MIXED_ACTIVITY_THRESHOLD=.55, MARKET_SHARP_THRESHOLD=1.20)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MARKET_SERIES' for t in node.targets):
            ns['MARKET_SERIES'] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


class MarketAlignmentChecks(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('socket.socket.connect', side_effect=AssertionError('Offline only')))
        self.ns = load_market()

    def build(self, histories, edition='2034-10-20'):
        def fetch(series, day, lookback):
            self.assertEqual(edition, day)
            self.assertEqual(20, lookback)
            key = next(k for k, v in self.ns['MARKET_SERIES'].items() if v == series)
            value = histories.get(key)
            if isinstance(value, Exception):
                raise value
            return value or []
        get = Mock(side_effect=fetch)
        self.ns['_fetch_fred_series'] = get
        result = self.ns['_build_market_context'](edition)
        self.assertEqual(3, get.call_count)  # Same free data requests, no retries.
        return result

    def histories(self):
        return {
            'nasdaq': [('2034-10-17', 100), ('2034-10-18', 102), ('2034-10-19', 90)],
            'sp500': [('2034-10-17', 100), ('2034-10-18', 102)],
            'dow': [('2034-10-17', 100), ('2034-10-18', 102), ('2034-10-19', 90)],
        }

    def test_lagging_index_aligns_all_returns_and_date_instead_of_mixing_directions(self):
        indexes, state, stats, close = self.build(self.histories())
        self.assertEqual('2034-10-18', close)
        self.assertEqual('strong_up', state)
        self.assertEqual(2., stats['average_pct_change'])
        self.assertEqual({('2034-10-17', '2034-10-18')},
                         {(x['previous_close_date'], x['close_date']) for x in indexes.values()})
        self.assertEqual({2.}, {x['pct_change'] for x in indexes.values()})

    def test_same_close_but_different_previous_sessions_cannot_be_combined(self):
        rows = self.histories()
        rows['nasdaq'] = [('2034-10-17', 100), ('2034-10-19', 103)]
        rows['sp500'] = [('2034-10-18', 100), ('2034-10-19', 101)]
        indexes, state, stats, close = self.build(rows)
        self.assertEqual('unknown', state)
        self.assertIsNone(close)
        self.assertEqual({'average_pct_change': None, 'average_abs_pct_change': None}, stats)
        self.assertTrue(all(x['pct_change'] is None for x in indexes.values()))

    def test_an_older_complete_pair_survives_a_later_missing_session(self):
        rows = {k: [('2034-10-16', 100), ('2034-10-17', 101), ('2034-10-19', 103)] for k in self.ns['MARKET_SERIES']}
        rows['dow'].insert(2, ('2034-10-18', 102))
        self.assertEqual('2034-10-17', self.build(rows)[3])

    def test_weekends_holidays_and_year_boundaries_use_observed_sessions(self):
        for previous, close in [('2034-10-20', '2034-10-23'), ('2034-10-20', '2034-10-24'),
                                ('2033-12-30', '2034-01-03')]:
            with self.subTest(previous=previous, close=close):
                rows = {k: [(previous, 100), (close, 101)] for k in self.ns['MARKET_SERIES']}
                indexes, state, _, actual_close = self.build(rows, close)
                self.assertEqual(close, actual_close)
                self.assertEqual('up', state)
                self.assertEqual({previous}, {x['previous_close_date'] for x in indexes.values()})

    def test_one_unavailable_feed_retains_existing_two_index_minimum(self):
        rows = self.histories()
        rows['sp500'] = RuntimeError('Saved failure control')
        indexes, state, _, close = self.build(rows)
        self.assertEqual('2034-10-19', close)
        self.assertEqual('strong_down', state)
        self.assertIsNone(indexes['sp500']['pct_change'])
        self.assertEqual('Saved failure control', indexes['sp500']['error'])
        rows['dow'] = []
        self.assertEqual('unknown', self.build(rows)[1])

    def test_future_observations_and_row_order_do_not_move_the_edition_forward(self):
        rows = self.histories()
        rows['sp500'].append(('2034-10-21', 110))
        for values in rows.values():
            values.reverse()
        before = deepcopy(rows)
        self.assertEqual('2034-10-18', self.build(rows)[3])
        self.assertEqual(before, rows)

    def test_bad_prices_and_conflicting_duplicates_never_bridge_a_known_session(self):
        for bad in (0, -1, float('nan'), float('inf'), 'missing'):
            rows = {k: [('2034-10-17', 100), ('2034-10-18', 101), ('2034-10-19', 102)] for k in self.ns['MARKET_SERIES']}
            rows['sp500'][1] = ('2034-10-18', bad)
            with self.subTest(bad=bad):
                self.assertEqual('unknown', self.build(rows)[1])
        rows = {k: [('2034-10-17', 100), ('2034-10-18', 101)] for k in self.ns['MARKET_SERIES']}
        rows['sp500'].extend([('2034-10-18', 102), ('2034-10-18', 101)])
        indexes, state, _, _ = self.build(rows)
        self.assertIsNone(indexes['sp500']['pct_change'])
        self.assertEqual('up', state)  # The other two intact series still qualify.

    def test_partial_feed_still_exposes_a_missing_intermediate_session(self):
        rows = {k: [('2034-10-17', 100), ('2034-10-19', 102)] for k in ('nasdaq', 'dow')}
        rows['sp500'] = [('2034-10-18', 101)]
        self.assertEqual('unknown', self.build(rows)[1])

    def test_aggregation_rejects_mixed_periods_and_unproven_legacy_returns(self):
        indexes = {k: {'pct_change': 2., 'previous_close_date': '2034-10-17', 'close_date': '2034-10-18'} for k in self.ns['MARKET_SERIES']}
        for changed in ({'close_date': '2034-10-19'}, {'previous_close_date': '2034-10-16'},
                        {'previous_close_date': None}, {'pct_change': float('nan')}):
            sample = deepcopy(indexes)
            sample['sp500'].update(changed)
            with self.subTest(changed=changed):
                self.assertEqual('unknown', self.ns['_overall_market_state'](sample))
                self.assertIsNone(self.ns['_market_stats'](sample)['average_pct_change'])
                self.assertIsNone(self.ns['_common_close_date'](sample))

    def test_aligned_returns_keep_existing_market_thresholds(self):
        for values, expected in [([2, 1.5, 1], 'strong_up'), ([-3, -2, -1], 'strong_down'),
                                 ([.4, .5, .6], 'up'), ([-.4, -.5, -.6], 'down'),
                                 ([0, .1, -.1], 'flat'), ([2, -2, 0], 'mixed')]:
            indexes = {k: {'pct_change': v, 'previous_close_date': '2034-10-17', 'close_date': '2034-10-18'}
                       for k,v in zip(self.ns['MARKET_SERIES'], values)}
            with self.subTest(values=values):
                self.assertEqual(expected, self.ns['_overall_market_state'](indexes))

    def test_main_publishes_aligned_date_and_preserves_other_card_fields(self):
        for histories in (self.histories(), {}):
            self.build(histories)  # Install the saved/synthetic fetch boundary.
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'articles_with_bias_2034-10-20.json').write_text('[]', encoding='utf-8')
                write, explain = Mock(), Mock(return_value=('Saved explanation', 'Saved implications'))
                self.ns.update(BASE_DIR=root, _parse_args=lambda: types.SimpleNamespace(date='2034-10-20'),
                    _write_json=write, _build_inflation_signal=lambda day: {'value': 'frozen inflation'},
                    _build_jobs_signal=lambda day: {'value': 'frozen jobs'},
                    _build_rate_signal=lambda day: ({'value': 'frozen rate'}, []),
                    _build_secondary_context=lambda day: {}, _semantic_evidence=lambda articles: {},
                    _official_calendar_events=lambda day: ([], {'complete': True, 'sources': {}}),
                    _next_signal=lambda *a, **kw: None, _what_changed=lambda **kw: 'Saved change',
                    _synthesize_explainer=explain)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.ns['main']()
                out = write.call_args.args[1]
                self.assertEqual('frozen inflation', out['signals']['inflation']['value'])
                self.assertEqual('frozen jobs', out['signals']['jobs']['value'])
                self.assertEqual('frozen rate', out['signals']['rates']['value'])
                self.assertEqual('Saved change', out['what_changed'])
                expected_date = '2034-10-18' if histories else None
                self.assertEqual(expected_date, out['signals']['markets']['as_of'])
                self.assertEqual(expected_date, explain.call_args.kwargs['market_close_date'])
                if not histories:
                    self.assertEqual('Unavailable', out['signals']['markets']['display_value'])
                    self.assertEqual('Market trend unavailable', out['signals']['markets']['trend_label'])

    def test_unavailable_pulse_is_not_described_to_the_model_as_a_small_move(self):
        api = Mock(return_value={'choices': [{'message': {'content': '{"what_happened":"Saved control","what_it_means":"Saved control"}'}}]})
        self.ns.update(openai=types.SimpleNamespace(ChatCompletion=types.SimpleNamespace(create=api)),
                       MARKET_MODEL='offline-placeholder', MARKET_MODEL_FALLBACK='unused',
                       _latest_release_date=lambda *args: None)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-placeholder'}):
            self.ns['_synthesize_explainer']('2034-10-20', 'unknown', None, {}, {}, {}, {}, {}, [])
        prompt = api.call_args.kwargs['messages'][1]['content']
        self.assertIn('sufficient aligned index data', prompt)
        self.assertNotIn('The market move was not unusually large', prompt)
        api.assert_called_once()


if __name__ == '__main__':
    unittest.main()
