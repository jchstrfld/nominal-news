"""Offline ranking safety checks. No pipeline import, network, or paid calls."""
import ast
import copy
import os
from pathlib import Path
import re
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'final_cohesion_check.py'
FUNCTIONS = {
    '_clamp01', '_gdelt_shadow_blended_score', '_gdelt_review_bounds',
    '_gdelt_assignments_may_change', '_gdelt_ranking_can_stop',
    '_gdelt_global_attention_from_articles', '_clusters_for_output',
    'canonicalize_url', '_cluster_sig_urls',
}


def load_checks():
    tree = ast.parse(PATH.read_text(encoding='utf-8'))
    names = {'EVENT_MAX_CALLS', '_OUTPUT_TRANSIENT_KEYS', '_TRACKING_PARAMS'}
    nodes = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    scope = dict(os=os, copy=copy, re=re, urlparse=urlparse)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), scope)

    # Exercise the actual cache/call-limit branch, stopping before publication.
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For)
                and ast.unparse(n.target) == '(queue_pos, c)')
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'original_articles')
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.AugAssign)
               and ast.unparse(n.target) == 'reviewed_candidates')
    shell = ast.parse('''
def simulate_limit(review_queue, event_cache, purify_cluster_with_gpt):
    event_calls = event_cache_hits = reviewed_candidates = 0
    purifier_stop_reason = 'queue_exhausted'
    purity_report = False
    EVENT_CACHE_VERSION = 'test'
    EVENT_MODEL = 'test'
    for queue_pos, c in enumerate(review_queue):
        pass
    return event_calls, event_cache_hits, reviewed_candidates, purifier_stop_reason
''')
    shell.body[0].body[-2].body = copy.deepcopy(loop.body[start:end + 1])
    exec(compile(ast.fix_missing_locations(shell), str(PATH), 'exec'), scope)
    return scope


class RankingBounds(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')))
        with patch.dict(os.environ):
            os.environ.pop('NN_EVENT_MAX_CALLS', None)
            self.ns = load_checks()
        self.blend = self.ns['_gdelt_shadow_blended_score']
        self.bounds = self.ns['_gdelt_review_bounds']

    def matcher(self, *signals):
        return {'status': 'OK', 'catalog': [
            {'candidate_id': f'catalog-{i}', 'global_attention': {'discovery_percentile': s}}
            for i, s in enumerate(signals)]}

    def retained(self, *signals):
        return {'eligible': bool(signals), 'candidates': [
            {'candidate_id': f'retained-{i}', 'discovery_percentile': s}
            for i, s in enumerate(signals)]}

    def test_oct29_local_fallback_can_cross_tenth_place(self):
        # Saved queue position 50: local 50/85, tentative global .1667.
        local, global_signal, cutoff = 50 / 85, .1667, 55.294
        self.assertEqual(50.393, self.blend(local, global_signal))
        bound = self.bounds(local, self.retained(global_signal), None)
        self.assertEqual(58.824, bound['score_upper'])
        self.assertGreater(bound['score_upper'], cutoff)
        self.assertFalse(self.ns['_gdelt_ranking_can_stop'](10, 20, bound['score_upper'], cutoff, [], set()))

    def test_cleaning_can_reveal_catalog_match_absent_from_tentative_options(self):
        bound = self.bounds(.3, {}, self.matcher(.1, .9))
        self.assertEqual(78, bound['score_upper'])
        self.assertEqual({'catalog-0', 'catalog-1'}, bound['candidate_ids'])

    def test_all_retained_alternatives_count_without_a_catalog(self):
        bound = self.bounds(.3, self.retained(.1, .8), {'status': 'UNAVAILABLE'})
        self.assertEqual(70, bound['score_upper'])
        self.assertEqual({'retained-0', 'retained-1'}, bound['candidate_ids'])

    def test_no_global_evidence_preserves_local_only_bound(self):
        self.assertEqual({'score_upper': 60, 'candidate_ids': set()}, self.bounds(.6, {}, None))

    def test_bounds_cover_cleaning_loss_and_all_available_global_signals(self):
        for local_upper in (0, .1, .35, .588235, .8, 1):
            for signals in ((), (.1,), (.9,), (.2, .85), (.99996,)):
                bound = self.bounds(local_upper, self.retained(*signals), self.matcher(*signals))['score_upper']
                for step in range(101):
                    cleaned_local = local_upper * step / 100
                    for final_global in (None,) + tuple(round(s, 4) for s in signals):
                        self.assertLessEqual(self.blend(cleaned_local, final_global), bound)

    def test_shared_catalog_alternative_prevents_premature_stop(self):
        approved = [{'_gdelt_catalog_match_options': [
            {'candidate_id': 'current'}, {'candidate_id': 'alternative'}]}]
        stop = self.ns['_gdelt_ranking_can_stop']
        self.assertFalse(stop(10, 20, 50, 60, approved, {'alternative'}))
        self.assertTrue(stop(10, 20, 50, 60, approved, {'unrelated'}))

    def test_shared_retained_candidate_also_prevents_assignment_changes(self):
        articles = [{'url': f'https://outlet{i}.example/a', 'gdelt_candidate_id': 'shared',
                     'gdelt_global_attention': {'discovery_percentile': .8}} for i in range(2)]
        approved = [{'related_articles': articles}]
        self.assertTrue(self.ns['_gdelt_assignments_may_change'](approved, {'shared'}))
        self.assertFalse(self.ns['_gdelt_assignments_may_change'](approved, set()))

    def test_ties_and_unsecured_local_comparison_do_not_stop(self):
        stop = self.ns['_gdelt_ranking_can_stop']
        self.assertFalse(stop(10, 10, 40, 50, [], set()))
        self.assertFalse(stop(5, 10, 50, 50, [], set()))
        self.assertFalse(stop(11, 10, 40, 50, [], set()))
        self.assertTrue(stop(5, 10, 40, 50, [], set()))

    def test_default_56_and_explicit_environment_override(self):
        self.assertEqual(56, self.ns['EVENT_MAX_CALLS'])
        with patch.dict(os.environ, {'NN_EVENT_MAX_CALLS': '12'}):
            self.assertEqual(12, load_checks()['EVENT_MAX_CALLS'])

    def test_limit_allows_56_mock_reviews_but_never_a_57th(self):
        queue = [{'articles': [{'url': f'https://outlet.example/{i}'}]} for i in range(60)]
        reviewer = Mock(return_value={'action': 'REJECT', 'event_type': 'TOPIC_WAVE'})
        result = self.ns['simulate_limit'](queue, {}, reviewer)
        self.assertEqual((56, 0, 56, 'live_call_cap_reached_before_rank_secured'), result)
        self.assertEqual(56, reviewer.call_count)

    def test_cached_decisions_do_not_spend_the_live_call_allowance(self):
        queue = [{'articles': [{'url': f'https://outlet.example/{i}'}]} for i in range(58)]
        cached = {'test::test::' + self.ns['_cluster_sig_urls'](queue[56]):
                  {'action': 'REJECT', 'event_type': 'TOPIC_WAVE'}}
        reviewer = Mock(return_value={'action': 'REJECT', 'event_type': 'TOPIC_WAVE'})
        result = self.ns['simulate_limit'](queue, cached, reviewer)
        self.assertEqual((56, 1, 57, 'live_call_cap_reached_before_rank_secured'), result)

    def test_private_bounds_are_not_published_and_scores_are_unchanged(self):
        cluster = {'attention_score': 12.5, 'articles': [],
                   '_gdelt_shadow_score_upper_bound': 100,
                   '_gdelt_shadow_possible_candidate_ids': ['candidate']}
        original = copy.deepcopy(cluster)
        self.assertEqual([{'attention_score': 12.5, 'articles': []}], self.ns['_clusters_for_output']([cluster]))
        self.assertEqual(original, cluster)
        self.assertEqual(80, self.blend(.8, None))
        self.assertEqual(68, self.blend(.8, .2))


if __name__ == '__main__':
    unittest.main()
