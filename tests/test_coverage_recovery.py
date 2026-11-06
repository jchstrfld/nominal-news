"""Offline evidence-recovery and publication-boundary checks."""
import ast
from collections import Counter
import copy
import hashlib
from html import unescape
import json
from math import log2
from pathlib import Path
import re
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def load_checks():
    path = ROOT / 'final_cohesion_check.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'_coverage_recovery_plans', '_recover_publication_evidence', '_review_attention_upper',
             '_publication_cache_key', '_publication_excerpt', 'canonicalize_url', 'domain_from_url',
             'attention_score', 'attention_metadata_from_articles', 'source_diversity', '_clusters_for_output'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                 {'_TRACKING_PARAMS', '_OUTPUT_TRANSIENT_KEYS'} for t in n.targets)]
    ns = dict(copy=copy, hashlib=hashlib, json=json, re=re, unescape=unescape,
              urlparse=urlparse, Counter=Counter, log2=log2, EVENT_CACHE_VERSION='test',
              EVENT_MODEL='same-model', EVENT_SUPPORT_MODEL='same-model')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def article(i, cid=None):
    row = {'title': 'Council approved the river bridge permit.', 'description': '',
           'url': f'https://outlet{i}.test/permit', 'source': f'outlet{i}.test', 'bias': 'Unknown'}
    if cid:
        row.update(gdelt_candidate_id=cid, gdelt_writeup_family_id=f'family{i}')
    return row


def audit(date='2030-02-03', cid='candidate', count=12):
    families = []
    for i in range(count):
        a = article(i)
        r = dict(url=a['url'], title=a['title'], domain=a['source'], target_date=date,
                 displayable_english=True, writeup_family_id=f'family{i}',
                 writeup_family_outlet_count=1, day_basis='TARGET_EXPLICIT')
        families.append(dict(family_id=f'family{i}', representative_url=r['url'],
                             representative_title=r['title'], day_role='TARGET_EXPLICIT', receipts=[r]))
    return {'date': date, 'candidates': [{'candidate_id': cid, 'writeup_families': families}],
            'global_source_receipt_catalog': {'candidates': {cid: {'target_date': date,
                'writeup_families': families, 'receipts': [f['receipts'][0] for f in families]}}}}


class CoverageRecovery(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('socket.socket.connect', side_effect=AssertionError('Offline only')))
        self.ns = load_checks()
        self.groups = [{'articles': [article(i, 'candidate') for i in range(4)]},
                       {'articles': [article(i) for i in range(4, 7)]}]
        self.audit = audit()
        self.key = self.ns['_publication_cache_key']
        self.reject = {'action': 'REJECT', 'event_type': 'DISCRETE_EVENT',
                       'event': 'Council approved the river bridge permit.',
                       'why': 'Insufficient independently supported core'}

    def plans(self, groups=None, data=None):
        return self.ns['_coverage_recovery_plans'](groups or self.groups,
            data or self.audit, '2030-02-03', {'outlet7.test': 'Center'})

    def test_split_evidence_is_pooled_but_no_source_is_accepted(self):
        before = copy.deepcopy((self.groups, self.audit))
        plans = self.plans()
        left, right = [plans[self.key(g['articles'])] for g in self.groups]
        self.assertEqual(left, right)
        self.assertEqual(11, len(left['articles']))
        self.assertEqual(before, (self.groups, self.audit))
        self.assertNotIn('core_indexes', left)
        new = left['articles'][7:]
        self.assertEqual(4, len({a['gdelt_writeup_family_id'] for a in new}))
        self.assertEqual('Center', new[0]['bias'])
        self.assertTrue(all(a['published_date'] == '' for a in new))
        self.assertTrue(all(a['gdelt_observed_date'] == '2030-02-03' for a in new))

    def test_matching_topic_without_url_or_provenance_is_not_a_link(self):
        group = {'articles': [article(i) for i in range(30, 34)]}
        self.assertEqual({}, self.plans([group]))

    def test_wrong_date_missing_catalog_and_ambiguous_links_do_not_recover(self):
        for data in (audit('2030-02-04'), {'date': '2030-02-03'},
                     dict(self.audit, global_source_receipt_catalog={'candidates': []})):
            self.assertEqual({}, self.plans(data=data))
        data = copy.deepcopy(self.audit)
        extra = audit(cid='other')
        data['candidates'] += extra['candidates']
        data['global_source_receipt_catalog']['candidates'].update(extra['global_source_receipt_catalog']['candidates'])
        self.assertEqual({}, self.plans(data=data))

    def test_oversized_union_and_conflicting_source_versions_are_not_silently_trimmed(self):
        groups = [{'articles': [article(i, 'candidate') for i in range(21)]}]
        self.assertEqual({}, self.plans(groups))
        groups = copy.deepcopy(self.groups)
        conflict = copy.deepcopy(groups[0]['articles'][0]); conflict['title'] = 'Permit not approved.'
        groups[1]['articles'].append(conflict)
        self.assertEqual({}, self.plans(groups))

    def test_additions_exclude_wrong_date_language_family_and_duplicate_outlets(self):
        data = copy.deepcopy(self.audit)
        families = data['global_source_receipt_catalog']['candidates']['candidate']['writeup_families']
        families[7]['receipts'][0]['target_date'] = '2029-01-01'
        families[8]['receipts'][0]['displayable_english'] = False
        families[9]['receipts'][0]['writeup_family_id'] = 'forged'
        families[10]['receipts'][0]['url'] = article(0)['url'] + '/another'
        packet = next(iter(self.plans(data=data).values()))['articles']
        self.assertEqual(8, len(packet))
        self.assertEqual('family11', packet[-1]['gdelt_writeup_family_id'])

    def test_no_new_evidence_and_approved_uncertain_or_failed_reviews_are_not_retried(self):
        plan = next(iter(self.plans().values()))
        api = self.ns['purify_cluster_with_gpt'] = Mock()
        for result in ({'action': 'KEEP'}, {'action': 'ERROR'},
                       dict(self.reject, evidence_status='UNCERTAIN'),
                       dict(self.reject, review_incomplete=True),
                       dict(self.reject, why='No qualifying discrete current event')):
            output = self.ns['_recover_publication_evidence'](result, self.groups[0]['articles'], plan, {})
            self.assertEqual(result, output[0]); self.assertEqual({}, output[2])
        self.ns['_recover_publication_evidence'](self.reject, plan['articles'], plan, {})
        api.assert_not_called()

    def test_recovery_uses_unchanged_review_then_caches_url_membership_once(self):
        plan = next(iter(self.plans().values()))
        api = self.ns['purify_cluster_with_gpt'] = Mock(return_value={
            'action': 'CLEAN', 'event_type': 'DISCRETE_EVENT', 'core_indexes': [1, 2, 8, 9],
            'related_indexes': [1, 2, 8, 9], 'event': 'Council approved the permit.'})
        cache = {}; attempts = {}
        first = self.ns['_recover_publication_evidence'](self.reject, self.groups[0]['articles'], plan, cache, attempts=attempts)
        second = self.ns['_recover_publication_evidence'](self.reject, self.groups[1]['articles'], plan, cache, attempts=attempts)
        api.assert_called_once_with(plan['articles'], recovery=True, recovery_event=self.reject['event'])
        self.assertEqual(4, len(first[0]['core_urls']))
        self.assertEqual(first[0], second[0])
        self.assertEqual(0, second[2]['live_reviews'])
        # A new process resumes using the expanded packet's cache, not a source
        # count inherited from either original packet.
        third = self.ns['_recover_publication_evidence'](self.reject, self.groups[0]['articles'], plan, cache, 0)
        self.assertTrue(third[2]['cache_hit']); api.assert_called_once()

    def test_budget_and_review_errors_remain_unresolved_without_repeating(self):
        plan = next(iter(self.plans().values()))
        api = self.ns['purify_cluster_with_gpt'] = Mock(return_value={
            'action': 'ERROR', 'why': 'Budget reached', 'core_indexes': []})
        cache = {}; attempts = {}
        result, _, diag = self.ns['_recover_publication_evidence'](self.reject, self.groups[0]['articles'], plan, cache, 0)
        self.assertEqual('ERROR', result['action']); self.assertEqual(0, diag['live_reviews'])
        api.assert_not_called()
        for group in self.groups:
            result, _, _ = self.ns['_recover_publication_evidence'](self.reject, group['articles'], plan, cache, attempts=attempts)
            self.assertEqual('ERROR', result['action'])
        api.assert_called_once(); self.assertEqual({}, cache)

    def test_recovery_failure_cannot_fall_back_to_an_approval_or_retry_loop(self):
        plan = next(iter(self.plans().values()))
        api = self.ns['purify_cluster_with_gpt'] = Mock(return_value=self.reject)
        result, packet, _ = self.ns['_recover_publication_evidence'](self.reject, self.groups[0]['articles'], plan, {})
        again = self.ns['_recover_publication_evidence'](result, packet, plan, {})
        self.assertEqual('REJECT', again[0]['action']); api.assert_called_once()

    def test_expanded_evidence_raises_stopping_bound_without_altering_score_or_output(self):
        plan = next(iter(self.plans().values()))
        group = dict(self.groups[0], attention_score=10, _coverage_recovery_plan=plan)
        before = copy.deepcopy(group)
        self.assertGreater(self.ns['_review_attention_upper'](group), 10)
        self.assertEqual(before, group)
        output = self.ns['_clusters_for_output']([group])[0]
        self.assertEqual(10, output['attention_score'])
        self.assertNotIn('_coverage_recovery_plan', output)

    def test_recovery_keeps_model_and_schema_but_requires_own_source_stage_evidence(self):
        from test_core_evidence import load_checks as core_checks
        api = Mock(return_value=SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason='stop', message={'content':'{}', 'refusal':None})]))
        ns = core_checks(api)
        excerpts = ns['_publication_excerpts'](self.groups[0]['articles'])
        candidate = {'event':'Council approved the river bridge permit.',
                     'anchor_id':'1:title', 'support_target':'DEVELOPMENT'}
        for recovery in (False, True):
            with self.assertRaises(ValueError):
                ns['_verify_one_event'](excerpts, candidate, recovery=recovery)
        ordinary, expanded = [call.kwargs for call in api.call_args_list]
        for key in ('model','response_format','reasoning_effort','max_completion_tokens'):
            self.assertEqual(ordinary[key], expanded[key])
        message = expanded['messages'][1]['content']
        self.assertIn('does NOT establish completion', message)
        self.assertIn("article's own fields", message)
        self.assertNotIn('expanded evidence packet', ordinary['messages'][1]['content'])

    def test_recovery_cannot_reuse_a_generic_whole_packet_approval(self):
        plan = next(iter(self.plans().values()))
        api = self.ns['purify_cluster_with_gpt'] = Mock(return_value=self.reject)
        cache = {self.key(plan['articles']): {'action':'KEEP','event_type':'DISCRETE_EVENT',
                                             'core_indexes':[1,2,3,4]}}
        result, _, diag = self.ns['_recover_publication_evidence'](
            self.reject, self.groups[0]['articles'], plan, cache)
        self.assertEqual('REJECT', result['action'])
        self.assertFalse(diag['cache_hit']); api.assert_called_once()

    def test_different_fixed_events_never_share_a_recovery_decision(self):
        plan = next(iter(self.plans().values()))
        api = self.ns['purify_cluster_with_gpt'] = Mock(return_value=self.reject)
        cache = {}; attempts = {}
        first = self.ns['_recover_publication_evidence'](self.reject, self.groups[0]['articles'], plan, cache, attempts=attempts)
        other = dict(self.reject, event='Council will consider a river bridge permit.')
        second = self.ns['_recover_publication_evidence'](other, self.groups[0]['articles'], plan, cache, attempts=attempts)
        self.assertEqual(2, api.call_count)
        self.assertNotEqual(first[2]['cache_key'], second[2]['cache_key'])

    def test_recovery_selector_is_bound_to_original_event_and_requires_matching_anchor(self):
        from test_core_evidence import load_checks as core_checks
        import os
        api = Mock(return_value=SimpleNamespace(usage=None,choices=[SimpleNamespace(finish_reason='stop',
            message={'content':'{}','refusal':None})]))
        with patch.dict(os.environ,{'OPENAI_API_KEY':'offline'}):
            ns = core_checks(api)
            ns['purify_cluster_with_gpt'](self.groups[0]['articles'], recovery=True, recovery_event=self.reject['event'])
        request = api.call_args.kwargs
        schema = request['response_format']['json_schema']['schema']['properties']['candidates']
        self.assertEqual(1, schema['maxItems'])
        self.assertEqual([self.reject['event']], schema['items']['properties']['event']['enum'])
        self.assertIn('anchor establishing completion',request['messages'][1]['content'])

    def test_out_of_scope_recovery_proposal_is_rejected_before_support_review(self):
        from test_core_evidence import load_checks as core_checks
        import os
        reply = {'stated_events': {f'{i}:title':['Council approved the permit.'] for i in range(1,5)},
                 'candidates':[{'event':'Council requested a permit.', 'anchor_id':'1:title',
                                'core_excerpt_ids':['1:title'], 'related_indexes':[1,2,3,4]}]}
        api = Mock(return_value=SimpleNamespace(usage=None,choices=[SimpleNamespace(finish_reason='stop',
            message={'content':json.dumps(reply),'refusal':None})]))
        with patch.dict(os.environ,{'OPENAI_API_KEY':'offline'}):
            ns = core_checks(api)
            result = ns['purify_cluster_with_gpt'](self.groups[0]['articles'], recovery=True,
                                                   recovery_event=self.reject['event'])
        self.assertEqual('ERROR',result['action'])
        self.assertEqual('Recovery attempted to change the fixed event',result['why'])
        api.assert_called_once()

    def test_daily_loop_applies_recovery_before_membership_and_bounds_include_it(self):
        tree = ast.parse((ROOT/'final_cohesion_check.py').read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls = {name: [n.lineno for n in ast.walk(main) if isinstance(n,ast.Call)
                         and isinstance(n.func,ast.Name) and n.func.id==name]
                 for name in ['_coverage_recovery_plans','_recover_publication_evidence',
                              '_review_attention_upper','filter_related_articles_to_core']}
        self.assertLess(calls['_coverage_recovery_plans'][0], calls['_recover_publication_evidence'][0])
        self.assertLess(calls['_recover_publication_evidence'][0], calls['filter_related_articles_to_core'][0])
        self.assertEqual(3, len(calls['_review_attention_upper']))


if __name__ == '__main__':
    unittest.main()
