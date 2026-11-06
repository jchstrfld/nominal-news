"""Offline publication-evidence contract checks; no model-quality claims."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
from html import unescape
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def load_checks(api):
    tree = ast.parse((ROOT/'final_cohesion_check.py').read_text(encoding='utf-8'))
    functions = {'canonicalize_url', 'domain_from_url', '_publication_excerpt',
                 '_publication_cache_key', '_publication_excerpts', '_resolve_core_evidence',
                 '_event_json_request', '_publication_article_text', '_event_candidates', '_verify_event_candidates', '_verify_one_event', '_verify_core_support', 'purify_cluster_with_gpt'}
    functions.update({'_remember_event_response', '_classify_event_kinds', '_support_quotes_bound'})
    constants = {'_TRACKING_PARAMS', 'EVENT_MODEL', 'EVENT_SUPPORT_MODEL', 'EVENT_CACHE_VERSION',
                 'MIN_PUBLISH_ARTICLES', 'MIN_PUBLISH_DOMAINS', 'EVENT_BUDGET_USD',
                 '_EVENT_SPENT_USD', '_EVENT_RESPONSE_CACHE', '_EVENT_PENDING_RESPONSE', '_EVENT_BUDGET_REACHED'}
    nodes = [n for n in tree.body if
        isinstance(n, ast.FunctionDef) and n.name in functions or
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    scope = dict(os=os, re=re, json=json, hashlib=hashlib, unescape=unescape,
                 urlparse=urlparse, _EVENT_FINISH_REASONS=Counter(), _EVENT_TOKEN_USAGE=Counter(),
                 _EVENT_REQUEST_COUNTS=Counter(),
                 openai=SimpleNamespace(ChatCompletion=SimpleNamespace(create=api)))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/'final_cohesion_check.py'), 'exec'), scope)
    return scope


class CoreEvidenceChecks(unittest.TestCase):
    def test_event_selection_schema_enforces_existing_description_lengths(self):
        self.run_reply(self.good, self.proposal(self.good))
        request = self.api.call_args_list[0].kwargs
        self.assertEqual('event_selection', request['response_format']['json_schema']['name'])
        properties = request['response_format']['json_schema']['schema']['properties']
        schemas = [row['items'] for row in properties['stated_events']['properties'].values()]
        schemas.append(properties['candidates']['items']['properties']['event'])
        for schema in schemas:
            self.assertEqual({'type': 'string', 'minLength': 1, 'maxLength': 200}, schema)
        prompt = request['messages'][1]['content']
        self.assertIn('1–200 characters', prompt)
        self.assertIn('never truncate a statement to fit', prompt)

    def test_event_description_length_boundaries_preserve_complete_text(self):
        excerpts = self.ns['_publication_excerpts'](self.good)
        candidate = {k:v for k,v in self.proposal(self.good).items() if k not in ('event_type', 'why')}
        candidate['event'] = self.good[0]['title']
        original = {'stated_events': {f'{i}:title': [a['title']] for i,a in enumerate(self.good,1)},
                    'candidates': [candidate]}
        for field in ('inventory', 'candidate'):
            for length in (199, 200, 201):
                obj = deepcopy(original)
                # Count Unicode characters, retain a qualification at the end,
                # and never clip even one character to satisfy the contract.
                description = 'É'*(length-len(' — still uncertain'))+' — still uncertain'
                if field == 'inventory': obj['stated_events']['1:title'] = [description]
                else: obj['candidates'][0]['event'] = description
                before = deepcopy(obj)
                with self.subTest(field=field, length=length):
                    if length > 200:
                        with self.assertRaises(ValueError): self.ns['_event_candidates'](self.good, excerpts, obj)
                    else:
                        selected = self.ns['_event_candidates'](self.good, excerpts, obj)
                        if field == 'candidate': self.assertEqual(description, selected[0]['event'])
                    self.assertEqual(before, obj)

    def test_overlong_inventory_stays_blocked_before_independent_review(self):
        candidate = {k:v for k,v in self.proposal(self.good).items() if k not in ('event_type', 'why')}
        candidate['event'] = self.good[0]['title']
        selection = {'stated_events': {f'{i}:title': [a['title']] for i,a in enumerate(self.good,1)},
                     'candidates': [candidate]}
        selection['stated_events']['1:title'] = ['x'*201]
        result = self.run_reply(self.good, selection)
        self.assertEqual('ERROR', result['action'])
        self.assertEqual('Incomplete source event inventory', result['why'])
        self.assertEqual([], result['core_indexes'])
        self.api.assert_called_once()

    def test_quote_typography_preserves_words_case_and_field_binding(self):
        check = self.ns['_support_quotes_bound']
        source = "The council's permit was approved after a vote."
        excerpts = {'1:title': {'index': 1, 'text': source}}
        for quote in (source, '"'+source+'"', '“'+source+'”', source.replace("'", '’')):
            with self.subTest(quote=quote):
                self.assertTrue(check(excerpts, ['1:title'], {'1:title': quote}))
        for quote in ('"'+source, source+'"', '"'+source+'”}},',
                      source.lower(), source.replace('approved', 'denied'),
                      source.replace('after', 'before'), source.replace("council's", 'councils'),
                      source.replace('permit', '“permit”'), '"approvedd"', '""'):
            with self.subTest(quote=quote):
                self.assertFalse(check(excerpts, ['1:title'], {'1:title': quote}))
        self.assertFalse(check(excerpts, ['1:title'], {'2:title': source}))

    def test_invisible_formatting_is_ignored_only_at_field_or_whitespace_edges(self):
        check = self.ns['_support_quotes_bound']
        for control in ('\u200b', '\u200c', '\u200d', '\ufeff'):
            with self.subTest(control=repr(control)):
                excerpts = {'1:title': {'index': 1, 'text': 'The permit was '+control+'approved.'}}
                self.assertTrue(check(excerpts, ['1:title'], {'1:title': 'The permit was approved.'}))
                excerpts['1:title']['text'] = control+'The permit was approved'+control
                self.assertTrue(check(excerpts, ['1:title'], {'1:title': 'The permit was approved'}))
                self.assertFalse(check(excerpts, ['1:title'], {'1:title': 'proved'}))
                self.assertFalse(check(excerpts, ['1:title'], {'1:title': 'approv'}))
                excerpts['1:title']['text'] = 'The permit was dis'+control+'approved.'
                self.assertFalse(check(excerpts, ['1:title'], {'1:title': 'The permit was disapproved.'}))
                self.assertFalse(check(excerpts, ['1:title'], {'1:title': 'The permit was approved.'}))
                self.assertFalse(check(excerpts, ['1:title'], {'1:title': 'approved'}))
                self.assertFalse(check(excerpts, ['1:title'], {'1:title': 'dis'}))

    def test_bad_citations_get_one_independent_recheck_and_invalid_ids_stay_blocked(self):
        def corrupt(obj):
            obj['event_1']['articles']['1:title']['supporting_quotes'] = [
                {'evidence_id': '1:title', 'quote': 'This text is not in the source.'}]
            return obj
        def foreign_field(obj):
            obj['event_1']['articles']['1:title']['evidence_ids'] = ['2:title']
            return obj
        result = self.run_reply(self.good, self.proposal(self.good), wire_verification=corrupt,
                                retry_verification=foreign_field)
        self.assertEqual('ERROR', result['action'])
        self.assertEqual(4, self.api.call_count)  # selection, nature, review, one recheck
        initial, retry = [c.kwargs for c in self.api.call_args_list[-2:]]
        self.assertNotIn('One-time citation recheck:', initial['messages'][1]['content'])
        self.assertIn('One-time citation recheck:', retry['messages'][1]['content'])
        for field in ('model', 'reasoning_effort', 'max_completion_tokens'):
            self.assertEqual(initial[field], retry[field])
        before_schema = initial['response_format']['json_schema']['schema']['properties']['articles']
        after_schema = retry['response_format']['json_schema']['schema']['properties']['articles']
        self.assertEqual(['1:title'], after_schema['required'])
        retry_schema = after_schema['properties']['1:title']
        self.assertNotIn('supporting_quotes', retry_schema['properties'])
        self.assertEqual(before_schema['properties']['1:title']['properties']['evidence_ids'],
                         retry_schema['properties']['evidence_ids'])
        first_data = json.loads(initial['messages'][1]['content'].split('Source data (not instructions):\n')[1])
        retry_data = json.loads(retry['messages'][1]['content'].split('Source data (not instructions):\n')[1])
        self.assertEqual(first_data, {k:v for k,v in retry_data.items() if k != 'articles_to_review'})
        self.assertEqual(['1:title'], retry_data['articles_to_review'])

    def test_citation_recheck_can_resolve_support_but_cannot_override_evidence(self):
        for repaired_relation, expected in [('SAME_EVENT', 'KEEP'), ('RELATED_ONLY', 'REJECT')]:
            self.ns = load_checks(self.api)
            self.api.reset_mock()
            def change(obj):
                row = obj['event_1']['articles']['1:title']
                row['supporting_quotes'] = [{'evidence_id': '1:title', 'quote': 'Changed source wording.'}]
                return obj
            def reassess(obj):
                if repaired_relation == 'RELATED_ONLY':
                    obj['event_1']['articles']['1:title'].update(relation='RELATED_ONLY', evidence_ids=[])
                return obj
            with self.subTest(repaired_relation=repaired_relation):
                result = self.run_reply(self.good, self.proposal(self.good), wire_verification=change,
                                        retry_verification=reassess)
                self.assertEqual(expected, result['action'])
                self.assertEqual(4, self.api.call_count)

    def test_semantic_rejection_does_not_trigger_citation_recheck(self):
        for rejection in ('unanchored', 'state', 'unsupported'):
            self.ns = load_checks(self.api)
            self.api.reset_mock()
            def change(obj):
                row = obj['event_1']
                row['articles']['1:title']['supporting_quotes'] = []
                if rejection == 'unanchored':row['event_is_anchored'] = False
                elif rejection == 'state':row['event_kind'] = 'ONGOING_ACTIVITY_OR_STATE'
                else:row['unsupported_claims'] = ['The selected action is unsupported.']
                return obj
            with self.subTest(rejection=rejection):
                result = self.run_reply(self.good, self.proposal(self.good), wire_verification=change)
                self.assertEqual('ERROR', result['action'])
                self.assertEqual(3, self.api.call_count)

    def test_citation_recheck_does_not_revisit_other_articles(self):
        articles = self.five_source_control()
        def change(obj):
            rows = obj['event_1']['articles']
            rows['4:title'].update(relation='RELATED_ONLY', evidence_ids=[], supporting_quotes=[])
            rows['5:title']['supporting_quotes'] = [
                {'evidence_id':'5:title','quote':articles[0]['title']}]
            return obj
        def reassess(obj):
            rows = obj['event_1']['articles']
            self.assertEqual({'1:title','5:title'}, set(rows))
            rows['5:title'].update(relation='RELATED_ONLY', evidence_ids=[])
            return obj
        result = self.run_reply(articles, self.proposal(articles), wire_verification=change,
                                retry_verification=reassess)
        self.assertEqual('REJECT', result['action'])
        request = self.api.call_args.kwargs
        data = json.loads(request['messages'][1]['content'].split('Source data (not instructions):\n')[1])
        self.assertEqual(['1:title','5:title'], data['articles_to_review'])
        self.assertEqual({f'{i}:title' for i in range(1,6)}, set(data['source_fields']))

    def test_failed_anchor_recheck_cannot_reuse_other_sources_as_an_anchor(self):
        def change(obj):
            obj['event_1']['articles']['4:title']['supporting_quotes'] = []
            return obj
        def reassess(obj):
            obj['event_1']['event_is_anchored'] = False
            return obj
        result = self.run_reply(self.good, self.proposal(self.good), wire_verification=change,
                                retry_verification=reassess)
        self.assertEqual('REJECT', result['action'])
        self.assertEqual([], result['core_indexes'])

    def test_citation_recheck_copies_only_selected_original_fields(self):
        articles = deepcopy(self.good)
        articles[0]['description'] = "Ms. O’Neil-Smith confirmed the permit was approved.\u200b"
        captured = []
        bound = self.ns['_support_quotes_bound']
        def inspect(excerpts, ids, quotes):
            captured.append((deepcopy(excerpts), deepcopy(ids), deepcopy(quotes)))
            return bound(excerpts, ids, quotes)
        self.ns['_support_quotes_bound'] = inspect
        def corrupt(obj):
            obj['event_1']['articles']['1:title']['supporting_quotes'] = []
            return obj
        def reassess(obj):
            obj['event_1']['articles']['1:title']['evidence_ids'] = ['1:description']
            return obj
        result = self.run_reply(articles, self.proposal(articles), wire_verification=corrupt,
                                retry_verification=reassess)
        self.assertEqual('KEEP', result['action'])
        excerpts, ids, quotes = captured[-1]
        self.assertEqual(['1:description'], ids)
        self.assertEqual({'1:title': None, '1:description': excerpts['1:description']['text']}, quotes)
        self.assertIn('O’Neil-Smith', quotes['1:description'])
        self.assertEqual(articles[0]['description'], quotes['1:description'])

    def test_citation_recheck_rejects_missing_foreign_duplicate_and_extra_evidence(self):
        def corrupt(obj):
            obj['event_1']['articles']['1:title']['supporting_quotes'] = []
            return obj
        for ids in ([], ['2:title'], ['9:title'], ['1:title','1:title'], [True], '1:title'):
            def invalid(obj):
                obj['event_1']['articles']['1:title']['evidence_ids'] = ids
                return obj
            with self.subTest(ids=ids):
                result = self.run_reply(self.good, self.proposal(self.good), wire_verification=corrupt,
                                        retry_verification=invalid)
                self.assertEqual('ERROR', result['action'])
                self.assertEqual([], result['core_indexes'])
        def injected_quote(obj):
            obj['event_1']['articles']['1:title']['supporting_quotes'] = {'1:title': 'Invented evidence.'}
            return obj
        result = self.run_reply(self.good, self.proposal(self.good), wire_verification=corrupt,
                                retry_verification=injected_quote)
        self.assertEqual('ERROR', result['action'])

    def test_citation_recheck_still_blocks_unsupported_events_and_states(self):
        def corrupt(obj):
            obj['event_1']['articles']['1:title']['supporting_quotes'] = []
            return obj
        for mode in ('unsupported', 'ongoing', 'mixed'):
            def reject(obj):
                row = obj['event_1']
                if mode == 'unsupported': row['unsupported_claims'] = ['The action is not established.']
                elif mode == 'ongoing': row['event_kind'] = 'ONGOING_ACTIVITY_OR_STATE'
                else: row['event_kind'] = 'TOPIC_OR_MIXED'
                return obj
            with self.subTest(mode=mode):
                result = self.run_reply(self.good, self.proposal(self.good), wire_verification=corrupt,
                                        retry_verification=reject)
                self.assertEqual('REJECT', result['action'])
                self.assertEqual([], result['core_indexes'])

    def test_exact_sentence_quotes_survive_joined_source_sentences(self):
        check = self.ns['_support_quotes_bound']
        for source, quote in (
                ('The council approved the permit.Residents attended.',
                 'The council approved the permit.'),
                ('Caption)The council approved the permit.The vote was unanimous.',
                 'The council approved the permit.'),
                ('The notice reads(approved).', '(approved)'),
                ('The council approved the permit.', 'The council approved the permit.')):
            with self.subTest(source=source):
                excerpts = {'1:description': {'index': 1, 'text': source}}
                self.assertTrue(check(excerpts, ['1:description'], {'1:description': quote}))

    def test_joined_sentence_handling_keeps_word_and_source_boundaries(self):
        check = self.ns['_support_quotes_bound']
        for source, quote in (
                ('The council disapproved the permit.', 'approved'),
                ('The council disapproved the permit.', 'approved the permit.'),
                ('The council approved the permit.', 'ed the permit.'),
                ('The council approved permits.', 'The council approved permit'),
                ('The council did not approve the permit.', 'The council approved the permit.'),
                ('The council may approve the permit.', 'The council approved the permit.'),
                ('The event starts in 2027.', 'The event starts in 2026.'),
                ('The council approved the permit.', 'The council ... the permit.'),
                ('The council approved the permit.', 'the council approved the permit.')):
            with self.subTest(source=source, quote=quote):
                excerpts = {'1:title': {'index': 1, 'text': source},
                            '2:title': {'index': 2, 'text': quote}}
                self.assertFalse(check(excerpts, ['1:title'], {'1:title': quote}))
                self.assertFalse(check(excerpts, ['1:title'], {'2:title': quote}))

    def test_supporting_quotes_must_match_the_cited_field(self):
        excerpts = {
            '1:title': {'index': 1, 'text': 'The council disapproved the permit.'},
            '1:description': {'index': 1, 'text': 'The council rejected the permit after a vote.'},
            '2:title': {'index': 2, 'text': 'The council approved the permit.'},
        }
        check = self.ns['_support_quotes_bound']
        self.assertTrue(check(excerpts, ['1:description'], [
            {'evidence_id': '1:description', 'quote': 'The council\n rejected the permit'}]))
        for proof in (
                {'evidence_id': '1:title', 'quote': excerpts['2:title']['text']},
                {'evidence_id': '2:title', 'quote': excerpts['2:title']['text']},
                {'evidence_id': '1:title', 'quote': 'approved'},
                {'evidence_id': '1:title', 'quote': 'The council refused the permit.'},
                {'evidence_id': '1:title', 'quote': 'The council ... the permit.'},
                {'evidence_id': '1:title', 'quote': ''},
                {'evidence_id': '1:title', 'quote': None},
                {'evidence_id': '1:title'},
                None):
            with self.subTest(proof=proof):
                self.assertFalse(check(excerpts, ['1:title'], [proof]))

    def test_every_cited_field_needs_a_bound_quote(self):
        excerpts = self.ns['_publication_excerpts'](self.good)
        ids = ['1:title', '1:description']
        quotes = [{'evidence_id': eid, 'quote': excerpts[eid]['text']} for eid in ids]
        check = self.ns['_support_quotes_bound']
        self.assertTrue(check(excerpts, ids, quotes))
        for incomplete in ([], quotes[:1], quotes + quotes[:1]):
            self.assertFalse(check(excerpts, ids, incomplete))
        self.assertFalse(check(excerpts, ids[:1], quotes))

    def test_field_keyed_quotes_preserve_exact_provenance_and_legacy_receipts(self):
        excerpts = self.ns['_publication_excerpts'](self.good)
        ids = ['1:title', '1:description']
        proof = {eid: excerpts[eid]['text'] for eid in ids}
        check = self.ns['_support_quotes_bound']
        self.assertTrue(check(excerpts, ids, proof))
        self.assertTrue(check(excerpts, ids[:1], {**proof, '1:description': None}))
        self.assertTrue(check(excerpts, ids, [
            {'evidence_id': eid, 'quote': quote} for eid, quote in proof.items()]))
        for invalid in (
                {'1:title': proof['1:title']},
                {**proof, '2:title': None},
                {**proof, '1:description': proof['1:title']},
                {**proof, '1:title': [proof['1:title'], proof['1:title']]},
                {**proof, '1:title': None},
                {**proof, '1:title': ''},
                {**proof, '1:title': True}):
            with self.subTest(proof=invalid):
                self.assertFalse(check(excerpts, ids, invalid))

    def test_keyed_quotes_use_same_support_and_source_count_rules(self):
        def keyed(obj):
            for row in obj['event_1']['articles'].values():
                ids = row['evidence_ids']
                index = ids[0].split(':')[0]
                quotes = {f'{index}:title': None, f'{index}:description': None}
                quotes.update({q['evidence_id']: q['quote'] for q in row['supporting_quotes']})
                row['supporting_quotes'] = quotes
            return obj
        legacy = self.run_reply(self.good, self.proposal(self.good))
        mapped = self.run_reply(self.good, self.proposal(self.good), wire_verification=keyed)
        self.assertEqual(legacy, mapped)
        options = self.api.call_args.kwargs
        self.assertEqual('medium', options['reasoning_effort'])
        self.assertEqual(10000, options['max_completion_tokens'])
        self.assertEqual(180, options['request_timeout'])
        fields = options['response_format']['json_schema']['schema']['properties']
        proof = fields['articles']['properties']['1:title']['properties']['supporting_quotes']
        self.assertEqual('object', proof['type'])
        self.assertEqual({'1:title', '1:description'}, set(proof['properties']))
        self.assertEqual({'1:title', '1:description'}, set(proof['required']))
        self.assertFalse(proof['additionalProperties'])
        self.assertEqual(['string', 'null'], proof['properties']['1:title']['type'])

    @staticmethod
    def five_source_control():
        titles = ['Council reopens River bridge after repairs.',
                  'Drivers return as River bridge reopens.',
                  'River bridge reopening restores the town crossing.',
                  'Residents celebrate the reopening of River bridge.',
                  'Ferry service suspended because of fog.']
        return [{'title': title, 'url': f'https://source{i}.example/news'}
                for i, title in enumerate(titles, 1)]

    def test_borrowed_text_withholds_one_member_and_keeps_supported_story(self):
        articles = self.five_source_control()
        self.ns['_EVENT_RESPONSE_CACHE'] = {}
        self.ns['_save_event_cache'] = Mock()
        def borrowed(obj):
            obj['event_1']['articles']['5:title']['supporting_quotes'] = [
                {'evidence_id': '5:title', 'quote': articles[0]['title']}]
            return obj
        def reassess(obj):
            obj['event_1']['articles']['5:title'].update(relation='DIFFERENT_EVENT', evidence_ids=[])
            return obj
        result = self.run_reply(articles, self.proposal(articles), wire_verification=borrowed,
                                retry_verification=reassess)
        self.assertIn(result['action'], ('KEEP', 'CLEAN'))
        self.assertEqual([1, 2, 3, 4], result['core_indexes'])
        self.assertNotIn(articles[4]['url'], result['core_evidence'])
        self.assertEqual(4, self.api.call_count)  # one bounded citation recheck; still excluded
        self.assertEqual(3, len(self.ns['_EVENT_RESPONSE_CACHE']))  # selection, nature, valid recheck only
        props = self.api.call_args.kwargs['response_format']['json_schema']['schema']['properties']
        row_schema = props['articles']['properties']['5:title']
        self.assertEqual(['5:title'], row_schema['properties']['evidence_ids']['items']['enum'])
        self.assertFalse(row_schema['additionalProperties'])

    def test_unbound_member_cannot_fill_article_or_domain_minimums(self):
        articles = self.five_source_control()
        for mode in ('articles', 'domains'):
            case = deepcopy(articles[:4] if mode == 'articles' else articles)
            bad = len(case)
            if mode == 'domains':
                for i, article in enumerate(case[:-1]):
                    article['url'] = f'https://source{i % 2}.example/{i}'
                case[-1]['url'] = 'https://third.example/news'
            def unbound(obj):
                obj['event_1']['articles'][f'{bad}:title']['supporting_quotes'] = []
                return obj
            def reassess(obj):
                obj['event_1']['articles'][f'{bad}:title'].update(relation='UNCLEAR_IDENTITY', evidence_ids=[])
                return obj
            with self.subTest(mode=mode):
                result = self.run_reply(case, self.proposal(case), wire_verification=unbound,
                                        retry_verification=reassess)
                self.assertEqual('REJECT', result['action'])
                self.assertEqual('UNCERTAIN', result['evidence_status'])
                self.assertEqual([], result['core_indexes'])

    def test_unbound_anchor_never_uses_other_members_as_a_replacement_anchor(self):
        def unbound(obj):
            obj['event_1']['articles']['1:title']['supporting_quotes'] = []
            return obj
        def reassess(obj):
            obj['event_1']['articles']['1:title'].update(relation='UNCLEAR_IDENTITY', evidence_ids=[])
            return obj
        result = self.run_reply(self.good, self.proposal(self.good), wire_verification=unbound,
                                retry_verification=reassess)
        self.assertEqual('REJECT', result['action'])
        self.assertEqual([], result['core_indexes'])

    def test_old_id_only_response_cannot_be_used_as_verified_support(self):
        def old_contract(obj):
            for row in obj['event_1']['articles'].values():
                row.pop('supporting_quotes')
            return obj
        self.assertEqual('ERROR', self.run_reply(self.good, self.proposal(self.good),
                                               wire_verification=old_contract)['action'])
        new_key = self.ns['_publication_cache_key'](self.good)
        self.ns['EVENT_CACHE_VERSION'] = 'v16-explicit-event-support-target'
        self.assertNotEqual(new_key, self.ns['_publication_cache_key'](self.good))

    def test_bound_quote_never_overrides_a_nonmatching_semantic_judgment(self):
        def related(obj):
            obj['event_1']['articles']['4:title']['relation'] = 'RELATED_ONLY'
            return obj
        result = self.run_reply(self.good, self.proposal(self.good), wire_verification=related)
        self.assertEqual('REJECT', result['action'])
        self.assertEqual('Insufficient independently supported core', result['why'])

    def test_independent_support_target_is_bound_to_the_membership_review(self):
        for target in ('COMMUNICATION', 'DEVELOPMENT'):
            result = self.run_reply(self.good, self.proposal(self.good), classification_target=target)
            self.assertEqual('KEEP', result['action'])
            request = self.api.call_args.kwargs
            data = json.loads(request['messages'][1]['content'].split('Source data (not instructions):\n', 1)[1])
            self.assertEqual(target, data['support_target'])
        self.api.reset_mock()
        with self.assertRaises(ValueError):
            self.ns['_verify_one_event'](self.ns['_publication_excerpts'](self.good),
                                         {'event': self.good[0]['title'], 'anchor_id': '1:title'})
        self.api.assert_not_called()

    def test_event_nature_is_checked_before_counting_matching_articles(self):
        proposal = self.proposal(self.good)
        proposal['event'] = 'Two laboratories share a campus.'
        candidates = [{'event': proposal['event'], 'anchor_id': '1:title',
                       'core_excerpt_ids': ['1:title'], 'related_indexes': [1, 2, 3, 4]}]
        selection = {'stated_events': {f'{i}:title': [a['title']] for i, a in enumerate(self.good, 1)},
                     'candidates': candidates}
        def response(obj):
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message={'content': json.dumps(obj)})], usage={})
        for category in ('STATE', 'TOPIC', 'UNCLEAR'):
            self.api.reset_mock()
            self.api.side_effect = [response(selection), response({'event_1': {'reason': 'No established new action.', 'category': category, 'support_target': 'NONE'}})]
            result = self.ns['purify_cluster_with_gpt'](self.good)
            self.assertEqual('REJECT', result['action'])
            self.assertEqual([], result['core_indexes'])
            self.assertEqual(2, self.api.call_count)  # no article-matching call
            prompt = self.api.call_args.kwargs['messages'][1]['content']
            data = json.loads(prompt.split('Source data:\n', 1)[1])
            self.assertEqual(proposal['event'], data[0]['fixed_event'])
            self.assertNotIn('core_excerpt_ids', data[0])
            self.assertNotIn('source_count', data[0])

    def test_invalid_event_nature_response_is_an_error_not_a_rejection(self):
        excerpts = self.ns['_publication_excerpts'](self.good)
        candidate = {'event': self.good[0]['title'], 'anchor_id': '1:title'}
        for obj in ({}, {'event_1': {'reason': '', 'category': 'ACTION'}},
                    {'event_1': {'reason': 'No decision', 'category': 'MAYBE'}}):
            self.api.return_value = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message={'content': json.dumps(obj)})], usage={})
            with self.assertRaises(ValueError):
                self.ns['_classify_event_kinds'](excerpts, [candidate])

    def test_optional_details_are_separate_from_occurrence_membership(self):
        def partial_details(obj):
            for eid, row in obj['event_1']['articles'].items():
                row['evidence_ids'] = [eid]
            return obj
        result = self.run_reply(self.good, self.proposal(self.good), wire_verification=partial_details)
        self.assertEqual('KEEP', result['action'])
        self.assertEqual([1, 2, 3, 4], result['core_indexes'])

    def test_shared_occurrence_cannot_authorize_an_unsupported_description(self):
        def no_details(obj):
            obj['event_1']['unsupported_claims'] = ['An unverified additional outcome.']
            return obj
        result = self.run_reply(self.good, self.proposal(self.good), wire_verification=no_details)
        self.assertEqual('REJECT', result['action'])
        self.assertEqual('UNCERTAIN', result['evidence_status'])
        self.assertEqual([], result['core_indexes'])

    def test_insufficient_identity_evidence_is_an_explicit_nonapproval(self):
        def unclear(obj):
            obj['event_1']['articles']['4:title'] = {
                'reason': 'Insufficient identity evidence.', 'relation': 'UNCLEAR_IDENTITY',
                'evidence_ids': [], 'supporting_quotes': []}
            return obj
        self.assertEqual('REJECT', self.run_reply(self.good, self.proposal(self.good), wire_verification=unclear)['action'])

    def test_budget_blocks_request_before_api_dispatch(self):
        self.ns['EVENT_BUDGET_USD'] = .000001
        result = self.ns['purify_cluster_with_gpt'](self.good)
        self.assertEqual('ERROR', result['action'])
        self.assertTrue(self.ns['_EVENT_BUDGET_REACHED'])
        self.api.assert_not_called()

    def test_verified_request_cache_resumes_without_repeating_calls(self):
        self.ns['_EVENT_RESPONSE_CACHE'] = {}
        self.ns['_save_event_cache'] = Mock()
        first = self.run_reply(self.good, self.proposal(self.good))
        self.assertEqual('KEEP', first['action'])
        calls = self.api.call_count
        spent = self.ns['_EVENT_SPENT_USD']
        self.api.side_effect = AssertionError('Cached request must not call API')
        second = self.ns['purify_cluster_with_gpt'](self.good)
        self.assertEqual(first, second)
        self.assertEqual(calls, self.api.call_count)
        self.assertEqual(spent, self.ns['_EVENT_SPENT_USD'])

    def test_malformed_evidence_is_not_saved_as_reusable_verification(self):
        self.ns['_EVENT_RESPONSE_CACHE'] = {}
        self.ns['_save_event_cache'] = Mock()
        def bad(obj):
            obj['event_1']['articles']['1:title']['evidence_ids'] = ['2:title']
            return obj
        self.assertEqual('ERROR', self.run_reply(self.good, self.proposal(self.good), wire_verification=bad)['action'])
        self.assertEqual(2, len(self.ns['_EVENT_RESPONSE_CACHE']))  # selection and event nature

    def setUp(self):
        self.enterContext(patch('socket.socket.connect', side_effect=AssertionError('No network')))
        self.enterContext(patch.dict(os.environ, {'OPENAI_API_KEY': 'OFFLINE_TEST_SENTINEL',
            'NN_MIN_PUBLISH_ARTICLES': '4', 'NN_MIN_PUBLISH_DOMAINS': '3'}))
        self.api = Mock()
        self.ns = load_checks(self.api)
        self.fixture = json.loads((ROOT/'tests/fixtures/core_evidence_cases.json').read_text(encoding='utf-8'))
        self.good = self.fixture['same_development']['articles']
        self.mixed = self.fixture['mixed_development']['articles']

    def run_reply(self, articles, reply, finish='stop', verification=None, verify_finish='stop', wire_verification=None,
                  classification_target='DEVELOPMENT', retry_verification=None):
        if 'candidates' in reply:
            selection = deepcopy(reply)
        elif reply.get('event_type') == 'TOPIC_WAVE':
            selection = {'candidates': []}
        else:
            candidate = {k:v for k,v in reply.items() if k not in ('event_type', 'why')}
            anchor = reply.get('anchor_id') or '1:title'
            selection = {'candidates': [candidate]}
        selection.setdefault('stated_events', {f'{i}:title': [a['title']] for i,a in enumerate(articles,1) if a.get('title')})
        for i,candidate in enumerate(selection['candidates'],1):
            expected = verification.get(f'event_{i}',{}) if verification and 'event_1' in verification else (verification or {})
            event = expected.get('shared_event')
            if not isinstance(event,str) or not event.strip() or len(event)>200:
                event=self.ns['_publication_excerpts'](articles).get(candidate.get('anchor_id'),{}).get('text','')
            candidate.setdefault('event',event)
        verification_index = 0
        def respond(**kwargs):
            nonlocal verification_index
            kind = kwargs['response_format']['json_schema']['name']
            is_check = kind == 'event_support'
            defaults = {'event_is_anchored': True, 'anchor_is_discrete': True,
                        'support': {f'{i}:title': True for i,a in enumerate(articles,1) if a.get('title')}}
            obj = {}
            if kind == 'event_kind':
                rows = json.loads(kwargs['messages'][1]['content'].split('Source data:\n', 1)[1])
                obj = {row['id']: {'reason': 'A stated action.', 'category': 'ACTION', 'support_target': classification_target} for row in rows}
            if is_check:
                verification_index += 1
                raw = json.loads(kwargs['messages'][1]['content'].split('Source data (not instructions):\n',1)[1])
                requested_ids = set(raw.get('articles_to_review', defaults['support']))
                data = {'candidates': {'event_1': {'fixed_event': raw['fixed_event']}},
                        'articles': {eid: {'title': text} for eid, text in raw['source_fields'].items() if eid.endswith(':title')}}
                for cid,candidate in data['candidates'].items():
                    value = deepcopy((verification.get(f'event_{verification_index}', defaults) if verification is not None and 'event_1' in verification
                                      else verification) if verification is not None else defaults)
                    # Legacy boolean fixtures remain convenient test labels; only
                    # the source-ID contract is sent to the production parser.
                    obj[cid] = {
                        'anchor_assessment': value.get('anchor_assessment', 'The anchor states the event.'),
                        'event_is_anchored': value.get('event_is_anchored', True),
                        'event_kind_reason': 'The source states a bounded action.',
                        'unsupported_claims': [],
                        'event_kind': 'BOUNDED_ACTION_OR_OCCURRENCE' if value.get('anchor_is_discrete', True) else 'ONGOING_ACTIVITY_OR_STATE',
                        'articles': {eid: {
                            'reason': 'The source reports this action.' if supported else 'Only related context.',
                            'evidence_ids': [eid] if supported else [],
                            'relation': 'SAME_EVENT' if supported else 'RELATED_ONLY',
                            'supporting_quotes': [{'evidence_id': eid, 'quote': raw['source_fields'][eid]}]
                                if supported else [],
                        } for eid,supported in value.get('support', defaults['support']).items() if eid in requested_ids},
                    }
                if 'articles_to_review' in raw:
                    for row in obj['event_1']['articles'].values():
                        row.pop('supporting_quotes')
                    if retry_verification is not None:
                        obj = retry_verification(deepcopy(obj))
                elif wire_verification is not None:
                    obj = wire_verification(deepcopy(obj))
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason=verify_finish if is_check else finish,
                message={'content': json.dumps(obj['event_1'] if is_check else obj if kind == 'event_kind' else selection)})],
                usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15})
        self.api.side_effect = respond
        before = deepcopy(articles)
        result = self.ns['purify_cluster_with_gpt'](articles)
        self.assertEqual(before, articles)
        return result

    def proposal(self, articles):
        indexes = list(range(1, len(articles)+1))
        return {'anchor_id': '1:title', 'event_type': 'DISCRETE_EVENT',
                'related_indexes': indexes, 'why': 'Same development.',
                'core_excerpt_ids': [f'{i}:title' for i in indexes]}

    def test_saved_unsupported_approval_cannot_pass_without_source_ids(self):
        old = self.fixture['mixed_development']['previous_decision']
        result = self.run_reply(self.mixed, old)
        self.assertEqual('ERROR', result['action'])
        self.assertEqual([], result['core_indexes'])
        self.assertIn('source IDs', result['why'])

    def test_saved_concrete_anchors_have_insufficient_same_event_support(self):
        # Manual labels from the saved source text, not fresh model decisions:
        # the full refusal/subpoena headline, prosecution demand, and televised
        # accusation each have only one supplied report of that exact event.
        for anchor in ('1:title', '3:title', '4:title'):
            reply = {'anchor_id': anchor, 'event_type': 'DISCRETE_EVENT',
                     'core_excerpt_ids': [anchor], 'related_indexes': [1,2,3,4,5]}
            result = self.run_reply(self.mixed, reply, verification={
                'event_is_anchored': True, 'anchor_is_discrete': True,
                'support': {f'{i}:title': f'{i}:title' == anchor for i in range(1,6)}})
            self.assertEqual('REJECT', result['action'])
            self.assertEqual('Insufficient independently supported core', result['why'])
            self.assertEqual(self.ns['_publication_excerpts'](self.mixed)[anchor]['text'], result['event'])

    def test_saved_same_event_control_requires_independent_bounded_requests(self):
        result = self.run_reply(self.good, self.proposal(self.good))
        self.assertEqual('KEEP', result['action'])
        self.assertEqual(4, len(result['core_evidence']))
        self.assertEqual(3, self.api.call_count)
        args = self.api.call_args_list[0].kwargs
        self.assertEqual(6000, args['max_completion_tokens'])
        self.assertEqual(45, args['request_timeout'])
        self.assertEqual('low', args['reasoning_effort'])
        prompt = args['messages'][1]['content']
        self.assertIn(self.ns['_publication_excerpt'](self.good[0]['description']), prompt)
        self.assertIn('cannot satisfy CORE support', prompt)
        self.assertIn('separate', prompt.lower())
        self.assertEqual(self.good[0]['title'], result['event'])
        self.assertEqual([1,2,3,4], result['core_indexes'])
        self.assertEqual(45, self.ns['_EVENT_TOKEN_USAGE']['total_tokens'])
        self.assertEqual({'event_selection': 1, 'event_kind': 1, 'event_support': 1}, self.ns['_EVENT_REQUEST_COUNTS'])
        check_args = self.api.call_args_list[2].kwargs
        self.assertEqual(self.ns['EVENT_SUPPORT_MODEL'],check_args['model'])
        self.assertEqual('medium',check_args['reasoning_effort'])
        self.assertNotIn('max_tokens',check_args)
        self.assertGreaterEqual(check_args['max_completion_tokens'],10000)
        self.assertLessEqual(check_args['max_completion_tokens'],12000)
        schema = args['response_format']['json_schema']
        self.assertTrue(schema['strict'])
        props = schema['schema']['properties']['candidates']['items']['properties']
        self.assertIn('1:title', props['anchor_id']['enum'])
        self.assertNotIn('1', props['anchor_id']['enum'])
        self.assertNotIn('1:description', props['anchor_id']['enum'])
        self.assertIn('1:description', props['core_excerpt_ids']['items']['enum'])
        self.assertFalse(schema['schema']['additionalProperties'])

    def test_ids_cannot_be_fabricated_duplicated_or_miscounted(self):
        good = self.proposal(self.good)
        for change in ('fabricated', 'missing', 'boolean', 'extra'):
            reply = deepcopy(good)
            if change == 'two_fields_one_article': reply['core_excerpt_ids'][1] = '1:description'
            elif change == 'fabricated': reply['core_excerpt_ids'][0] = '9:title'
            elif change == 'duplicate': reply['core_excerpt_ids'][1] = '1:title'
            elif change == 'missing': reply['core_excerpt_ids'].pop(0)
            elif change == 'boolean': reply['core_excerpt_ids'][0] = True
            else: reply['core_excerpt_ids'].append('8:description')
            with self.subTest(change=change):
                self.assertEqual('ERROR', self.run_reply(self.good, reply)['action'])

    def test_proposal_passages_never_count_as_additional_articles(self):
        proposal = self.proposal(self.good)
        proposal['core_excerpt_ids'] += ['1:title', '1:description']
        result = self.run_reply(self.good, proposal)
        self.assertEqual('KEEP', result['action'])
        self.assertEqual([1,2,3,4], result['core_indexes'])
        proposal['core_excerpt_ids'] = ['1:description','2:title','3:title','4:title']
        self.assertEqual('KEEP', self.run_reply(self.good, proposal)['action'])
        self.assertEqual('REJECT', self.run_reply(self.good, proposal, verification={
            'support': {f'{i}:title': i < 4 for i in range(1,5)}})['action'])

    def test_description_only_support_and_html_cleanup_are_general(self):
        for event in ('The board approved the transit contract.', 'The court dismissed the appeal.'):
            articles = [{'title': 'Regional update', 'description': '<p>'+event+' &amp; details</p>',
                         'url': f'https://outlet{i}.example/update'} for i in range(4)]
            articles[0]['title'] = event
            reply = self.proposal(articles)
            reply['core_excerpt_ids'] = ['1:title', '2:description', '3:description', '4:description']
            result = self.run_reply(articles, reply)
            self.assertEqual('KEEP', result['action'])
            self.assertEqual(event, result['event'])
            self.assertTrue(all(event+' & details' in text for text in result['core_evidence'].values()))
        self.assertEqual('A & B', self.ns['_publication_excerpt']('<p>A &amp; B</p>'))
        self.assertEqual(300, len(self.ns['_publication_excerpt']('x'*1000)))

    def test_same_outlet_and_duplicate_urls_do_not_supply_breadth(self):
        articles = deepcopy(self.good)
        for i,a in enumerate(articles): a['url'] = f'https://one.example/{i}'
        self.assertEqual('REJECT', self.run_reply(articles, self.proposal(articles))['action'])
        articles = deepcopy(self.good)
        articles[1]['url'] = articles[0]['url']
        self.assertEqual('ERROR', self.run_reply(articles, self.proposal(articles))['action'])

    def test_truncation_and_transport_errors_do_not_become_approvals(self):
        self.assertEqual('ERROR', self.run_reply(self.good, self.proposal(self.good), 'length')['action'])
        self.api.side_effect = RuntimeError('Transport unavailable')
        self.assertEqual('ERROR', self.ns['purify_cluster_with_gpt'](self.good)['action'])

    def test_cache_binds_exact_supplied_evidence_and_is_order_independent(self):
        key = self.ns['_publication_cache_key']
        original = key(self.good)
        self.assertEqual(original, key(list(reversed(self.good))))
        for field in ('title', 'description', 'url'):
            changed = deepcopy(self.good); changed[0][field] += ' changed'
            self.assertNotEqual(original, key(changed))
        self.assertNotIn('v6-discrete-event-anchor', original)
        self.assertNotIn('v7-core-source-evidence', original)
        self.assertNotIn('v8-source-anchor-ids', original)
        self.assertNotIn('v9-independent-core-support', original)
        self.assertNotIn('v10-complete-article-support', original)
        self.assertNotIn('v11-shared-event-support', original)
        self.assertNotIn('v12-verified-event-selection', original)
        self.ns['EVENT_SUPPORT_MODEL'] = 'different-verifier'
        self.assertNotEqual(original, key(self.good))

    def test_topic_rejection_does_not_require_core_quotes(self):
        reply = {'anchor_id': None, 'event_type': 'TOPIC_WAVE', 'core_excerpt_ids': [], 'related_indexes': []}
        self.assertEqual('REJECT', self.run_reply(self.good, reply)['action'])

    def test_saved_broad_event_cannot_be_written_or_used_as_description_anchor(self):
        saved = self.fixture['mixed_development']['legacy_response']
        self.assertIn('legal scrutiny', saved['event'])
        self.assertEqual('ERROR', self.run_reply(self.mixed, saved)['action'])
        # Mapping its chosen passages to IDs does not make the broad description
        # an allowed event anchor. It is evidence context, not a headline anchor.
        reply = {'anchor_id': '2:description', 'event_type': 'DISCRETE_EVENT',
                 'core_excerpt_ids': ['1:title','2:description','4:title','5:description'],
                 'related_indexes': [1,2,3,4,5]}
        self.assertEqual('ERROR', self.run_reply(self.mixed, reply)['action'])
        for anchor in ('2:title', '5:title'):
            reply['anchor_id'] = anchor
            self.assertEqual('ERROR', self.run_reply(self.mixed, reply)['action'])

    def test_saved_control_copy_error_is_eliminated_by_ids(self):
        wrong_quote = self.fixture['same_development']['legacy_response']['core_evidence'][0]['quote']
        self.assertNotIn(wrong_quote, self.good[0]['title'])
        reply = self.proposal(self.good)
        result = self.run_reply(self.good, reply)
        self.assertEqual('KEEP', result['action'])
        url = self.ns['canonicalize_url'](self.good[0]['url'])
        self.assertEqual(self.good[0]['title']+'\n'+self.good[0]['description'], result['core_evidence'][url])
        self.assertNotEqual(wrong_quote, result['core_evidence'][url])

    def test_extra_generated_event_or_membership_cannot_replace_fixed_anchor(self):
        for field,value in [('support_target','DEVELOPMENT'), ('statement','The region faces controversy.'), ('shared_event','An unverified rewrite.'), ('core_indexes',[1,2,3,4]),
                            ('core_evidence',[{'quote':'invented words'}])]:
            reply = self.proposal(self.good); reply[field] = value
            self.assertEqual('ERROR', self.run_reply(self.good, reply)['action'])

    def test_reordered_sources_keep_url_bound_evidence_and_cache_identity(self):
        first = self.run_reply(self.good, self.proposal(self.good))
        reordered = list(reversed(self.good)); reply = self.proposal(reordered)
        reply['anchor_id'] = '4:title'
        second = self.run_reply(reordered, reply)
        self.assertEqual(first['event'], second['event'])
        self.assertEqual(first['event_anchor_url'], second['event_anchor_url'])
        self.assertEqual(first['core_evidence'], second['core_evidence'])

    def test_independent_check_rejects_saved_alaska_even_with_repaired_ids(self):
        reply = {'anchor_id': '1:title', 'event_type': 'DISCRETE_EVENT',
                 'core_excerpt_ids': ['1:title', '3:title', '4:title', '5:description'],
                 'related_indexes': [1,3,4,5], 'why': 'WRONG_SELECTOR_RATIONALE'}
        result = self.run_reply(self.mixed, reply, verification={
            'event_is_anchored': True, 'anchor_is_discrete': True,
            'support': {f'{i}:title': i == 1 for i in range(1,6)}})
        self.assertEqual('REJECT', result['action'])
        self.assertEqual('Insufficient independently supported core', result['why'])
        self.assertEqual([], result['core_indexes'])
        prompt = self.api.call_args.kwargs['messages'][1]['content']
        self.assertNotIn('WRONG_SELECTOR_RATIONALE', prompt)
        self.assertNotIn('at least 4', prompt)
        self.assertNotIn('at least 3', prompt)

    def test_saved_live_selection_and_support_decisions(self):
        # Replay old judgments, treating previously unchecked articles as false.
        # This checks structural compatibility, not the new prompt's accuracy.
        for name, expected in [('mixed_development', 'REJECT'), ('same_development', 'KEEP')]:
            case = self.fixture[name]
            verification = deepcopy(case['support_response'])
            verification['event_is_anchored'] = True
            verification['support'] = {f'{i}:title': verification['support'].get(f'{i}:title', False)
                                       for i in range(1,len(case['articles'])+1)}
            selection = deepcopy(case['selection_response'])
            result = self.run_reply(case['articles'], selection, verification=verification)
            self.assertEqual(expected, result['action'])

    def test_support_check_removes_unsupported_articles_and_recounts_outlet_breadth(self):
        articles = deepcopy(self.good)
        articles.append({'url': 'https://fifth.example/news', 'title': 'A separate event.'})
        reply = self.proposal(articles)
        result = self.run_reply(articles, reply, verification={
            'event_is_anchored': True, 'anchor_is_discrete': True, 'support': {f'{i}:title': i < 5 for i in range(1,6)}})
        self.assertEqual('CLEAN', result['action'])
        self.assertEqual([1,2,3,4], result['core_indexes'])
        self.assertEqual([1,2,3,4], result['related_indexes'])
        articles[0]['url'] = 'https://one.example/a'
        articles[1]['url'] = 'https://one.example/b'
        articles[2]['url'] = 'https://two.example/a'
        articles[3]['url'] = 'https://two.example/b'
        result = self.run_reply(articles, reply, verification={
            'event_is_anchored': True, 'anchor_is_discrete': True, 'support': {f'{i}:title': i < 5 for i in range(1,6)}})
        self.assertEqual('REJECT', result['action'])

    def test_missing_extra_uncertain_and_invalid_verifications_fail_closed(self):
        reply = self.proposal(self.good)
        def change(obj, mode):
            check = obj['event_1']
            if mode == 'missing_grounding': check.pop('event_is_anchored')
            elif mode == 'invalid_grounding': check['event_is_anchored'] = 'true'
            elif mode == 'missing': check['articles'].pop('2:title')
            elif mode == 'extra': check['articles']['5:title'] = check['articles']['1:title']
            elif mode == 'invalid_relation': check['articles']['2:title']['relation'] = True
            elif mode == 'missing_kind': check.pop('event_kind')
            elif mode == 'invalid_kind': check['event_kind'] = 'Maybe'
            elif mode == 'empty_assessment': check['anchor_assessment'] = ''
            else: check['shared_event'] = 'A replacement event'
            return obj
        for mode in ('missing_grounding', 'invalid_grounding', 'missing', 'extra', 'invalid_relation',
                     'missing_kind', 'invalid_kind', 'empty_assessment', 'rewrite'):
            with self.subTest(mode=mode):
                result = self.run_reply(self.good, reply, wire_verification=lambda obj: change(obj, mode))
                self.assertEqual('ERROR', result['action'])
        for kind in ('ONGOING_ACTIVITY_OR_STATE', 'TOPIC_OR_MIXED'):
            def ongoing(obj):
                obj['event_1']['event_kind'] = kind
                return obj
            self.assertEqual('REJECT', self.run_reply(self.good, reply, wire_verification=ongoing)['action'])
        self.assertEqual('REJECT', self.run_reply(self.good, reply, verification={
            'support': {f'{i}:title': i > 1 for i in range(1,5)}})['action'])

    def test_evidence_ids_must_belong_to_the_claimed_article(self):
        articles = [{'title': 'A separate observation '+str(i), 'description': 'The River bridge reopened.',
                     'url': f'https://source{i}.example/story'} for i in range(4)]
        proposal = self.proposal(articles)
        proposal['event'] = 'The River bridge reopened.'
        def fields(obj):
            for eid, row in obj['event_1']['articles'].items():
                row['evidence_ids'] = [eid.replace(':title', ':description')]
                row['supporting_quotes'] = [{'evidence_id': row['evidence_ids'][0],
                                             'quote': 'The River bridge reopened.'}]
            return obj
        result = self.run_reply(articles, proposal, wire_verification=fields)
        self.assertEqual('KEEP', result['action'])
        for ids in ([], ['2:title'], ['9:description'], ['1:title', '1:title'], [True], '1:title'):
            def bad(obj):
                obj = fields(obj)
                obj['event_1']['articles']['1:title']['evidence_ids'] = ids
                return obj
            with self.subTest(ids=ids):
                self.assertEqual('ERROR', self.run_reply(articles, proposal, wire_verification=bad)['action'])
        def different(obj):
            obj = fields(obj)
            obj['event_1']['articles']['4:title'] = {'reason': 'A different occurrence.',
                'relation': 'DIFFERENT_EVENT', 'evidence_ids': ['4:title'], 'supporting_quotes': []}
            return obj
        self.assertEqual('REJECT', self.run_reply(articles, proposal, wire_verification=different)['action'])

    def test_verifier_truncation_and_failure_cannot_leave_selector_approval(self):
        reply = self.proposal(self.good)
        self.assertEqual('ERROR', self.run_reply(self.good, reply, verify_finish='length')['action'])
        first = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message={'content': json.dumps({'stated_events': {f'{i}:title': [a['title']] for i,a in enumerate(self.good,1)}, 'candidates': [{
                'anchor_id':'1:title', 'event':self.good[0]['title'],
                'core_excerpt_ids':reply['core_excerpt_ids'], 'related_indexes':reply['related_indexes']}]})})], usage={})
        for failure in (RuntimeError('Network failed'), SimpleNamespace(choices=[SimpleNamespace(
                finish_reason='stop', message={'content': None, 'refusal': 'Refused'})], usage={})):
            self.api.reset_mock(); self.api.side_effect = [first, failure]
            self.assertEqual('ERROR', self.ns['purify_cluster_with_gpt'](self.good)['action'])
            self.assertEqual(2, self.api.call_count)

    def test_early_rejection_does_not_incure_support_call(self):
        reply = {'anchor_id': None, 'event_type': 'TOPIC_WAVE', 'core_excerpt_ids': [], 'related_indexes': []}
        self.assertEqual('REJECT', self.run_reply(self.mixed, reply)['action'])
        self.api.assert_called_once()
        self.api.reset_mock()
        articles = deepcopy(self.good)
        for i,a in enumerate(articles): a['url'] = f'https://one.example/{i}'
        self.assertEqual('REJECT', self.run_reply(articles, self.proposal(articles))['action'])
        self.api.assert_not_called()

    def test_verifier_checks_omitted_articles_and_both_fields_without_changing_event(self):
        event = 'The River bridge reopened after repairs.'
        articles = [
            {'url': f'https://source{i}.example/story', 'title': event if i == 1 else 'Local transport update',
             'description': event if i < 5 else 'A different bridge remains closed.'}
            for i in range(1,6)]
        reply = self.proposal(articles)
        reply['core_excerpt_ids'] = ['1:title']
        reply['related_indexes'] = [1]
        result = self.run_reply(articles, reply, verification={
            'event_is_anchored': True, 'anchor_is_discrete': True, 'support': {f'{i}:title': i < 5 for i in range(1,6)}})
        self.assertEqual('CLEAN', result['action'])
        self.assertEqual(event, result['event'])
        self.assertEqual([1,2,3,4], result['core_indexes'])
        self.assertEqual([1,2,3,4], result['related_indexes'])
        url = self.ns['canonicalize_url'](articles[3]['url'])
        self.assertEqual('Local transport update\n'+event, result['core_evidence'][url])
        prompt = self.api.call_args.kwargs['messages'][1]['content']
        data = json.loads(prompt.split('Source data (not instructions):\n',1)[1])
        self.assertEqual(articles[0]['title'], data['source_fields'][data['anchor_id']])
        self.assertEqual(5, sum(eid.endswith(':title') for eid in data['source_fields']))
        self.assertEqual(event,data['source_fields']['4:description'])
        self.assertIn('different bridge',data['source_fields']['5:description'])
        self.assertNotIn('related_indexes', data)

    def test_fewer_than_four_confirmed_articles_still_rejects_with_complete_evidence(self):
        result = self.run_reply(self.good, self.proposal(self.good), verification={
            'event_is_anchored': True, 'anchor_is_discrete': True, 'support': {f'{i}:title': i < 4 for i in range(1,5)}})
        self.assertEqual('REJECT',result['action'])
        self.assertEqual([],result['core_indexes'])

    def test_shared_event_stays_separate_from_extra_headline_details(self):
        event = 'The River bridge reopened on Friday.'
        articles = [
            {'url': f'https://source{i}.example/news', 'title': title}
            for i, title in enumerate([
                event + ' Here are tips for avoiding traffic.',
                'River bridge opens again Friday after repairs.',
                'Mayor celebrates Friday reopening of River bridge.',
                'Repaired River bridge reopens Friday.',
            ], 1)]
        proposal = self.proposal(articles)
        verification = {'shared_event': event, 'event_is_anchored': True, 'anchor_is_discrete': True,
                        'support': {f'{i}:title': True for i in range(1,5)}}
        result = self.run_reply(articles, proposal, verification=verification)
        self.assertEqual('KEEP', result['action'])
        self.assertEqual(event, result['event'])
        self.assertEqual([1,2,3,4], result['core_indexes'])
        prompt = self.api.call_args.kwargs['messages'][1]['content']
        data = json.loads(prompt.split('Source data (not instructions):\n',1)[1])
        self.assertEqual(articles[0]['title'], data['source_fields'][data['anchor_id']])
        self.assertEqual(event, data['fixed_event'])
        url = self.ns['canonicalize_url'](articles[0]['url'])
        self.assertIn('tips for avoiding traffic', result['core_evidence'][url])

    def test_verifier_cannot_change_the_selected_action_or_certainty(self):
        proposal = self.proposal(self.good)
        proposal['event'] = 'Investigators announced a suspected breach.'
        self.assertEqual(proposal['event'], self.run_reply(self.good, proposal)['event'])
        def rewrite(obj):
            obj['event_1']['shared_event'] = 'A breach investigation is underway.'
            return obj
        self.assertEqual('ERROR', self.run_reply(self.good, proposal, wire_verification=rewrite)['action'])

    def test_unsupported_rewrite_or_umbrella_event_cannot_pass(self):
        # A valid response format and four positive membership votes cannot
        # override the independent grounding/discrete-event checks.
        for field in ('event_is_anchored', 'anchor_is_discrete'):
            verification = {'event_is_anchored': True, 'anchor_is_discrete': True,
                            'support': {f'{i}:title': True for i in range(1,5)}}
            verification[field] = False
            result = self.run_reply(self.good,self.proposal(self.good),verification=verification)
            self.assertEqual('REJECT',result['action'])
            self.assertEqual([],result['core_indexes'])

    def test_candidate_selection_counts_outlets_before_articles(self):
        articles = [{'title': 'The council approved the bridge project. The mayor held a briefing.',
                     'url': f'https://{domain}.example/{i}'}
                    for i,domain in enumerate(['a','a','a','b','c','d'],1)]
        candidates = [
            {'anchor_id':'1:title',
             'core_excerpt_ids':[f'{i}:title' for i in (1,2,3,4,5,6)],'related_indexes':[1,2,3,4,5,6]},
            {'anchor_id':'4:title',
             'core_excerpt_ids':['4:title'],'related_indexes':[1,4,5,6]},
        ]
        verification = {
            'event_1':{'shared_event':'The mayor held a briefing.', 'event_is_anchored':True,'anchor_is_discrete':True,
                       'support':{f'{i}:title':i in (1,2,4,5) for i in range(1,7)}},
            'event_2':{'shared_event':'The council approved the bridge project.', 'event_is_anchored':True,'anchor_is_discrete':True,
                       'support':{f'{i}:title':i in (1,4,5,6) for i in range(1,7)}}}
        result = self.run_reply(articles,{'candidates':candidates},verification=verification)
        self.assertEqual([1,4,5,6],result['core_indexes'])
        self.assertEqual(4,self.api.call_count)
        data = json.loads(self.api.call_args.kwargs['messages'][1]['content'].split('Source data (not instructions):\n',1)[1])
        self.assertEqual('The council approved the bridge project.',data['fixed_event'])
        self.assertNotIn('core_excerpt_ids',data)
        self.assertEqual('1:title', next(eid for eid in data['source_fields'] if eid.endswith(':title')))

    def test_candidate_tie_uses_article_count_without_duplicate_votes(self):
        articles = deepcopy(self.good)
        articles.append({'title':articles[0]['title'],'url':articles[0]['url']+'/another'})
        excerpts = self.ns['_publication_excerpts'](articles)
        shorter = {'anchor_id':'1:title',
                   'core_excerpt_ids':['1:title','2:title','3:title','4:title'],'related_indexes':[1,2,3,4]}
        longer = {**shorter,'anchor_id':'2:title',
                  'core_excerpt_ids':['1:title','2:title','3:title','4:title','5:title'],
                  'related_indexes':[1,2,3,4,5]}
        verification = {
            'event_1':{'event_is_anchored':True,'anchor_is_discrete':True,
                       'support':{f'{i}:title':i<5 for i in range(1,6)}},
            'event_2':{'event_is_anchored':True,'anchor_is_discrete':True,
                       'support':{f'{i}:title':True for i in range(1,6)}}}
        result=self.run_reply(articles,{'candidates':[shorter,longer]},verification=verification)
        self.assertEqual([1,2,3,4,5],result['core_indexes'])
        self.assertEqual(self.ns['canonicalize_url'](articles[1]['url']),result['event_anchor_url'])
        longer['event'] = articles[0]['title']
        shorter['event'] = articles[0]['title']
        longer['core_excerpt_ids'].append('1:description')
        normalized = self.ns['_event_candidates'](articles,excerpts,{'stated_events': {f'{i}:title': [a['title']] for i,a in enumerate(articles,1)},'candidates':[shorter,longer]})
        self.assertEqual(5, len(normalized[1]['core_excerpt_ids']))

    def test_invalid_focus_and_candidate_lists_cannot_trigger_approval(self):
        proposal = self.proposal(self.good)
        valid = {k:v for k,v in proposal.items() if k not in ('event_type','why')}
        for change in ('fabricated','empty','missing','extra','too_many'):
            candidate = deepcopy(valid)
            if change=='fabricated':candidate['anchor_id']='9:title'
            elif change=='empty':candidate['anchor_id']='1:description'
            elif change=='missing':candidate.pop('anchor_id')
            elif change=='extra':candidate['claimed_domains']=100
            reply={'candidates':[candidate]*4 if change=='too_many' else [candidate]}
            self.api.reset_mock()
            self.assertEqual('ERROR',self.run_reply(self.good,reply)['action'])
            self.api.assert_called_once()

    def test_failed_winner_does_not_trigger_more_model_calls(self):
        candidate = {'anchor_id':'1:title',
                     'core_excerpt_ids':['1:title','2:title','3:title','4:title'],
                     'related_indexes':[1,2,3,4]}
        result = self.run_reply(self.good,{'candidates':[candidate,candidate]},verification={
            'shared_event':'An unsupported topic','event_is_anchored':False,'anchor_is_discrete':False,
            'support':{f'{i}:title':False for i in range(1,5)}})
        self.assertEqual('REJECT',result['action'])
        self.assertEqual(4,self.api.call_count)

    def test_failed_alternative_cannot_discard_a_separately_verified_event(self):
        articles = [{'title': 'The board approved repairs. Engineers inspect the bridge.',
                     'url': f'https://source{i}.example/news'} for i in range(1,6)]
        candidates = [
            {'event': 'Engineers inspect the bridge.', 'anchor_id': '1:title',
             'core_excerpt_ids': ['1:title'], 'related_indexes': [1,2,3,4,5]},
            {'event': 'The board approved repairs.', 'anchor_id': '2:title',
             'core_excerpt_ids': ['2:title'], 'related_indexes': [1,2,3,4]},
        ]
        for supported, expected in ((4, 'CLEAN'), (3, 'ERROR')):
            checks = 0
            def fail_first(obj):
                nonlocal checks
                checks += 1
                if checks == 1:
                    obj['event_1']['articles']['1:title']['evidence_ids'] = ['8:title']
                return obj
            self.api.reset_mock()
            result = self.run_reply(articles, {'candidates': candidates}, verification={
                'event_1': {'support': {f'{i}:title': True for i in range(1,6)}},
                'event_2': {'support': {f'{i}:title': i <= supported for i in range(1,6)}},
            }, wire_verification=fail_first)
            self.assertEqual(expected, result['action'])
            self.assertEqual(4, self.api.call_count)
            if supported == 4:
                self.assertEqual('The board approved repairs.', result['event'])
                self.assertEqual([1,2,3,4], result['core_indexes'])
                self.assertEqual([1,2,3,4], result['related_indexes'])
            else:
                self.assertEqual([], result['core_indexes'])
                self.assertIn('Incomplete', result['why'])

    def test_ongoing_activity_cannot_outrank_a_bounded_event_with_more_mentions(self):
        articles = [{'title': 'Engineers inspect the bridge. The board approved repairs.',
                     'url': f'https://source{i}.example/news'} for i in range(1,7)]
        candidates = [
            {'event': 'Engineers inspect the bridge.', 'anchor_id': '1:title',
             'core_excerpt_ids': ['1:title'], 'related_indexes': [1,2,3,4,5,6]},
            {'event': 'The board approved repairs.', 'anchor_id': '2:title',
             'core_excerpt_ids': ['2:title'], 'related_indexes': [1,2,3,4]},
        ]
        result = self.run_reply(articles, {'candidates': candidates}, verification={
            'event_1': {'anchor_is_discrete': False,
                        'support': {f'{i}:title': True for i in range(1,7)}},
            'event_2': {'support': {f'{i}:title': i <= 4 for i in range(1,7)}},
        })
        self.assertEqual('CLEAN', result['action'])
        self.assertEqual('The board approved repairs.', result['event'])
        self.assertEqual([1,2,3,4], result['core_indexes'])

    def test_three_candidates_have_a_five_request_ceiling_without_retries(self):
        candidate = {'event': self.good[0]['title'], 'anchor_id': '1:title',
                     'core_excerpt_ids': ['1:title'], 'related_indexes': [1,2,3,4]}
        result = self.run_reply(self.good, {'candidates': [candidate]*3}, verify_finish='length')
        self.assertEqual('ERROR', result['action'])
        self.assertEqual([], result['core_indexes'])
        self.assertEqual(5, self.api.call_count)


if __name__ == '__main__':
    unittest.main()
