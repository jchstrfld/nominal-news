"""Offline checkpoint integration checks with synthetic source evidence."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_next_signals as pipeline
import next_signal_checkpoint_review as checkpoint


def fixture():
    sentence = 'The panel will publish its findings on October 18, 2034.'
    return {'edition': '2034-10-10',
            'topic': {'topic_title': 'Panel investigation', 'summary': 'The panel is reviewing evidence.'},
            'evidence': [{'title': 'Panel investigation', 'description': sentence,
                          'grounded_sentence': sentence, 'published_at': '2034-10-10',
                          'url': 'https://example.test/panel'}]}


def receipt(worker, case, row, claim='TARGET_SCHEDULED'):
    """Human-authored contract control, not a saved model judgment."""
    prompt = json.loads(worker.request(case, row)['messages'][1]['content'])
    return {'assessments': [{
        'candidate_id': row['candidate_id'], 'central_story_state': 'Panel investigation.',
        'scheduled_change': {'kind': 'DECISION_OR_UPDATE', 'description': 'Publication of findings.'},
        'story_connection': {'kind': 'EXPLICIT_NEXT_STEP', 'evidence_id': 1,
                             'reason': 'The findings resolve the same investigation.'},
        'claims': [{'evidence_id': item['id'],
                    'claim': claim if item['id'] == 1 else 'OTHER_EVENT_OR_NO_STATUS',
                    'reason': 'Synthetic source assessment.'}
                   for item in prompt['evidence_for_status_comparison']],
        'reason': 'Synthetic offline contract control.'}]}


@contextlib.contextmanager
def memory_run(cases, cache, *, enabled=True, prepared=True, extra=()):
    """Exercise main's routing and writes; never touch the published edition."""
    topics = [{**deepcopy(case['topic']), 'next_signal': 'Previous verified signal',
               'untouched': {'images': ['fixed'], 'counts': [1, 2]}} for case in cases]
    # Caller builds receipts against the real topic shape used by the pipeline.
    files = {'summary': topics, 'clusters': [{'evidence': deepcopy(c['evidence'])} for c in cases],
             'checkpoint-cache': deepcopy(cache), 'report': [{'previous': True}], 'legacy-cache': {}}
    writes = []
    def load(path, default):
        return deepcopy(files.get(path, default))
    def save(path, value):
        files[path] = deepcopy(value)
        writes.append(path)
    def match(topic, clusters):
        index = next(i for i, c in enumerate(cases) if c['topic']['topic_title'] == topic['topic_title'])
        return index, {'url_overlap': 1}
    argv = ['next-signal', '--date', cases[0]['edition'], '--summary-file', 'summary',
            '--cluster-file', 'clusters', '--article-pool-file', 'missing-pool',
            '--report-file', 'report', '--cache-file', 'legacy-cache',
            '--sidecar-cache-file', 'sidecar-cache', '--no-sidecar', '--no-external',
            '--checkpoint-cache-file', 'checkpoint-cache', *extra]
    if enabled:
        argv.append('--checkpoint-review')
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(sys, 'argv', argv))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline only')))
        api = stack.enter_context(patch.object(pipeline.openai.ChatCompletion, 'create',
                                              side_effect=AssertionError('No model requests')))
        stack.enter_context(patch.object(pipeline, '_load_json', side_effect=load))
        stack.enter_context(patch.object(pipeline, '_save_json_atomic', side_effect=save))
        stack.enter_context(patch.object(pipeline, 'match_topic_to_cluster', side_effect=match))
        stack.enter_context(patch.object(pipeline, 'select_signal_evidence', side_effect=lambda c, d: c['evidence']))
        stack.enter_context(patch.object(pipeline, '_rank_sidecar_candidates', return_value=[]))
        if prepared:
            stack.enter_context(patch.object(pipeline, '_model_evidence_candidates', side_effect=lambda d, e, **kw: deepcopy(e)))
        external = stack.enter_context(patch.object(pipeline, 'build_external_signal_evidence',
                                                   side_effect=AssertionError('No external research')))
        yield files, writes, api, external


class CheckpointIntegrationChecks(unittest.TestCase):
    def setUp(self):
        self.case = fixture()
        self.worker = checkpoint.ReviewAdapter(pipeline)

    def cache_for(self, cases, claim='TARGET_SCHEDULED'):
        cache = {}
        for case in cases:
            actual = deepcopy(case)
            actual['topic'].update(next_signal='Previous verified signal',
                                   untouched={'images': ['fixed'], 'counts': [1, 2]})
            for row in self.worker.candidates(actual):
                self.worker.remember(cache, actual, row, receipt(self.worker, actual, row, claim))
        return cache

    def test_opt_in_replaces_only_next_signal_with_original_source_mapping(self):
        cache = self.cache_for([self.case])
        with memory_run([self.case], cache) as (files, writes, api, external):
            before = deepcopy(files['summary'])
            with patch.object(pipeline, 'evaluate_signal_triage', side_effect=AssertionError('No legacy route')):
                pipeline.main()
            expected = deepcopy(before)
            expected[0]['next_signal'] = self.case['evidence'][0]['grounded_sentence']
            self.assertEqual(expected, files['summary'])
            self.assertEqual(['summary', 'report'], writes)
            report = files['report'][0]
            self.assertEqual('checkpoint_review', report['signal_stage'])
            self.assertEqual(0, report['model_call_count'])
            self.assertEqual(checkpoint.CONTRACT_VERSION, report['validation_prompt_version'])
            self.assertEqual(self.case['evidence'][0]['url'], report['supporting_source']['url'])
            self.assertFalse(report['external_enrichment_enabled'])
            self.assertEqual(cache, files['checkpoint-cache'])
            api.assert_not_called()
            external.assert_not_called()

    def test_matching_weekday_date_reaches_full_review_before_approval(self):
        sentence = 'The panel will publish its findings on Wednesday, October 18, 2034.'
        self.case['evidence'][0].update(description=sentence, grounded_sentence=sentence)
        rows = self.worker.candidates(self.case)
        self.assertTrue(rows)
        self.assertTrue(all(r['window'] == ('2034-10-18', '2034-10-18') for r in rows))
        with memory_run([self.case], {}) as (files, writes, _, _):
            before = deepcopy(files)
            with self.assertRaises(RuntimeError):
                pipeline.main()
            self.assertEqual(before, files)
            self.assertEqual([], writes)
        with memory_run([self.case], self.cache_for([self.case])) as (files, _, api, _):
            pipeline.main()
            self.assertEqual(sentence, files['summary'][0]['next_signal'])
            self.assertTrue(files['report'][0]['candidate_validation_decision']['all_candidates_reviewed'])
            api.assert_not_called()

    def test_conflicting_weekday_cannot_use_explicit_year_fallback(self):
        for unverified in (False, True):
            case = deepcopy(self.case)
            sentence = 'The panel will publish its findings on Thursday, October 18, 2034.'
            case['evidence'][0].update(description=sentence, grounded_sentence=sentence,
                                       reference_date_unverified=unverified)
            rows = self.worker.candidates(case)
            self.assertTrue(rows)
            self.assertTrue(all(r['window'] is None for r in rows))
            result = self.worker.evaluate(case['edition'], case['topic'], case['evidence'], cache={})
            self.assertFalse(result.decision['eligible'])

    def test_unverified_reference_still_requires_explicit_year(self):
        for timing, expected in (('October 18', None), ('October 18, 2034', ('2034-10-18', '2034-10-18'))):
            case = deepcopy(self.case)
            sentence = f'The panel will publish its findings on Wednesday, {timing}.'
            case['evidence'][0].update(description=sentence, grounded_sentence=sentence,
                                       reference_date_unverified=True)
            rows = self.worker.candidates(case)
            self.assertTrue(rows)
            self.assertTrue(all(r['window'] == expected for r in rows))

    def test_explicit_future_date_keeps_announcement_weekday_as_background(self):
        for unverified in (False, True):
            case = deepcopy(self.case)
            sentence = ('Officials announced on Monday that the panel will publish its findings '
                        'on Wednesday, October 18, 2034.')
            case['evidence'][0].update(description=sentence, grounded_sentence=sentence,
                                       reference_date_unverified=unverified)
            rows = self.worker.candidates(case)
            self.assertTrue(rows)
            self.assertTrue(all(r['window'] == ('2034-10-18', '2034-10-18') for r in rows))
            conflicting = sentence.replace('Wednesday,', 'Thursday,')
            case['evidence'][0].update(description=conflicting, grounded_sentence=conflicting)
            rows = self.worker.candidates(case)
            self.assertTrue(rows)
            self.assertTrue(all(r['window'] is None for r in rows))

    def test_output_field_does_not_invalidate_its_own_cache(self):
        case = deepcopy(self.case)
        case['topic']['next_signal'] = 'Previous output'
        row = self.worker.candidates(case)[0]
        key = self.worker.cache_identity(case, row)[0]
        case['topic']['next_signal'] = ''
        self.assertEqual(key, self.worker.cache_identity(case, row)[0])
        cache = self.cache_for([self.case])
        with memory_run([self.case], cache) as (files, writes, api, _):
            pipeline.main()
            first = deepcopy(files['summary'])
            pipeline.main()
            self.assertEqual(first, files['summary'])
            self.assertEqual(0, files['report'][0]['model_call_count'])
            api.assert_not_called()

    def test_context_binding_survives_final_pipeline_guards(self):
        source = self.case['evidence'][0]
        expected = source['grounded_sentence']
        source.update(grounded_sentence='The hearing has ended.', description='The hearing has ended.',
                      source_context='Background. ' + expected)
        with memory_run([self.case], self.cache_for([self.case])) as (files, _, _, _):
            pipeline.main()
            self.assertEqual(expected, files['summary'][0]['next_signal'])
            support = files['report'][0]['supporting_source']
            self.assertEqual(expected, support['grounded_sentence'])
            self.assertEqual(source['url'], support['url'])
            self.assertEqual('source_context', files['report'][0]['candidate_validation_decision']['source_binding']['field'])

    def test_missing_later_story_review_preserves_entire_previous_edition(self):
        second = deepcopy(self.case)
        second['topic']['topic_title'] = 'Another investigation'
        cache = self.cache_for([self.case])
        with memory_run([self.case, second], cache) as (files, writes, _, _):
            before = deepcopy(files)
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                pipeline.main()
            self.assertEqual(before, files)
            self.assertEqual([], writes)

    def test_cache_corruption_stops_instead_of_publishing_a_blank(self):
        for cache in ([], {'legacy-final-decision': {'eligible': True}}):
            with memory_run([self.case], cache) as (files, writes, _, _):
                before = deepcopy(files)
                with self.assertRaises((ValueError, RuntimeError)):
                    pipeline.main()
                self.assertEqual(before, files)
                self.assertEqual([], writes)
        cache = self.cache_for([self.case])
        next(iter(cache.values()))['response']['assessments'][0]['claims'].pop()
        with memory_run([self.case], cache) as (files, writes, _, _):
            before = deepcopy(files)
            with self.assertRaises(RuntimeError):
                pipeline.main()
            self.assertEqual(before, files)
            self.assertEqual([], writes)

    def test_verified_rejection_can_clear_previous_signal_without_other_changes(self):
        with memory_run([self.case], self.cache_for([self.case], 'TARGET_CANCELLED')) as (files, _, _, _):
            before = deepcopy(files['summary'])
            pipeline.main()
            before[0]['next_signal'] = ''
            self.assertEqual(before, files['summary'])
            self.assertFalse(files['report'][0]['eligible'])

    def test_default_path_does_not_initialize_or_read_checkpoint_cache(self):
        with memory_run([self.case], [], enabled=False) as (files, _, api, _):
            with (patch.object(checkpoint, 'ReviewAdapter', side_effect=AssertionError('Default changed')),
                  patch.object(pipeline, 'evaluate_signal_triage', return_value={'status': 'NO_SIGNAL', 'reason': 'Control'}) as triage):
                pipeline.main()
            triage.assert_called_once()
            api.assert_not_called()
            self.assertEqual(pipeline.PROMPT_VERSION, files['report'][0]['prompt_version'])

    def test_opt_in_ignores_legacy_final_decision_cache(self):
        with memory_run([self.case], self.cache_for([self.case])) as (files, _, _, _):
            files['legacy-cache'] = {'unrelated-old-approval': {'eligible': True}}
            pipeline.main()
            self.assertEqual({'unrelated-old-approval': {'eligible': True}}, files['legacy-cache'])

    def test_dry_run_does_not_read_receipts_or_write(self):
        with memory_run([self.case], [], extra=['--dry-run']) as (_, writes, _, _):
            with patch.object(checkpoint, 'ReviewAdapter', side_effect=AssertionError('Dry run initialized reviewer')):
                pipeline.main()
            self.assertEqual([], writes)

    def test_refresh_cannot_trigger_fresh_requests_in_replay_mode(self):
        with memory_run([self.case], {}, extra=['--refresh']) as (_, writes, api, _):
            with self.assertRaisesRegex(ValueError, '--refresh'):
                pipeline.main()
            self.assertEqual([], writes)
            api.assert_not_called()


class CheckpointSentenceSelectionChecks(unittest.TestCase):
    """Synthetic contract controls; model accuracy is checked with saved receipts."""
    def setUp(self):
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline only')))
        self.worker = checkpoint.ReviewAdapter(pipeline)
        self.case = fixture()
        self.case['evidence'] = []
        for index, sentence in enumerate((
            'The panel is meeting on October 18, 2034.',
            'The panel will meet on October 18, 2034.',
        )):
            self.case['evidence'].append({'description': sentence, 'grounded_sentence': sentence,
                                         'published_at': '2034-10-10', 'url': f'https://example.test/{index}'})

    def cache(self, transform=None):
        cache = {}
        for row in self.worker.candidates(self.case):
            answer = receipt(self.worker, self.case, row)
            if transform:
                transform(row, answer['assessments'][0])
            self.worker.remember(cache, self.case, row, answer)
        return cache

    def evaluate(self, cache):
        return self.worker.evaluate(self.case['edition'], self.case['topic'], self.case['evidence'], cache=cache)

    def test_reuses_fully_reviewed_same_date_wording_with_original_provenance(self):
        cache = self.cache()
        before_case, before_cache = deepcopy(self.case), deepcopy(cache)
        result = self.evaluate(cache)
        self.assertTrue(result.decision['eligible'])
        self.assertEqual(self.case['evidence'][1]['grounded_sentence'], result.decision['signal'])
        self.assertEqual(2, result.decision['source_index'])
        self.assertEqual('https://example.test/1', result.evidence[1]['url'])
        self.assertEqual(2, result.decision['source_binding']['source_index'])
        self.assertTrue(result.decision['all_candidates_reviewed'])
        self.assertEqual(1, len(result.decision['display_rejections']))
        self.assertIn('future', result.decision['display_rejections'][0]['reason'])
        self.assertEqual(0, result.review_calls)
        self.assertEqual(before_case, self.case)
        self.assertEqual(before_cache, cache)

    def test_cannot_skip_an_unprintable_earliest_date_for_a_later_one(self):
        sentence = 'The panel will meet on October 20, 2034.'
        self.case['evidence'][1].update(description=sentence, grounded_sentence=sentence)
        result = self.evaluate(self.cache())
        self.assertFalse(result.decision['eligible'])
        self.assertEqual('', result.decision['signal'])
        self.assertEqual(1, len(result.decision['display_rejections']))

    def test_same_date_unrelated_sentence_is_not_a_substitute(self):
        def unrelated(row, answer):
            if row['source_index'] == 2:
                answer['story_connection'].update(kind='SHARED_SUBJECT_ONLY', evidence_id=1)
        result = self.evaluate(self.cache(unrelated))
        self.assertFalse(result.decision['eligible'])

    def test_uncertain_or_cancelled_candidate_cannot_be_hidden_as_bad_wording(self):
        for claim in ('UNCERTAIN', 'TARGET_CANCELLED'):
            def conflicting(row, answer):
                if row['source_index'] == 1:
                    answer['claims'][0]['claim'] = claim
            with self.subTest(claim=claim):
                result = self.evaluate(self.cache(conflicting))
                self.assertFalse(result.decision['eligible'])
                self.assertEqual([], result.decision['display_rejections'])

    def test_missing_review_cannot_be_replaced_with_a_display_choice(self):
        cache = self.cache()
        row = next(r for r in self.worker.candidates(self.case) if r['source_index'] == 1)
        del cache[self.worker.cache_identity(self.case, row)[0]]
        with self.assertRaises(checkpoint.CacheMiss):
            self.evaluate(cache)

    def test_every_alternative_keeps_the_final_chronology_guard(self):
        sentence = 'The panel will gather on October 18, 2034.'
        self.case['evidence'][0].update(description=sentence, grounded_sentence=sentence)
        cache = self.cache()
        with patch.object(pipeline, '_unresolved_future_alternative', side_effect=['Unresolved timing control', '']) as guard:
            result = self.evaluate(cache)
        self.assertTrue(result.decision['eligible'])
        self.assertEqual(2, guard.call_count)
        self.assertTrue(all(len(call.args[1]) == 2 for call in guard.call_args_list))
        self.assertEqual('unresolved_timing', result.decision['display_rejections'][0]['reason_code'])
        with patch.object(pipeline, '_unresolved_future_alternative', return_value='Unresolved timing control') as guard:
            result = self.evaluate(cache)
        self.assertFalse(result.decision['eligible'])
        self.assertEqual(2, guard.call_count)

    def test_binding_failure_raises_instead_of_trying_another_source(self):
        with patch.object(self.worker, '_bound_evidence', side_effect=checkpoint.ReviewError('Broken source binding')):
            with self.assertRaisesRegex(checkpoint.ReviewError, 'Broken source binding'):
                self.evaluate(self.cache())

    def test_pipeline_fallback_changes_only_next_signal_and_keeps_selected_source(self):
        self.case['topic'].update(next_signal='Previous verified signal',
                                  untouched={'images': ['fixed'], 'counts': [1, 2]})
        cache = self.cache()
        with memory_run([self.case], cache) as (files, writes, api, external):
            before = deepcopy(files['summary'])
            pipeline.main()
            before[0]['next_signal'] = self.case['evidence'][1]['grounded_sentence']
            self.assertEqual(before, files['summary'])
            self.assertEqual('https://example.test/1', files['report'][0]['supporting_source']['url'])
            self.assertEqual(['summary', 'report'], writes)
            api.assert_not_called()
            external.assert_not_called()


class CheckpointContextChecks(unittest.TestCase):
    """Offline provenance and gate checks, not measurements of model accuracy."""
    def setUp(self):
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline only')))
        self.worker = checkpoint.ReviewAdapter(pipeline)
        self.case = fixture()
        self.prior = 'The panel completed its hearing.'
        self.target = self.case['evidence'][0]['grounded_sentence']
        self.context = self.prior + ' ' + self.target
        self.case['evidence'][0]['source_context'] = self.context

    def items(self, case=None):
        case = case or self.case
        row = self.worker.candidates(case)[0]
        request = self.worker.request(case, row)
        return json.loads(request['messages'][1]['content'])['evidence_for_status_comparison']

    def test_excerpt_links_to_its_literal_context_without_changing_evidence(self):
        before = deepcopy(self.case)
        row = self.worker.candidates(self.case)[0]
        items = self.items()
        original = checkpoint._evidence(self.case, row)
        self.assertEqual(original, [{k: v for k, v in item.items() if k != 'source_refs'} for item in items])
        fragment = next(i for i in items if i['text'] == self.prior)
        context = next(i for i in items if i['text'] == self.context)
        self.assertEqual([{'source_index': 1, 'fields': ['source_context'],
                           'context_evidence_ids': [context['id']]}], fragment['source_refs'])
        self.assertEqual(before, self.case)

    def test_summary_does_not_borrow_article_provenance_even_when_text_matches(self):
        self.case['topic']['summary'] = self.prior
        items = self.items()
        summary = next(i for i in items if i['origin'] == 'newsletter_summary')
        self.assertEqual(self.prior, summary['text'])
        self.assertEqual([], summary['source_refs'])
        self.assertTrue(next(i for i in items if i['text'] == self.context)['source_refs'])

    def test_repeated_excerpt_keeps_separate_source_contexts_and_all_fields(self):
        second_context = self.prior + ' The panel has not announced its next step.'
        self.case['evidence'][0]['source_overview'] = self.context
        self.case['evidence'].append({'url': 'https://example.test/second',
                                     'source_context': second_context})
        items = self.items()
        fragment = next(i for i in items if i['text'] == self.prior)
        contexts = {i['text']: i['id'] for i in items}
        self.assertEqual([
            {'source_index': 1, 'fields': ['source_overview', 'source_context'],
             'context_evidence_ids': [contexts[self.context]]},
            {'source_index': 2, 'fields': ['source_context'],
             'context_evidence_ids': [contexts[second_context]]}], fragment['source_refs'])
        prompt = json.loads(self.worker.request(self.case, self.worker.candidates(self.case)[0])['messages'][1]['content'])
        self.assertEqual(1, prompt['target_event']['source_index'])
        self.assertEqual([{'source_index': i, 'source_url': s['url']}
                          for i, s in enumerate(self.case['evidence'], 1)], prompt['sources'])

    def test_other_article_cannot_supply_context_for_a_unique_excerpt(self):
        different = 'The panel completed its investigation.'
        self.case['evidence'].append({'url': 'https://example.test/other',
                                     'source_context': different})
        items = self.items()
        other = next(i for i in items if i['text'] == different)
        self.assertEqual([{'source_index': 2, 'fields': ['source_context'],
                           'context_evidence_ids': []}], other['source_refs'])
        fragment = next(i for i in items if i['text'] == self.prior)
        self.assertNotIn(other['id'], fragment['source_refs'][0]['context_evidence_ids'])

    def test_paraphrase_is_not_given_a_literal_source_binding(self):
        row = self.worker.candidates(self.case)[0]
        row['proof_quotes'].append('The panel finished hearing testimony.')
        items = json.loads(self.worker.request(self.case, row)['messages'][1]['content'])['evidence_for_status_comparison']
        self.assertEqual([], next(i for i in items if i['text'] == row['proof_quotes'][-1])['source_refs'])

    def test_distinct_prior_steps_can_coexist_with_an_explicit_future_target(self):
        # Human-authored classifications exercise the unchanged publication gates.
        for prior, target in (
                ('The panel completed its hearing.', 'The panel will issue its ruling on October 18, 2034.'),
                ('Officials approved the new rules.', 'The new rules will take effect on October 18, 2034.'),
                ('The venue announced its program.', 'The venue will host the performance on October 18, 2034.'),
                ('The curfew has started.', 'The curfew will end on October 18, 2034.')):
            with self.subTest(target=target):
                case = fixture()
                case['evidence'][0].update(grounded_sentence=target, description=target,
                                           source_context=prior + ' ' + target)
                cache = {}
                for row in self.worker.candidates(case):
                    answer = receipt(self.worker, case, row)
                    answer['assessments'][0]['scheduled_change']['description'] = target
                    if prior.startswith('The curfew'):
                        answer['assessments'][0]['scheduled_change']['kind'] = 'END_OR_EXPIRY'
                    items = checkpoint._evidence(case, row)
                    for claim in answer['assessments'][0]['claims']:
                        text = items[claim['evidence_id'] - 1]['text']
                        claim.update(claim='TARGET_SCHEDULED' if target in text else 'OTHER_EVENT_OR_NO_STATUS',
                                     reason='Synthetic control: the prior step is distinct from the future target.')
                    self.worker.remember(cache, case, row, answer)
                result = self.worker.evaluate(case['edition'], case['topic'], case['evidence'], cache=cache)
                self.assertTrue(result.decision['eligible'], result.decision)
                self.assertEqual(target, result.decision['signal'])
                self.assertEqual(0, result.review_calls)

    def test_context_links_never_override_a_target_veto_or_uncertainty(self):
        for status, contrary in (
                ('TARGET_STARTED', 'The panel has begun publishing its findings.'),
                ('TARGET_COMPLETED', 'The panel has already published its findings.'),
                ('TARGET_CANCELLED', 'The panel cancelled publication of its findings.'),
                ('UNCERTAIN', 'The panel may have published its findings.')):
            with self.subTest(status=status):
                case = fixture()
                case['evidence'][0]['source_context'] = self.target + ' ' + contrary
                row = self.worker.candidates(case)[0]
                answer = receipt(self.worker, case, row)
                items = self.items(case)
                veto = next(i for i in items if i['text'] == contrary)
                self.assertTrue(veto['source_refs'][0]['context_evidence_ids'])
                answer['assessments'][0]['claims'][veto['id'] - 1].update(
                    claim=status, reason='Synthetic control: this status concerns the exact target action.')
                cache = {}
                self.worker.remember(cache, case, row, answer)
                result = self.worker.evaluate(case['edition'], case['topic'], case['evidence'], cache=cache)
                self.assertFalse(result.decision['eligible'])
                self.assertEqual(0, result.review_calls)

    def test_previous_request_receipt_cannot_certify_contextual_review(self):
        row = self.worker.candidates(self.case)[0]
        current = self.worker.request(self.case, row)
        previous = deepcopy(current)
        prompt = json.loads(previous['messages'][1]['content'])
        prompt['task'] = checkpoint.PROMPT
        prompt.pop('sources')
        prompt['target_event'].pop('source_index')
        for item in prompt['evidence_for_status_comparison']:
            item.pop('source_refs')
        previous['messages'][1]['content'] = json.dumps(prompt, ensure_ascii=False, separators=(',', ':'))
        cache = {}
        answer = receipt(self.worker, self.case, row)
        with patch.object(self.worker, 'request', return_value=previous):
            self.worker.remember(cache, self.case, row, answer)
            old_key = self.worker.cache_identity(self.case, row)[0]
        self.assertNotEqual(old_key, self.worker.cache_identity(self.case, row)[0])
        self.assertEqual(current['model'], previous['model'])
        self.assertEqual(current['max_completion_tokens'], previous['max_completion_tokens'])
        self.assertEqual(current['response_format'], previous['response_format'])
        with self.assertRaises(checkpoint.CacheMiss):
            self.worker.evaluate(self.case['edition'], self.case['topic'], self.case['evidence'], cache=cache)


if __name__ == '__main__':
    unittest.main()
