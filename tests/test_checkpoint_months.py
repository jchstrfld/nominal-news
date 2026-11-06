"""Offline calendar-window controls. Synthetic receipts do not certify events."""
from copy import deepcopy
import socket
import unittest
from unittest.mock import patch

from test_checkpoint_review import fixture, memory_run, receipt
import generate_next_signals as pipeline
import next_signal_checkpoint_review as checkpoint


class ExplicitMonthChecks(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline only')))
        self.worker = checkpoint.ReviewAdapter(pipeline)

    def case(self, *sentences):
        case = fixture()
        source = case['evidence'][0]
        case['evidence'] = [dict(source, grounded_sentence=s, description=s,
                                 url=f'https://example.test/{i}') for i, s in enumerate(sentences)]
        return case

    def cache(self, case, change=None):
        cache = {}
        for row in self.worker.candidates(case):
            answer = receipt(self.worker, case, row)
            if change:
                change(answer['assessments'][0])
            self.worker.remember(cache, case, row, answer)
        return cache

    def evaluate(self, case, change=None):
        return self.worker.evaluate(case['edition'], case['topic'], case['evidence'],
                                    cache=self.cache(case, change))

    def test_explicit_month_bounds_include_year_rollover_and_leap_day(self):
        for timing, expected in (
                ('January 2035', ('2035-01-01', '2035-01-31')),
                ('Jan. 2035', ('2035-01-01', '2035-01-31')),
                ('February 2035', ('2035-02-01', '2035-02-28')),
                ('February 2036', ('2036-02-01', '2036-02-29')),
                ('December 2034', ('2034-12-01', '2034-12-31'))):
            with self.subTest(timing=timing):
                sentence = f'The panel will publish findings in {timing}.'
                case = self.case(sentence)
                rows = self.worker.candidates(case)
                self.assertEqual(1, len(rows))
                self.assertEqual(expected, rows[0]['window'])
                self.assertEqual('month', rows[0]['window_precision'])
                result = self.evaluate(case)
                self.assertTrue(result.decision['eligible'])
                self.assertEqual(sentence, result.decision['signal'])
                self.assertEqual(0, result.review_calls)

    def test_month_in_another_clause_is_discovered_without_inventing_a_day(self):
        sentence = ('With the panel publishing findings in January 2035, '
                    'officials will plan the next phase.')
        case = self.case(sentence)
        row = self.worker.candidates(case)[0]
        self.assertEqual('explicit_month_year', row['discovery_kind'])
        self.assertEqual('January 2035', row['timing_quote'])
        result = self.evaluate(case)
        self.assertTrue(result.decision['eligible'])
        self.assertEqual(sentence, result.decision['signal'])
        self.assertFalse(pipeline.validate_generated_signal_grounding(
            case['edition'], sentence.replace('January 2035', 'January 1, 2035'), case['evidence'][0])[0])
        with self.assertRaises(checkpoint.CacheMiss):
            self.worker.evaluate(case['edition'], case['topic'], case['evidence'], cache={})

    def test_ambiguous_qualified_and_partly_elapsed_months_remain_blocked(self):
        timings = ('in January', 'on January 3', 'in 2035', 'in October 2034', 'in September 2034',
                   'by January 2035', 'before January 2035', 'after January 2035',
                   'until January 2035', 'from January 2035', 'around January 2035',
                   'as early as January 2035', 'in mid-January 2035',
                   'in January 2035 or later', 'in January 2035 or 2036',
                   'in January 2035, at the earliest', 'in January 2035 at the latest',
                   'between January and February 2035', 'in January-February 2035',
                   'in January 2035 through March 2035', 'on Friday in January 2035')
        for timing in timings:
            with self.subTest(timing=timing):
                case = self.case(f'The panel will publish findings {timing}.')
                self.assertIsNone(pipeline._explicit_future_month_window(case['edition'], case['evidence'][0]))
                result = self.worker.evaluate(case['edition'], case['topic'], case['evidence'], cache={})
                self.assertFalse(result.decision['eligible'])
                self.assertEqual(0, result.review_calls)

    def test_unverified_missing_future_and_stale_source_dates_do_not_gain_approval(self):
        for changes in ({'reference_date_unverified': True}, {'published_at': ''},
                        {'published_at': '2035-01-02'}):
            with self.subTest(changes=changes):
                case = self.case('The panel will publish findings in January 2035.')
                case['evidence'][0].update(changes)
                rows = self.worker.candidates(case)
                if changes.get('published_at') == '2035-01-02':
                    self.assertTrue(all(r['window'] is None for r in rows))
                else:
                    # Explicit month/year bounds require no inferred reference
                    # date, but this source still cannot authorize publication.
                    self.assertTrue(all(r['window'] == ('2035-01-01', '2035-01-31') for r in rows))
                    self.assertTrue(all(not r['publishable_source_copy'] for r in rows))
                self.assertFalse(self.evaluate(case).decision['eligible'])
        case = self.case('The panel will publish findings in September 2034.')
        case['evidence'][0]['published_at'] = '2034-08-01'
        self.assertTrue(self.worker.candidates(case))  # Future to the old source, past to this edition.
        self.assertFalse(self.evaluate(case).decision['eligible'])

    def test_month_never_overrides_semantic_uncertainty_or_event_status(self):
        case = self.case('The panel will publish findings in January 2035.')
        for claim in ('UNCERTAIN', 'TARGET_STARTED', 'TARGET_COMPLETED', 'TARGET_CANCELLED'):
            with self.subTest(claim=claim):
                result = self.evaluate(case, lambda answer: answer['claims'][0].update(claim=claim))
                self.assertFalse(result.decision['eligible'])
        self.assertFalse(self.evaluate(case, lambda a: a['scheduled_change'].update(kind='UNCERTAIN')).decision['eligible'])
        self.assertFalse(self.evaluate(case, lambda a: a['story_connection'].update(kind='SHARED_SUBJECT_ONLY')).decision['eligible'])
        uncertain = self.case('The panel may publish findings in January 2035.')
        self.assertFalse(self.evaluate(uncertain, lambda a: a['claims'][0].update(claim='UNCERTAIN')).decision['eligible'])

    def test_overlapping_day_and_month_and_different_same_month_events_are_blocked(self):
        first = 'The panel will publish findings in January 2035.'
        for second in ('The panel will meet on January 12, 2035.',
                       'The panel will meet in January 2035.'):
            for sentences in ((first, second), (second, first)):
                with self.subTest(sentences=sentences):
                    result = self.evaluate(self.case(*sentences))
                    self.assertFalse(result.decision['eligible'])
                    self.assertIn('earliest event unresolved', result.decision['reason'])

    def test_separate_windows_keep_the_earliest_month_or_day(self):
        for first, second in (
                ('in January 2035', 'in February 2035'),
                ('in January 2035', 'on February 2, 2035'),
                ('on January 3, 2035', 'in February 2035')):
            earlier = f'The panel will publish findings {first}.'
            later = f'The panel will meet {second}.'
            for sentences in ((earlier, later), (later, earlier)):
                with self.subTest(sentences=sentences):
                    result = self.evaluate(self.case(*sentences))
                    self.assertTrue(result.decision['eligible'], result.decision)
                    self.assertEqual(earlier, result.decision['signal'])

    def test_exact_duplicate_month_sentence_can_use_both_sources(self):
        sentence = 'The panel will publish findings in January 2035.'
        result = self.evaluate(self.case(sentence, sentence))
        self.assertTrue(result.decision['eligible'])
        self.assertTrue(result.decision['all_candidates_reviewed'])

    def test_month_change_cannot_reuse_an_unresolved_window_receipt(self):
        case = self.case('The panel will publish findings in January 2035.')
        row = self.worker.candidates(case)[0]
        old = deepcopy(row)
        old['window'] = None
        old.pop('window_precision')
        cache = {}
        self.worker.remember(cache, case, old, receipt(self.worker, case, old))
        self.assertNotEqual(self.worker.cache_identity(case, old)[0], self.worker.cache_identity(case, row)[0])
        with self.assertRaises(checkpoint.CacheMiss):
            self.worker.evaluate(case['edition'], case['topic'], case['evidence'], cache=cache)

    def test_pipeline_month_output_keeps_original_precision_and_other_fields(self):
        sentence = ('With the panel publishing findings in January 2035, '
                    'officials will plan the next phase.')
        case = self.case(sentence)
        actual = deepcopy(case)
        actual['topic'].update(next_signal='Previous verified signal', untouched={'images': ['fixed'], 'counts': [1, 2]})
        with memory_run([case], self.cache(actual)) as (files, _, api, external):
            before = deepcopy(files['summary'])
            pipeline.main()
            before[0]['next_signal'] = sentence
            self.assertEqual(before, files['summary'])
            api.assert_not_called()
            external.assert_not_called()

    def test_month_chronology_extension_does_not_change_legacy_default(self):
        case = self.case('With the panel publishing findings in January 2035, officials will plan the next phase.')
        self.assertTrue(pipeline._unresolved_future_alternative(case['edition'], case['evidence'], 1))
        self.assertEqual('', pipeline._unresolved_future_alternative(
            case['edition'], case['evidence'], 1, allow_explicit_month=True))


class SameEventPrecisionChecks(unittest.TestCase):
    setUp = ExplicitMonthChecks.setUp
    case = ExplicitMonthChecks.case
    month = 'The panel will publish findings in January 2035.'
    day = 'The panel will publish findings on January 12, 2035.'

    def reviewed(self, case, relation=None, veto=None, unrelated=()):
        rows = self.worker.candidates(case)
        cache = {}
        for row in rows:
            answer = receipt(self.worker, case, row)
            if row['source_index'] in unrelated:
                answer['assessments'][0]['story_connection']['kind'] = 'SHARED_SUBJECT_ONLY'
            items = checkpoint._evidence(case, row)
            for claim in answer['assessments'][0]['claims']:
                item = items[claim['evidence_id'] - 1]
                other = next((r for r in rows if r['sentence'] == item['text']), None)
                same = other is not None and (relation is None or relation(row, other))
                claim['claim'] = 'TARGET_SCHEDULED' if same else 'OTHER_EVENT_OR_NO_STATUS'
                if veto and other is not None and other['source_index'] == veto[0]:
                    claim['claim'] = veto[1]
            self.worker.remember(cache, case, row, answer)
        self.last_cache = cache
        return self.worker.evaluate(case['edition'], case['topic'], case['evidence'], cache=cache)

    def test_matching_month_and_day_selects_the_verbatim_exact_date(self):
        for texts in ((self.month, self.day), (self.day, self.month)):
            with self.subTest(texts=texts):
                case = self.case(*texts)
                before = deepcopy(case)
                result = self.reviewed(case)
                self.assertTrue(result.decision['eligible'], result.decision)
                self.assertEqual(self.day, result.decision['signal'])
                self.assertEqual(2, len(result.decision['equivalent_candidate_ids']))
                self.assertTrue(result.decision['all_candidates_reviewed'])
                self.assertEqual(before, case)

    def test_undated_month_constrains_but_cannot_supply_display_copy(self):
        case = self.case(self.month, self.day)
        case['evidence'][0].update(published_at='', reference_date_unverified=True)
        rows = self.worker.candidates(case)
        self.assertEqual(('2035-01-01', '2035-01-31'), rows[0]['window'])
        self.assertFalse(rows[0]['publishable_source_copy'])
        result = self.reviewed(case)
        self.assertTrue(result.decision['eligible'], result.decision)
        self.assertEqual(self.day, result.decision['signal'])
        self.assertEqual(2, result.decision['source_index'])

    def test_present_tense_absolute_date_binds_without_publication_timestamp(self):
        case = self.case('The panel publishes findings on January 12, 2035.')
        case['evidence'][0].update(published_at='', reference_date_unverified=True)
        rows = self.worker.candidates(case)
        self.assertEqual(('2035-01-12', '2035-01-12'), rows[0]['window'])
        # Calendar resolution is not approval: an adverse review still vetoes.
        self.assertFalse(self.reviewed(case, veto=(1, 'TARGET_CANCELLED')).decision['eligible'])

    def test_one_way_identity_or_same_project_does_not_reconcile_dates(self):
        for relation in (lambda a, b: a['source_index'] == b['source_index'],
                         lambda a, b: a['source_index'] <= b['source_index']):
            result = self.reviewed(self.case(self.month, self.day), relation)
            self.assertFalse(result.decision['eligible'])
            self.assertIn('earliest event unresolved', result.decision['reason'])
        case = self.case(self.month, 'The panel will meet on January 12, 2035.')
        result = self.reviewed(case, lambda a, b: a['source_index'] == b['source_index'])
        self.assertFalse(result.decision['eligible'])

    def test_conflicting_dates_for_one_event_never_choose_the_earlier_claim(self):
        for texts in ((self.month, self.day.replace('January', 'February')),
                      (self.day, self.day.replace('12', '13'))):
            result = self.reviewed(self.case(*texts))
            self.assertFalse(result.decision['eligible'])
            self.assertEqual('Conflicting dates for the same event', result.decision['reason'])

    def test_inconsistent_identity_cannot_be_resolved_transitively(self):
        case = self.case(self.month, self.day, 'The panel publishes its report in January 2035.')
        result = self.reviewed(case, lambda a, b: abs(a['source_index'] - b['source_index']) <= 1)
        self.assertFalse(result.decision['eligible'])
        self.assertEqual('Inconsistent event identity across sources', result.decision['reason'])

    def test_cancellation_uncertainty_and_completion_override_matching_dates(self):
        for claim in ('TARGET_CANCELLED', 'UNCERTAIN', 'TARGET_COMPLETED', 'TARGET_STARTED'):
            with self.subTest(claim=claim):
                self.assertFalse(self.reviewed(self.case(self.month, self.day), veto=(1, claim)).decision['eligible'])

    def test_earlier_distinct_checkpoint_is_not_hidden_by_reconciled_later_event(self):
        earlier = 'The panel will meet on December 12, 2034.'
        case = self.case(self.month, self.day, earlier)
        result = self.reviewed(case, lambda a, b: (a['source_index'] == 3) == (b['source_index'] == 3))
        self.assertTrue(result.decision['eligible'], result.decision)
        self.assertEqual(earlier, result.decision['signal'])

    def test_equivalence_exempts_only_the_exact_bound_source_sentence(self):
        case = self.case(self.month, self.day)
        args = (case['edition'], case['evidence'], 2)
        window = ('2035-01-01', '2035-01-31')
        for proof in ({(2, self.month): window}, {(1, self.month + ' Changed.'): window},
                      {(1, self.month): ('2035-02-01', '2035-02-28')}):
            self.assertTrue(pipeline._unresolved_future_alternative(
                *args, allow_explicit_month=True, verified_equivalent_quotes=proof))
        self.assertEqual('', pipeline._unresolved_future_alternative(
            *args, allow_explicit_month=True, verified_equivalent_quotes={(1, self.month): window}))

    def test_equivalent_context_can_constrain_timing_without_authorizing_output(self):
        case = self.case(self.month, self.day)
        result = self.reviewed(case, unrelated=(1,))
        self.assertTrue(result.decision['eligible'], result.decision)
        self.assertEqual(self.day, result.decision['signal'])
        self.assertEqual(2, result.decision['source_index'])
        self.assertEqual(2, len(result.decision['equivalent_candidate_ids']))
        self.assertFalse(self.reviewed(case, unrelated=(1, 2)).decision['eligible'])

    def test_equivalent_context_cannot_hide_a_conflicting_date_or_uncertainty(self):
        case = self.case(self.month.replace('January', 'February'), self.day)
        result = self.reviewed(case, unrelated=(1,))
        self.assertFalse(result.decision['eligible'])
        self.assertEqual('Conflicting dates for the same event', result.decision['reason'])
        result = self.reviewed(self.case(self.month, self.day), unrelated=(1,), veto=(1, 'UNCERTAIN'))
        self.assertFalse(result.decision['eligible'])

    def test_daily_route_retains_verified_precision_proof_through_final_guard(self):
        case = self.case(self.month, self.day)
        actual = deepcopy(case)
        actual['topic'].update(next_signal='Previous verified signal',
                               untouched={'images': ['fixed'], 'counts': [1, 2]})
        self.reviewed(actual, unrelated=(1,))
        with memory_run([case], self.last_cache) as (files, _, api, external):
            before = deepcopy(files['summary'])
            pipeline.main()
            before[0]['next_signal'] = self.day
            self.assertEqual(before, files['summary'])
            api.assert_not_called()
            external.assert_not_called()


class MissingReferenceReconciliationChecks(unittest.TestCase):
    setUp = ExplicitMonthChecks.setUp
    case = ExplicitMonthChecks.case
    reviewed = SameEventPrecisionChecks.reviewed
    day = 'The panel will publish findings on October 18, 2034.'
    relative = 'The panel will publish findings next week.'

    def undated_case(self, undated=None, dated=None):
        case = self.case(undated or self.relative, dated or self.day)
        case['evidence'][0].update(published_at='', reference_date_unverified=True)
        return case

    def test_missing_reference_reconciles_only_after_complete_reciprocal_reviews(self):
        for reverse in (False, True):
            case = self.undated_case()
            if reverse:
                case['evidence'].reverse()
            before = deepcopy(case)
            rows_before = self.worker.candidates(case)
            result = self.reviewed(case)
            self.assertTrue(result.decision['eligible'], result.decision)
            self.assertEqual(self.day, result.decision['signal'])
            self.assertTrue(result.decision['all_candidates_reviewed'])
            self.assertEqual(2, len(result.decision['verified_equivalent_quotes']))
            self.assertEqual(before, case)
            self.assertEqual(rows_before, self.worker.candidates(case))
            self.assertIsNone(next(r['window'] for r in rows_before if r['sentence'] == self.relative))
            self.assertTrue(next(r for r in result.evidence if r['description'] == self.relative)['reference_date_unverified'])

    def test_partial_calendar_dates_must_match_the_independent_date(self):
        for timing in ('Wednesday', 'October 18', '10/18', 'October'):
            with self.subTest(timing=timing):
                case = self.undated_case(f'The panel will publish findings on {timing}.')
                self.assertTrue(self.reviewed(case).decision['eligible'])
        for timing in ('Thursday', 'October 19', '10/19', 'November', 'next Wednesday',
                       'Thursday, October 18', 'October 18 or 19', 'within 3 days'):
            with self.subTest(timing=timing):
                case = self.undated_case(f'The panel will publish findings on {timing}.')
                self.assertFalse(self.reviewed(case).decision['eligible'])

    def test_matching_ranges_reconcile_without_collapsing_to_a_single_day(self):
        day_range = 'The panel will publish findings on October 18–19.'
        for missing in (self.relative, day_range.replace('–', ' to ')):
            case = self.undated_case(missing, day_range)
            result = self.reviewed(case)
            self.assertTrue(result.decision['eligible'], result.decision)
            self.assertEqual(day_range, result.decision['signal'])
            self.assertEqual(['2034-10-18', '2034-10-19'], result.decision['verified_equivalent_quotes'][0]['window'])
        case = self.undated_case(day_range.replace('18–19', '18–20'), day_range)
        self.assertFalse(self.reviewed(case).decision['eligible'])

    def test_past_context_weekday_does_not_override_the_bound_future_reference(self):
        case = self.undated_case('Speaking on Tuesday, the panel said it will publish findings next week.')
        self.assertTrue(self.reviewed(case).decision['eligible'])

    def test_different_events_one_way_identity_and_transitive_identity_stay_blocked(self):
        for relation in (lambda a, b: a['source_index'] == b['source_index'],
                         lambda a, b: a['source_index'] <= b['source_index']):
            result = self.reviewed(self.undated_case(), relation)
            self.assertFalse(result.decision['eligible'])
        case = self.undated_case()
        case['evidence'].append(dict(case['evidence'][1], url='https://example.test/third',
            description='The panel releases its findings on October 18, 2034.',
            grounded_sentence='The panel releases its findings on October 18, 2034.'))
        self.assertFalse(self.reviewed(case, lambda a, b: abs(a['source_index'] - b['source_index']) <= 1).decision['eligible'])

    def test_status_uncertainty_and_conflicts_cannot_be_repaired_by_a_date(self):
        for claim in ('TARGET_CANCELLED', 'TARGET_COMPLETED', 'TARGET_STARTED', 'UNCERTAIN'):
            with self.subTest(claim=claim):
                self.assertFalse(self.reviewed(self.undated_case(), veto=(1, claim)).decision['eligible'])

    def test_unresolved_verified_source_and_full_year_conflict_stay_blocked(self):
        # The exception is for a missing reference, never any arbitrary parse failure.
        case = self.undated_case('The panel will publish findings on Thursday, October 18, 2034.')
        for unverified in (True, False):
            case['evidence'][0]['reference_date_unverified'] = unverified
            self.assertFalse(self.reviewed(case).decision['eligible'])

    def test_undated_evidence_cannot_authorize_output_on_its_own(self):
        case = self.undated_case()
        case['evidence'] = case['evidence'][:1]
        self.assertFalse(self.reviewed(case).decision['eligible'])
        case = self.undated_case()
        self.assertFalse(self.reviewed(case, unrelated=(1, 2)).decision['eligible'])
        self.assertTrue(self.reviewed(case, unrelated=(1,)).decision['eligible'])

    def test_no_approval_with_missing_dated_review_and_no_cache_mutation(self):
        case = self.undated_case()
        self.reviewed(case)
        cache = deepcopy(self.last_cache)
        dated = next(r for r in self.worker.candidates(case) if r['window'])
        del cache[self.worker.cache_identity(case, dated)[0]]
        before = deepcopy(cache)
        with self.assertRaises(checkpoint.CacheMiss):
            self.worker.evaluate(case['edition'], case['topic'], case['evidence'], cache=cache)
        self.assertEqual(before, cache)

    def test_daily_route_retains_reconciliation_and_changes_only_next_signal(self):
        case = self.undated_case()
        actual = deepcopy(case)
        actual['topic'].update(next_signal='Previous verified signal',
                               untouched={'images': ['fixed'], 'counts': [1, 2]})
        self.reviewed(actual)
        with memory_run([case], self.last_cache) as (files, _, api, external):
            before = deepcopy(files['summary'])
            pipeline.main()
            before[0]['next_signal'] = self.day
            self.assertEqual(before, files['summary'])
            api.assert_not_called()
            external.assert_not_called()


if __name__ == '__main__':
    unittest.main()
