import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_next_signals as review_signals
import run_pipeline_by_date as daily_pipeline
from copy import deepcopy
from datetime import date, timedelta


def load_fixture(name):
    """Fixed offline examples; never read mutable daily outputs or caches."""
    fixture = Path(__file__).parent / "fixtures" / "next_signal_cases.json"
    return json.loads(fixture.read_text(encoding="utf-8"))[Path(name).name]


class Choice(dict):
    @property
    def message(self):
        return self["message"]


def response(payload, finish_reason="stop"):
    from types import SimpleNamespace
    # Existing tests isolate ordering/grounding from the new current-state gate.
    if "eligible" in payload:
        payload = {"checkpoint_state": "FUTURE", "state_evidence": "A pending checkpoint is stated.", **payload}
    return SimpleNamespace(choices=[Choice(message={"content": json.dumps(payload)}, finish_reason=finish_reason)])


class CalendarDateAppositionChecks(unittest.TestCase):
    def resolve(self, edition, sentence, timing, published=None):
        return review_signals._comparison_calendar_day(
            edition, {'description': sentence, 'published_at': published or edition}, timing)

    def test_matching_weekday_date_pairs_across_years_and_leap_day(self):
        for start in (date(2027, 6, 1), date(2032, 2, 26), date(2035, 12, 28)):
            for offset in range(7):
                target = start + timedelta(days=offset)
                edition = (target - timedelta(days=1)).isoformat()
                weekday = target.strftime('%A')
                for timing in (target.strftime('%B %d, %Y'), target.strftime('%d %B %Y'), target.isoformat()):
                    for pair in (f'{weekday}, {timing}', f'{weekday} {timing}', f'{timing}, {weekday}'):
                        with self.subTest(pair=pair):
                            self.assertEqual(target, self.resolve(edition, f'The panel will meet on {pair}.', timing))

    def test_matching_yearless_pair_uses_existing_verified_reference(self):
        for timing in ('November 3', '3 November'):
            self.assertEqual(date(2026, 11, 3), self.resolve(
                '2026-11-02', f'The panel will meet on Tuesday, {timing}.', timing))
            self.assertIsNone(self.resolve(
                '2026-11-02', f'The panel will meet on Tuesday, {timing}.', timing, '2025-11-02'))

    def test_conflicting_weekdays_are_not_ignored(self):
        for start in (date(2027, 6, 1), date(2032, 2, 26)):
            for offset in range(7):
                target = start + timedelta(days=offset)
                timing = target.strftime('%B %d, %Y')
                wrong = (target + timedelta(days=1)).strftime('%A')
                with self.subTest(target=target):
                    self.assertIsNone(self.resolve((target-timedelta(days=1)).isoformat(),
                        f'The panel will meet on {wrong}, {timing}.', timing))

    def test_missing_historical_and_same_day_anchors_stay_blocked(self):
        for text, timing in (
            ('The panel will meet on Monday, November 2, 2026.', 'November 2, 2026'),
            ('The panel met on Sunday, November 1, 2026.', 'November 1, 2026'),
            ('The panel met Sunday, November 1, and will return Tuesday, November 3.', 'November 3'),
            ('The panel will meet Tuesday or November 3.', 'November 3'),
            ('The panel will meet Tuesday and November 3.', 'November 3'),
            ('The panel will meet Tuesday, November 3-5.', 'November 3'),
            ('The panel will meet Tuesday, November 3, or Wednesday, November 4.', 'November 3'),
            ('The panel will meet next week.', 'next week'),
        ):
            with self.subTest(text=text):
                self.assertIsNone(self.resolve('2026-11-02', text, timing))

    def test_matching_apposition_does_not_shrink_qualified_timing(self):
        for qualifier in ('by', 'before', 'after', 'since', 'within', 'through', 'from',
                          'between', 'around', 'about', 'until', 'last', 'previous', 'past', 'next'):
            for timing in ('November 3', 'November 3, 2026'):
                with self.subTest(qualifier=qualifier, timing=timing):
                    self.assertIsNone(self.resolve('2026-11-02',
                        f'The panel will meet {qualifier} Tuesday, {timing}.', timing))


class FuturePassiveDiscoveryChecks(unittest.TestCase):
    def test_future_passive_keeps_its_own_weekday_across_dates(self):
        for edition in (date(2026,11,5), date(2030,5,9), date(2032,2,26)):
            target=edition+timedelta(days=3)
            for verb in ('approved','rejected','signed','opened','sentenced'):
                for frame in ('will be','will probably be','is due to be','is expected to be formally'):
                    sentence=f'The application {frame} {verb} on {target:%A}.'
                    with self.subTest(sentence=sentence,edition=edition):
                        anchors=review_signals._direct_future_anchors(sentence,edition)
                        self.assertEqual([f'weekday:{target:%A}'.lower()],[a['key'] for a in anchors])

    def test_completed_negative_and_other_clause_dates_stay_out(self):
        for sentence in (
            'The permit was approved on Monday.',
            'The permit had been approved on Monday.',
            'The permit will not be approved on Monday.',
            'The permit was expected to be approved on Monday.',
            'The mayor will speak after the council approved the permit Monday.',
            'The mayor will announce the permit approved on Monday.',
            'The office will be ready; the council signed the permit Monday.',
            'The mayor will be present when the council announced its decision Monday.',
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual([],review_signals._direct_future_anchors(sentence,date(2026,11,5)))


class FixedProposalReviewChecks(unittest.TestCase):
    """Offline regression checks for the active Next Signal implementation."""

    @staticmethod
    def comparison(index=1, timing="Monday", qualifies=True, before=False, **extra):
        return {"source_index": index, "event": "Panel findings", "timing": timing,
                "qualifies": qualifies, "before_selected": before,
                "reason": "Explicit future checkpoint.", **extra}

    @staticmethod
    def writer_assessments(comparisons):
        return [{key: value for key, value in item.items() if key != "before_selected"}
                for item in comparisons]

    def setUp(self):
        self.day = "2026-09-25"
        self.row = {
            "title": "Panel findings", "published_at": self.day,
            "description": "The panel may release its findings Monday.",
            "grounded_sentence": "The panel may release its findings Monday.",
            "grounded_time_keys": ["date:2026-09-28"],
        }
        self.topic = {"topic_title": "Panel findings", "summary": "The panel is evaluating materials."}
        self.proposal = {
            "status": "SIGNAL", "eligible": True, "source_index": 1,
            "signal": self.row["description"], "reason": "WRITER_RATIONALE_SENTINEL",
            "comparisons": [self.comparison(reason="WRITER_COMPARISON_SENTINEL")],
        }
        self.approval = {"eligible": True, "reason": "Supported.", "comparisons": [self.comparison()]}
        # Every request must be supplied explicitly by the test, including failures.
        self.api = self.enterContext(patch.object(
            review_signals.openai.ChatCompletion, "create",
            side_effect=AssertionError("Unexpected model request"),
        ))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")))

    def review(self, proposal=None, rows=None):
        return review_signals.validate_candidate_signal_semantics(
            self.day, self.topic, rows or [self.row], proposal or self.proposal,
        )

    def test_reviewer_model_has_separate_cache_without_invalidating_writer(self):
        keys = {stage: review_signals._cache_key(self.day, self.topic, [self.row], stage=stage)
                for stage in ("triage", "final", "candidate_validation")}
        with patch.object(review_signals, "REVIEW_MODEL", "different-reviewer"):
            for stage in ("triage", "final"):
                self.assertEqual(keys[stage], review_signals._cache_key(self.day, self.topic, [self.row], stage=stage))
            self.assertNotEqual(keys["candidate_validation"], review_signals._cache_key(
                self.day, self.topic, [self.row], stage="candidate_validation"))
        self.api.side_effect = None
        self.api.return_value = response(self.approval)
        self.review()
        request = self.api.call_args.kwargs
        self.assertEqual("gpt-5.4-mini-2026-03-17", request["model"])
        self.assertEqual("none", request["reasoning_effort"])
        self.assertNotIn("max_tokens", request)
        self.api.assert_called_once()

    def test_current_state_overrides_an_inconsistent_approval(self):
        self.api.side_effect = None
        for state in ('ALREADY_OCCURRED', 'SUPERSEDED', 'UNRESOLVED'):
            with self.subTest(state=state):
                self.api.return_value = response({**self.approval, 'checkpoint_state': state,
                    'state_evidence': 'The current story overtakes the earlier schedule.'})
                result = self.review()
                self.assertFalse(result['eligible'])
                self.assertEqual('checkpoint_not_future', result['reason_code'])
        self.api.return_value = response({**self.approval, 'checkpoint_state': None})
        with self.assertRaisesRegex(RuntimeError, 'current state'):
            self.review()

    def test_review_calendar_relations_cross_month_year_and_leap_day(self):
        for edition, target, weekday in (
            ('2026-11-02', '2026-11-03', 'Tuesday'),
            ('2031-12-31', '2032-01-01', 'Thursday'),
            ('2032-02-28', '2032-02-29', 'Sunday'),
            ('2040-08-14', '2040-08-15', 'Wednesday'),
        ):
            with self.subTest(edition=edition):
                sentence = f'The regional panel will release its findings {weekday}.'
                row = {**self.row, 'published_at': edition, 'grounded_sentence': sentence}
                prompt = review_signals._checkpoint_timing_comparison(
                    edition, [row], include_day_offsets=True)
                self.assertIn(f'{target} if upcoming ({weekday}; 1 calendar day(s) AFTER briefing)', prompt)
                # Adding reviewer context must not change the writer's input.
                self.assertNotIn('AFTER briefing', review_signals._checkpoint_timing_comparison(edition, [row]))

    def test_calendar_context_precedes_state_judgment_in_existing_review_call(self):
        self.api.side_effect = None
        self.api.return_value = response(self.approval)
        self.review()
        prompt = self.api.call_args.kwargs['messages'][1]['content']
        self.assertIn('BRIEFING DATE: 2026-09-25 (Friday)', prompt)
        self.assertIn('2026-09-28 if upcoming (Monday; 3 calendar day(s) AFTER briefing)', prompt)
        self.assertLess(prompt.index('CALENDAR CONTEXT'), prompt.index('Then establish the checkpoint'))
        self.assertIn('Present-tense news headlines', prompt)
        self.assertIn('quote explicit completion/underway evidence', prompt)
        schema = self.api.call_args.kwargs['response_format']['json_schema']['schema']
        self.assertEqual('calendar_check', next(iter(schema['properties'])))
        self.assertEqual(['2026-09-28 is 3 calendar day(s) AFTER briefing 2026-09-25'],
                         schema['properties']['calendar_check']['enum'])
        self.assertIn('Copy the words, not a paraphrase or inferred date',
                      schema['properties']['state_evidence']['description'])
        self.api.assert_called_once()

    def test_review_calendar_does_not_reanchor_old_or_undated_sources(self):
        sentence = 'The panel will release its findings Tuesday.'
        for changes in ({'published_at': '2026-10-26'},
                        {'published_at': '', 'reference_date_unverified': True}):
            with self.subTest(changes=changes):
                row = {**self.row, 'grounded_sentence': sentence, **changes}
                prompt = review_signals._checkpoint_timing_comparison(
                    '2026-11-02', [row], include_day_offsets=True)
                self.assertNotIn('2026-11-03 if upcoming', prompt)
                self.assertNotIn('AFTER briefing', prompt)
                self.assertIn('unresolved timing', prompt)

    def test_review_calendar_preserves_same_day_uncertainty(self):
        row = {**self.row, 'published_at': '2026-11-02',
               'grounded_sentence': 'The panel will release its findings Monday afternoon.'}
        prompt = review_signals._checkpoint_timing_comparison(
            '2026-11-02', [row], include_day_offsets=True)
        self.assertIn('2026-11-02 if upcoming (Monday; SAME briefing day; time-of-day status unverified)', prompt)
        self.assertNotIn('AFTER briefing', prompt)

    def test_fixed_calendar_fact_cannot_resolve_ambiguous_or_unverified_schedules(self):
        day = '2026-11-02'
        for sentence, changes in (
            ('The panel will release findings Monday afternoon.', {}),
            ('The panel will release findings next week.', {}),
            ('The panel will release findings Tuesday through Friday.', {}),
            ('The panel will release findings Tuesday.', {'reference_date_unverified': True}),
            ('The panel will release findings Tuesday.', {'published_at': '2026-10-26'}),
        ):
            with self.subTest(sentence=sentence, changes=changes):
                row = {**self.row, 'published_at': day, 'grounded_sentence': sentence, **changes}
                self.assertIn('unresolved', review_signals._review_calendar_check(day, row))
        row = {**self.row, 'published_at': day,
               'grounded_sentence': 'The panel will release findings Tuesday.'}
        self.assertEqual('2026-11-03 is 1 calendar day(s) AFTER briefing 2026-11-02',
                         review_signals._review_calendar_check(day, row))

    def test_following_correction_survives_both_excerpt_reductions(self):
        text = ('The regional panel met to discuss the application. '
                'The panel will release its findings Monday. '
                'Update: The panel cancelled the release. '
                'The chair said no replacement date has been set.')
        snippets = review_signals._future_milestone_snippets(text, date.fromisoformat(self.day))
        hit = next(x for x in snippets if self.row['grounded_sentence'] in x['text'] or 'will release' in x['text'])
        row = {**self.row, 'description': hit['text'], 'source_context': hit['source_context']}
        evidence = review_signals._model_evidence_candidates(self.day, [row])
        self.assertIn('cancelled the release', evidence[0]['source_context'])
        self.assertIn('no replacement date', review_signals._evidence_lines(evidence)[0])
        changed = [{**evidence[0], 'source_context': 'A revised source update.'}]
        self.assertNotEqual(review_signals._cache_key(self.day, self.topic, evidence),
                            review_signals._cache_key(self.day, self.topic, changed))

    def test_discovery_day_cannot_anchor_an_article_weekday(self):
        candidate = review_signals._candidate_from_row({
            'url': 'https://example.org/story', 'title': 'Panel release', 'target_date': self.day,
        }, 'coverage_receipt')
        self.assertEqual('', candidate['published_at'])
        self.assertTrue(candidate['reference_date_unverified'])
        row = {**self.row, **candidate, 'grounded_sentence': self.row['description']}
        self.assertFalse(review_signals.validate_generated_signal_grounding(
            self.day, row['grounded_sentence'], row)[0])
        # Actual source publication metadata remains usable.
        candidate = review_signals._candidate_from_row({
            'url': 'https://example.org/story', 'title': 'Panel release', 'published_at': self.day,
        }, 'existing_story_source')
        row.update(candidate)
        self.assertTrue(review_signals.validate_generated_signal_grounding(
            self.day, row['grounded_sentence'], row)[0])

    def test_saved_oct27_false_approvals_stay_withheld_after_context_review(self):
        sources = load_fixture("oct27_sentence_selection")
        reviews = load_fixture("oct27_sentence_selection_reviews")
        self.api.side_effect = None
        for saved in reviews:
            case = sources[saved["topic_index"] - 1]
            with self.subTest(story=saved["title"]):
                self.api.reset_mock()
                self.api.return_value = response(saved["review"])
                result = review_signals.review_signal_for_publication(
                    case["date"], case["topic"], case["evidence"], saved["proposal"])
                expected = saved["topic_index"] == 2
                self.assertEqual(expected, result["eligible"])
                if expected:
                    self.api.assert_called_once()
                    self.assertEqual(saved["proposal"]["signal"], result["signal"])
                else:
                    # Context can reach the existing reviewer, but a legacy
                    # approval without supported obsolete-event status cannot
                    # bypass the conservative ordering check.
                    self.assertLessEqual(self.api.call_count, 1)
                    self.assertIn(result["reason_code"], {"unresolved_timing", "review_rejection"})
        # Isolate the genuine earlier results checkpoint, without the other
        # source's past victory declaration. First-person copy is still useful
        # evidence against choosing the later election.
        election = sources[0]
        rows = election["evidence"][1:3]
        proposal = {"eligible": True, "source_index": 2, "signal": rows[1]["grounded_sentence"]}
        self.api.reset_mock()
        result = review_signals.review_signal_for_publication(election["date"], election["topic"], rows, proposal)
        self.assertFalse(result["eligible"])
        self.assertIn("Source 1", result["reason"])
        self.api.assert_not_called()

    def test_future_action_cannot_borrow_a_date_from_another_clause(self):
        for day in ("2026-10-27", "2031-03-04", "2040-08-14"):
            for text in (
                "The company will issue refunds after a payment failed over the weekend.",
                "The team will release a report after a workshop ended Friday.",
                "The ship will depart when the inspection that began Monday is complete.",
                "The panel will meet while the inquiry that opened Tuesday continues.",
            ):
                with self.subTest(day=day, text=text):
                    row = {"description": text, "grounded_sentence": text, "published_at": day}
                    proposal = {"eligible": True, "source_index": 1, "signal": text}
                    result = review_signals.review_signal_for_publication(day, self.topic, [row], proposal)
                    self.assertFalse(result["eligible"])
                    self.assertEqual("unresolved_timing", result["reason_code"])
        self.api.assert_not_called()

    def test_simple_when_clause_reaches_review_with_source_wording_intact(self):
        for day in ("2026-11-02", "2031-03-04", "2040-08-14"):
            for text in (
                "The committee will consider disciplinary steps when the assembly reconvenes next week.",
                "The operator will restore service when the terminal reopens next month.",
                "The board will discuss the findings when delegates meet next year.",
            ):
                with self.subTest(day=day, text=text):
                    row = {"description": text, "grounded_sentence": text, "published_at": day}
                    proposal = {"eligible": True, "source_index": 1, "signal": text}
                    rejection = {"eligible": False, "reason": "Independent review remains required."}
                    with patch.object(review_signals, "validate_candidate_signal_semantics", return_value=rejection) as reviewer:
                        result = review_signals.review_signal_for_publication(day, self.topic, [row], proposal)
                        self.assertEqual(rejection, result)
                        reviewer.assert_called_once_with(day, self.topic, [row], proposal)
                    self.assertTrue(review_signals.validate_generated_signal_grounding(day, text, row)[0])
        self.api.assert_not_called()

    def test_when_clause_cannot_borrow_embedded_background_dates(self):
        for text in (
            "The panel will act when the hearing that began Monday concludes.",
            "The board will vote when the report about Friday arrives.",
            "The board will vote when the report for next week arrives.",
            "The board will vote when delegates discuss Friday's incident.",
            "The board will vote when inspectors who arrived Monday finish.",
            "The board will vote when the office was reopened Monday.",
            "The board will vote after delegates meet next week.",
            "The board will vote because delegates meet next week.",
        ):
            with self.subTest(text=text):
                self.assertEqual([], review_signals._direct_future_anchors(text, date(2026, 11, 2)))

    def test_contextual_expiry_blocks_later_signal_without_becoming_copyable(self):
        self.api.side_effect = None
        for day in ("2026-10-19", "2027-03-01", "2031-09-01"):
            early = "Both would remain in effect until 6 am on Tuesday morning."
            later = "The council will discuss the restrictions Saturday."
            rows = [{"grounded_sentence": early, "published_at": day,
                     "source_context": "Police imposed a curfew and protest restrictions. " + early},
                    {"grounded_sentence": later, "published_at": day}]
            with self.subTest(day=day):
                self.api.reset_mock()
                self.api.return_value = response({"eligible": True, "reason": "Proposed approval.", "comparisons": [
                    self.comparison(1, "Tuesday", qualifies=False, before=False, checkpoint_state="UNRESOLVED",
                                    state_evidence="", event="Restrictions expire", reason="Status unclear."),
                    self.comparison(2, "Saturday", event="Council meeting"),
                ]})
                result = review_signals.review_signal_for_publication(day, self.topic, rows,
                    {"eligible": True, "source_index": 2, "signal": later})
                self.assertFalse(result["eligible"])
                self.assertEqual("unresolved_timing", result["reason_code"])
                self.assertIn("Source 1", result["reason"])
                self.assertFalse(review_signals.validate_signal_source_copy(early, rows[0])[0])
                self.api.assert_called_once()

    def test_metadata_selection_recovers_late_source_without_changing_cluster(self):
        for day in ("2026-11-02", "2031-03-04", "2040-08-14"):
            articles = [{"title": "Committee investigation", "description": "The committee is reviewing the evidence.",
                         "published_at": day, "url": f"https://example.com/{i}"} for i in range(14)]
            articles[12]["description"] = "The committee will consider next steps when the assembly reconvenes next week."
            cluster = {"articles": articles}
            before = deepcopy(cluster)
            selected = review_signals.select_signal_evidence(cluster, day)
            with self.subTest(day=day):
                self.assertEqual(before, cluster)
                self.assertEqual(12, len(selected))
                self.assertEqual([x["url"] for x in articles[:6]], [x["url"] for x in selected[:6]])
                self.assertIn(articles[12]["url"], [x["url"] for x in selected])
                self.assertLessEqual(len(review_signals._model_evidence_candidates(day, selected)), 8)
                self.assertEqual([x["url"] for x in articles[:12]],
                                 [x["url"] for x in review_signals.select_signal_evidence(cluster)])
        self.api.assert_not_called()

    def obsolete_review_case(self, state="ALREADY_OCCURRED"):
        quote = ("The regional panel completed its preliminary hearing yesterday." if state == "ALREADY_OCCURRED"
                 else "The regional panel cancelled its preliminary hearing.")
        early = {"title": "Panel inquiry", "published_at": "", "reference_date_unverified": True,
                 "description": "The regional panel will hold a preliminary hearing Monday.",
                 "grounded_sentence": "The regional panel will hold a preliminary hearing Monday.",
                 "source_overview": quote, "source_context": "The inquiry continues."}
        later = {**self.row, "description": "The panel will release final findings Wednesday.",
                 "grounded_sentence": "The panel will release final findings Wednesday."}
        proposal = {**self.proposal, "source_index": 2, "signal": later["grounded_sentence"]}
        reply = {"eligible": True, "checkpoint_state": "FUTURE", "state_evidence": later["grounded_sentence"],
                 "reason": "The initial hearing is obsolete; the later findings remain pending.", "comparisons": [
                     self.comparison(1, "Monday", qualifies=False, before=False, event=early["grounded_sentence"],
                                     checkpoint_state=state, state_evidence=quote,
                                     reason="The same panel's preliminary hearing is explicitly completed or cancelled."),
                     self.comparison(2, "Wednesday", event="Final findings"),
                 ]}
        return [early, later], proposal, reply

    def test_supported_obsolete_alternative_uses_existing_review_once(self):
        self.api.side_effect = None
        for state in ("ALREADY_OCCURRED", "SUPERSEDED"):
            with self.subTest(state=state):
                self.api.reset_mock()
                rows, proposal, reply = self.obsolete_review_case(state)
                before = deepcopy(rows)
                self.api.return_value = response(reply)
                result = review_signals.review_signal_for_publication(self.day, self.topic, rows, proposal)
                self.assertTrue(result["eligible"])
                self.assertEqual(proposal["signal"], result["signal"])
                self.assertEqual(before, rows)
                self.api.assert_called_once()
                prompt = self.api.call_args.kwargs["messages"][1]["content"]
                self.assertIn(rows[0]["source_overview"], prompt)
                self.assertEqual("", review_signals._unresolved_future_alternative(
                    self.day, rows, 2, result["comparisons"]))
                # Without reviewer evidence, the publication guard still blocks.
                self.assertIn("Source 1", review_signals._unresolved_future_alternative(self.day, rows, 2))

    def test_uncertain_or_unsupported_obsolete_assessments_stay_blocked(self):
        self.api.side_effect = None
        rows, proposal, original = self.obsolete_review_case()
        for changes in (
            {"checkpoint_state": "UNRESOLVED"}, {"checkpoint_state": "FUTURE"},
            {"state_evidence": ""}, {"state_evidence": "The regional panel cancelled the hearing yesterday."},
            {"state_evidence": "completed its preliminary hearing yesterday."},
            {"state_evidence": "The regional panel completed ... yesterday."},
            {"state_evidence": rows[0]["grounded_sentence"]},  # the old preview itself
            {"state_evidence": rows[1]["grounded_sentence"]},  # another source
            {"qualifies": True}, {"before_selected": None}, {"before_selected": True},
        ):
            with self.subTest(changes=changes):
                self.api.reset_mock()
                reply = deepcopy(original);reply["comparisons"][0].update(changes)
                self.api.return_value = response(reply)
                result = review_signals.review_signal_for_publication(self.day, self.topic, rows, proposal)
                self.assertFalse(result["eligible"])
                self.assertEqual("", result["signal"])
                expected_code = "calendar_conflict" if changes.get("qualifies") is True else "unresolved_timing"
                self.assertEqual(expected_code, result["reason_code"])
                self.api.assert_called_once()

    def test_missing_status_and_removed_context_cannot_clear_alternative(self):
        self.api.side_effect = None
        rows, proposal, reply = self.obsolete_review_case()
        for key in ("state_evidence", "checkpoint_state"):
            broken = deepcopy(reply);broken["comparisons"][0].pop(key)
            self.api.return_value = response(broken)
            self.assertFalse(review_signals.review_signal_for_publication(self.day, self.topic, rows, proposal)["eligible"])
        # A quote can no longer justify exclusion if it disappeared from source context.
        changed = deepcopy(rows);changed[0]["source_overview"] = "The panel has not announced a hearing outcome."
        self.api.return_value = response(reply)
        self.assertFalse(review_signals.review_signal_for_publication(self.day, self.topic, changed, proposal)["eligible"])

    def test_obsolete_proof_must_assess_the_candidate_event_not_another_action(self):
        self.api.side_effect = None
        rows, proposal, original = self.obsolete_review_case()
        rows[0]["title"] = "The regional panel announced its new chair Monday."
        rows[0]["source_context"] += " The regional panel announced its new chair Monday."
        for event in (
            "", "Preliminary hearing", "Monday",
            rows[0]["title"],  # Same source/person/day, different action.
            rows[1]["grounded_sentence"],  # Another source's action.
            "The regional panel will hold a preliminary hearing",  # No quoted timing.
            "The regional panel will hold ... hearing Monday.",
            "panel will hold a preliminary hearing Mon",  # Partial timing word.
        ):
            with self.subTest(event=event):
                reply = deepcopy(original)
                reply["comparisons"][0]["event"] = event
                self.api.return_value = response(reply)
                result = review_signals.review_signal_for_publication(self.day, self.topic, rows, proposal)
                self.assertFalse(result["eligible"])
                self.assertEqual("unresolved_timing" if event else "comparison_incomplete", result["reason_code"])

    def test_exact_event_clause_can_be_quoted_from_a_longer_candidate(self):
        rows, _, reply = self.obsolete_review_case()
        row = rows[0]
        row["grounded_sentence"] = "Background explains the inquiry. " + row["grounded_sentence"]
        self.assertTrue(review_signals._obsolete_alternative(row, 1, reply["comparisons"]))
        # Quotation matching cannot certify a model's interpretation of the quote.
        reply["comparisons"][0]["checkpoint_state"] = "UNRESOLVED"
        self.assertFalse(review_signals._obsolete_alternative(row, 1, reply["comparisons"]))

    def test_review_schema_requires_own_source_proof_for_an_obsolete_state(self):
        rows, _, reply = self.obsolete_review_case()
        foreign = "The other commission cancelled its investigation."
        rows[1]["source_overview"] = foreign
        options = review_signals._comparison_request_options("candidate_validation", 2, 2, evidence=rows)
        schema = options["response_format"]["json_schema"]["schema"]
        variants = schema["properties"]["comparisons"]["items"]["anyOf"]
        alternatives = [v["properties"] for v in variants if v["properties"]["source_index"]["enum"] == [1]]
        self.assertEqual(2, len(alternatives))
        for item in alternatives:
            self.assertLess(list(item).index("event"), list(item).index("state_evidence_index"))
            self.assertLess(list(item).index("checkpoint_state"), list(item).index("state_evidence_index"))
            states = item["checkpoint_state"]["enum"]
            indices = item["state_evidence_index"].get("enum", [])
            quotes = [review_signals._obsolete_state_quotes(rows[0])[i - 1] for i in indices]
            if "ALREADY_OCCURRED" in states:
                self.assertCountEqual([rows[0]["source_overview"], "The inquiry continues."], quotes)
                self.assertNotIn("", quotes)
                self.assertNotIn(foreign, quotes)
                self.assertNotIn(rows[0]["grounded_sentence"], quotes)
                self.assertEqual([False], item["qualifies"]["enum"])
                self.assertEqual([False], item["before_selected"]["enum"])
            else:
                self.assertEqual(["FUTURE", "UNRESOLVED"], states)
                self.assertEqual([], quotes)
        self.api.assert_not_called()

    def test_no_status_quotes_means_no_obsolete_response_option(self):
        rows, _, _ = self.obsolete_review_case()
        rows[0].pop("source_overview")
        rows[0].pop("source_context")
        options = review_signals._comparison_request_options("candidate_validation", 2, 2, evidence=rows)
        variants = options["response_format"]["json_schema"]["schema"]["properties"]["comparisons"]["items"]["anyOf"]
        alternatives = [v["properties"] for v in variants if v["properties"]["source_index"]["enum"] == [1]]
        self.assertEqual(1, len(alternatives))
        self.assertEqual(["FUTURE", "UNRESOLVED"], alternatives[0]["checkpoint_state"]["enum"])
        with self.assertRaisesRegex(ValueError, "evidence count"):
            review_signals._comparison_request_options("candidate_validation", 2, 2, evidence=rows[:1])

    def test_obsolete_contract_reaches_only_the_existing_review_request(self):
        rows, proposal, reply = self.obsolete_review_case()
        self.api.side_effect = None
        self.api.return_value = response(reply)
        self.assertTrue(review_signals.review_signal_for_publication(self.day, self.topic, rows, proposal)["eligible"])
        request = self.api.call_args.kwargs
        self.assertIn("Assess THIS quoted event, not the headline", request["messages"][1]["content"])
        self.assertIn("Cancellation or nonattendance is SUPERSEDED", request["messages"][1]["content"])
        variants = request["response_format"]["json_schema"]["schema"]["properties"]["comparisons"]["items"]["anyOf"]
        self.assertIn(rows[0]["source_overview"], request["messages"][1]["content"])
        self.assertTrue(any(v["properties"].get("state_evidence_index", {}).get("enum") for v in variants))
        self.assertEqual(review_signals.REVIEW_MODEL if "max_completion_tokens" in request else review_signals.MODEL, request["model"])
        self.assertEqual(680, request.get("max_tokens", request.get("max_completion_tokens")))
        self.api.assert_called_once()

    def test_status_evidence_ids_preserve_quotes_and_reject_invalid_indices(self):
        rows, proposal, reply = self.obsolete_review_case()
        quote = 'The panel completed the "preliminary hearing" yesterday.'
        rows[0]["source_overview"] = quote
        choices = review_signals._obsolete_state_quotes(rows[0])
        index = choices.index(quote) + 1
        item = reply["comparisons"][0]
        item.pop("state_evidence")
        item["state_evidence_index"] = index
        self.api.side_effect = None
        self.api.return_value = response(reply)
        result = review_signals.review_signal_for_publication(self.day, self.topic, rows, proposal)
        self.assertTrue(result["eligible"])
        self.assertEqual(quote, result["comparisons"][0]["state_evidence"])
        schema = self.api.call_args.kwargs["response_format"]["json_schema"]["schema"]
        self.assertNotIn(quote, json.dumps(schema))
        for invalid in (True, 0, -1, len(choices) + 1, "1"):
            item["state_evidence_index"] = invalid
            self.api.return_value = response(reply)
            with self.assertRaisesRegex(RuntimeError, "status evidence index"):
                self.review(proposal, rows)
        item["state_evidence_index"] = None
        self.api.return_value = response(reply)
        self.assertFalse(review_signals.review_signal_for_publication(self.day, self.topic, rows, proposal)["eligible"])

    def test_obsolete_quote_cannot_drop_negation_or_join_separate_sentences(self):
        row = {"source_context": "The panel did not cancel the hearing. A decision remains pending."}
        for quote in ("cancel the hearing.", "The panel cancelled the hearing.", row["source_context"]):
            comparisons = [self.comparison(qualifies=False, checkpoint_state="SUPERSEDED", state_evidence=quote)]
            self.assertFalse(review_signals._obsolete_alternative(row, 1, comparisons))

    def test_opening_context_is_bounded_preserved_and_review_only(self):
        opening = "The panel completed its preliminary hearing yesterday."
        text = opening + " " + "Background discussion of the investigation continues. " * 50
        text += "The panel will hold a preliminary hearing Monday."
        snippets = review_signals._future_milestone_snippets(text, date.fromisoformat(self.day))
        self.assertTrue(snippets)
        self.assertIn(opening, snippets[0]["source_overview"])
        self.assertLessEqual(len(snippets[0]["source_overview"]), 2000)
        self.assertTrue(snippets[0]["source_overview"].endswith("."))
        row = {**self.row, "source_overview": snippets[0]["source_overview"]}
        self.assertNotIn(opening, "\n".join(review_signals._evidence_lines([row])))
        self.assertIn(opening, "\n".join(review_signals._evidence_lines([row], include_overview=True)))
        updated = {**row, "source_overview": "Updated context."}
        self.assertNotEqual(review_signals._cache_key(self.day, self.topic, [row], stage="candidate_validation"),
                            review_signals._cache_key(self.day, self.topic, [updated], stage="candidate_validation"))
        # This extra context is review-only; unchanged writer inputs keep their cache.
        self.assertEqual(review_signals._cache_key(self.day, self.topic, [row], stage="triage"),
                         review_signals._cache_key(self.day, self.topic, [updated], stage="triage"))

    def test_cached_obsolete_proof_is_rechecked_at_publication(self):
        rows, proposal, reply = self.obsolete_review_case()
        self.api.side_effect = None
        self.api.return_value = response(reply)
        with self.pipeline() as (files, _, triage), patch.object(
            review_signals, "_model_evidence_candidates", return_value=rows,
        ):
            triage.return_value = proposal
            review_signals.main()
            self.assertEqual(proposal["signal"], files["summary"][0]["next_signal"])
            self.api.assert_called_once()
            review_signals.main()
            self.assertEqual(proposal["signal"], files["summary"][0]["next_signal"])
            self.api.assert_called_once()  # cached evidence, no extra review
            key = next(k for k in files["cache"] if ":candidate_validation:" in k)
            files["cache"][key]["comparisons"][0]["state_evidence"] = "An invented cancellation happened."
            review_signals.main()
            self.assertEqual("", files["summary"][0]["next_signal"])
            self.assertIn("earlier or unresolved timing", files["report"][0]["reason"])
            self.api.assert_called_once()

    def test_metadata_selection_preserves_late_sentence_and_following_cancellation(self):
        prefix = "Background information about the investigation. " * 12
        sentence = "The panel will meet next week."
        cancellation = "The meeting has since been cancelled."
        cluster = {"articles": [{"title": "Panel inquiry", "url": "https://example.com/panel",
                                "description": prefix + sentence + " " + cancellation,
                                "published_at": self.day}]}
        selected = review_signals.select_signal_evidence(cluster, self.day)
        self.assertIn(sentence, selected[0]["description"])
        self.assertIn(cancellation, selected[0]["source_context"])
        candidates = review_signals._model_evidence_candidates(self.day, selected)
        self.assertIn(cancellation, candidates[0]["source_context"])

    def test_metadata_selection_does_not_prioritize_unverified_relative_dates(self):
        articles = [{"title": "Panel inquiry", "description": "The inquiry is ongoing.",
                     "published_at": self.day, "url": f"https://example.com/{i}"} for i in range(12)]
        missing = {"title": "Next hearing", "description": "The panel will meet next week.",
                   "target_date": self.day, "url": "https://example.com/undated"}
        selected = review_signals.select_signal_evidence({"articles": articles + [missing]}, self.day)
        self.assertNotIn(missing["url"], [x["url"] for x in selected])
        selected = review_signals.select_signal_evidence({"articles": [missing]}, self.day)
        self.assertEqual("", selected[0]["published_at"])
        self.assertTrue(selected[0]["reference_date_unverified"])
        valid, reason = review_signals.validate_generated_signal_grounding(self.day, missing["description"], selected[0])
        self.assertFalse(valid)
        self.assertIn("publication date", reason)
        for invalid_date in ("unknown", "2026-02-30"):
            rows = review_signals.select_signal_evidence(
                {"articles": [{**missing, "published_at": invalid_date}]}, self.day)
            self.assertTrue(rows[0]["reference_date_unverified"])
            self.assertFalse(review_signals.validate_generated_signal_grounding(
                self.day, missing["description"], rows[0])[0])

    def test_duplicate_timed_metadata_does_not_fill_the_reserved_slots(self):
        articles = [{"title": "Panel inquiry", "description": "The inquiry is ongoing.",
                     "published_at": self.day, "url": f"https://example.com/{i}"} for i in range(6)]
        articles += [{"title": "Panel inquiry", "description": "The panel will meet next week.",
                      "published_at": self.day, "url": f"https://example.com/copy{i}"} for i in range(8)]
        articles.append({"title": "Panel inquiry", "description": "The panel will publish findings next month.",
                         "published_at": self.day, "url": "https://example.com/distinct"})
        selected = review_signals.select_signal_evidence({"articles": articles}, self.day)
        self.assertIn("https://example.com/distinct", [x["url"] for x in selected])
        self.assertEqual(12, len(selected))

    def test_metadata_selection_keeps_calendar_dates_as_source_dates(self):
        source = {"title": "Panel inquiry", "description": "The panel will meet Monday.",
                  "published_at": "2026-10-30T12:00:00Z", "url": "https://example.com/panel"}
        rows = review_signals.select_signal_evidence({"articles": [source]}, "2026-11-02")
        self.assertEqual(source["published_at"], rows[0]["published_at"])
        self.assertFalse(rows[0]["reference_date_unverified"])

    def test_direct_future_dates_survive_past_context_and_leading_dates(self):
        for text in (
            "After the hearing, the panel will publish its findings November 6.",
            "The panel will publish its findings November 6 after a hearing October 20.",
            "On November 6, the panel will publish its findings.",
            "The company must notify customers by November 6.",
            "The board will meet before November 6.",
            "The firm said that the board will meet November 6.",
        ):
            row = {"description": text, "grounded_sentence": text, "published_at": "2026-10-27"}
            with self.subTest(text=text):
                self.assertEqual("", review_signals._unresolved_future_alternative("2026-10-27", [row], 1))

    def test_potential_earlier_alternative_blocks_even_if_reviewer_calls_it_unrelated(self):
        for earlier, later in (("November 2", "November 6"), ("next week", "next year"),
                               ("next month", "November 6")):
            rows = [{"grounded_sentence": f"The panel will meet {when}.", "published_at": "2026-10-27"}
                    for when in (earlier, later)]
            with self.subTest(earlier=earlier, later=later):
                self.assertIn("Source 1", review_signals._unresolved_future_alternative("2026-10-27", rows, 2))
        # Equal/overlapping and provably later dates do not manufacture a strict
        # earlier checkpoint. The semantic review still has to approve.
        for other in ("November 6", "November 9"):
            rows = [{"grounded_sentence": f"The panel will meet {when}.", "published_at": "2026-10-27"}
                    for when in ("November 6", other)]
            self.assertEqual("", review_signals._unresolved_future_alternative("2026-10-27", rows, 1))

    def test_sentence_index_copies_saved_source_even_if_model_supplies_different_text(self):
        cases = load_fixture("oct27_sentence_selection")
        self.api.side_effect = None
        for case, index in [(cases[1], 2), (cases[4], 2)]:
            rows = case["evidence"]
            before = deepcopy(rows)
            expected = rows[index - 1]["grounded_sentence"]
            for writer in (review_signals.evaluate_signal_triage, review_signals.evaluate_next_signal):
                with self.subTest(story=case["topic"]["topic_title"], writer=writer.__name__):
                    self.api.reset_mock()
                    # Missing text is the real contract. A stray old/hostile text
                    # field cannot override the stored sentence either.
                    for extra in ({}, {"signal": "An unsupported event will happen tomorrow."}):
                        self.api.return_value = response({"status": "SIGNAL", "eligible": True,
                                                        "source_index": index, **extra})
                        result = writer(case["date"], case["topic"], rows)
                        self.assertEqual(expected, result["signal"])
                        self.assertEqual(index, result["source_index"])
                        self.assertTrue(review_signals.validate_signal_source_copy(result["signal"], rows[index - 1])[0])
                    self.assertEqual(2, self.api.call_count)  # One call per selection; no repair loop.
                    prompt = self.api.call_args.kwargs["messages"][1]["content"]
                    for row in rows:
                        self.assertIn(row["grounded_sentence"], prompt)
                    self.assertEqual(before, rows)

    def test_sentence_selection_keeps_uncopyable_earlier_alternative_for_review(self):
        early = {**self.row, "grounded_sentence": "They will release preliminary findings Monday.",
                 "description": "They will release preliminary findings Monday."}
        later = {**self.row, "grounded_sentence": "The panel will release final findings Wednesday.",
                 "description": "The panel will release final findings Wednesday."}
        rows = [early, later]
        self.api.side_effect = None
        self.api.return_value = response({"status": "SIGNAL", "source_index": 2})
        proposal = review_signals.evaluate_signal_triage(self.day, self.topic, rows)
        prompt = self.api.call_args.kwargs["messages"][1]["content"]
        self.assertIn("Whole-sentence copying: unavailable", prompt)
        self.assertIn(early["grounded_sentence"], prompt)
        self.api.return_value = response({**self.approval, "comparisons": [
            self.comparison(before=True, event="Preliminary findings", reason="Earlier direct checkpoint."),
            self.comparison(2, "Wednesday", event="Final findings"),
        ]})
        result = review_signals.review_signal_for_publication(self.day, self.topic, rows, proposal)
        self.assertFalse(result["eligible"])
        self.assertEqual("unresolved_timing", result["reason_code"])
        self.assertEqual(1, self.api.call_count)  # Local guard saves the review call.

    def test_sentence_selection_does_not_relax_existing_copy_guards(self):
        self.api.side_effect = None
        for source in ("They will meet Monday.", "The board will meet Monday...",
                       "The board will meet Monday if permission arrives."):
            row = {**self.row, "grounded_sentence": source, "description": source}
            self.api.return_value = response({"status": "SIGNAL", "source_index": 1,
                                            "signal": "The board will meet Monday."})
            proposal = review_signals.evaluate_signal_triage(self.day, self.topic, [row])
            self.assertEqual(source, proposal["signal"])
            if "if permission" not in source:
                self.api.reset_mock()
                result = review_signals.review_signal_for_publication(self.day, self.topic, [row], proposal)
                self.assertFalse(result["eligible"])
                self.api.assert_not_called()

    def test_sentence_selection_rejects_empty_source_without_retry(self):
        self.api.side_effect = None
        for writer in (review_signals.evaluate_signal_triage, review_signals.evaluate_next_signal):
            self.api.reset_mock()
            self.api.return_value = response({"status": "SIGNAL", "eligible": True, "source_index": 1})
            with self.assertRaisesRegex(RuntimeError, "without a sentence"):
                writer(self.day, self.topic, [{"grounded_sentence": "", "description": ""}])
            self.api.assert_called_once()

    def test_saved_oct19_reviews_cannot_skip_timed_alternatives(self):
        cases = load_fixture("oct19_timed_source_reviews")
        self.api.side_effect = None
        for i, case in enumerate(cases):
            with self.subTest(story=case["topic"]["topic_title"]):
                self.api.return_value = response(case["review"])
                result = review_signals.validate_candidate_signal_semantics(
                    case["date"], case["topic"], case["evidence"], case["proposal"])
                if i == 0:  # Both NIH deadline passages were inventoried.
                    self.assertTrue(result["eligible"])
                else:  # Saved approvals skipped earnings/expiry passages.
                    self.assertFalse(result["eligible"])
                    self.assertEqual("comparison_incomplete", result["reason_code"])
                    self.assertEqual("", result["signal"])
        self.assertEqual(3, self.api.call_count)  # Mocked, no automatic retries.

    def test_timed_background_can_be_excluded_but_cannot_be_omitted(self):
        self.api.side_effect = None
        earlier = {**self.row, "grounded_sentence": "The company will report earnings Sunday."}
        comparisons = [self.comparison(), self.comparison(
            2, "Sunday", qualifies=False, before=True, event="Quarterly earnings",
            reason="Pre-existing corporate reporting does not advance the panel review.")]
        self.api.return_value = response({**self.approval, "comparisons": comparisons})
        self.assertTrue(self.review(rows=[self.row, earlier])["eligible"])
        for field in ("event", "timing"):
            with self.subTest(missing=field):
                omitted = deepcopy(comparisons); omitted[1][field] = ""
                self.api.return_value = response({**self.approval, "comparisons": omitted})
                result = self.review(rows=[self.row, earlier])
                self.assertFalse(result["eligible"])
                self.assertEqual("comparison_incomplete", result["reason_code"])

    def test_inventoried_earlier_expiry_blocks_later_event_on_other_dates(self):
        self.api.side_effect = None
        for day in ("2027-03-01", "2031-09-01"):
            with self.subTest(day=day):
                rows = [{"description": "The council will meet Saturday.", "published_at": day},
                        {"description": "The temporary restrictions remain in effect until Tuesday.", "published_at": day}]
                proposal = {"eligible": True, "source_index": 1, "signal": rows[0]["description"]}
                self.api.return_value = response({"eligible": True, "comparisons": [
                    self.comparison(1, "Saturday", event="Council meeting"),
                    self.comparison(2, "Tuesday", before=True, event="Restrictions expire",
                                    reason="Earlier material change to the restrictions.")], "reason": "Supported."})
                result = review_signals.validate_candidate_signal_semantics(
                    day, {"topic_title": "Temporary restrictions", "summary": "Council imposes restrictions."}, rows, proposal)
                self.assertFalse(result["eligible"])
                self.assertEqual("comparison_conflict", result["reason_code"])

    def test_background_dates_do_not_become_future_checkpoints(self):
        rows = [{"description": "The panel met last Monday."},
                {"description": "A decision could follow within 90 days if the review starts."}]
        comparisons = [self.comparison(1, "Monday", qualifies=False, event="Past meeting"),
                       self.comparison(2, "within 90 days", qualifies=False, event="Conditional decision")]
        self.assertEqual("", review_signals._source_comparison_coverage_gap(self.day, rows, comparisons))
        # This check only audits completeness; it never promotes either event.
        self.assertTrue(all(not c["qualifies"] for c in comparisons))

    def test_plural_references_keep_expiry_context_across_dates(self):
        for day in (date(2026, 10, 19), date(2031, 9, 1)):
            for subject in ("Both", "These", "Those"):
                with self.subTest(day=day, subject=subject):
                    text = ("Officials imposed a curfew and restricted gatherings in the city. "
                            + subject + " would remain in effect until 6 am on Tuesday morning.")
                    snippets = review_signals._future_milestone_snippets(text, day)
                    matches = [s["text"] for s in snippets if "Tuesday" in s["text"]]
                    self.assertTrue(matches)
                    self.assertTrue(all("curfew" in s for s in matches))

    def test_local_calendar_blocks_incorrect_model_order_across_dates(self):
        self.api.side_effect = None
        for day in ("2026-10-19", "2027-03-01", "2031-09-01"):
            rows = [{"published_at": day, "description": "The council will meet Saturday."},
                    {"published_at": day, "description": "The restrictions will remain in effect until Tuesday."}]
            p = {"eligible": True, "source_index": 1, "signal": rows[0]["description"]}
            comparisons = [self.comparison(1, "Saturday", event="Council meeting"),
                           self.comparison(2, "Tuesday", before=False, event="Restrictions end")]
            with self.subTest(day=day):
                self.api.return_value = response({"eligible": True, "comparisons": comparisons})
                result = review_signals.validate_candidate_signal_semantics(day, self.topic, rows, p)
                self.assertFalse(result["eligible"])
                self.assertEqual("calendar_conflict", result["reason_code"])
                comparisons[1]["qualifies"] = False
                self.api.return_value = response({"eligible": True, "comparisons": comparisons})
                self.assertTrue(review_signals.validate_candidate_signal_semantics(day, self.topic, rows, p)["eligible"])

    def test_local_calendar_does_not_shrink_windows_or_reanchor_old_copy(self):
        day = "2026-10-19"
        for sentence, timing, published in (
            ("The storm will turn Saturday night into Sunday.", "Saturday", day),
            ("The order will apply from October 12 to October 24.", "October 24", day),
            ("The panel will decide within 90 days if approval arrives.", "within 90 days", day),
            ("The panel will meet Tuesday.", "Tuesday", "2026-10-12"),
            ("The panel met last Tuesday.", "Tuesday", day),
            ("The panel will decide by Tuesday.", "Tuesday", day),
            ("The panel will decide after October 24.", "October 24", day),
            ("The panel will meet next Tuesday.", "Tuesday", day),
            ("The panel will meet Monday.", "Monday", day),
        ):
            with self.subTest(sentence=sentence):
                self.assertIsNone(review_signals._comparison_calendar_day(day, {"description": sentence, "published_at": published}, timing))

    def test_saved_france_calendar_requires_relevance_and_orders_dates(self):
        case = load_fixture("oct19_timed_source_reviews")[2]
        rows = case["evidence"]
        comparisons = deepcopy(case["review"]["comparisons"])
        comparisons[3].update(event="Paris restrictions end", timing="Tuesday", qualifies=True, before_selected=False)
        conflict = review_signals._local_calendar_conflict(case["date"], rows, comparisons, 1)
        self.assertIn("2026-10-20", conflict)
        self.assertIn("2026-10-24", conflict)
        # Relevant November events are later; they cannot block October 24.
        comparisons[3]["qualifies"] = False
        for c in comparisons[1:3]:
            c["qualifies"] = True
        self.assertEqual("", review_signals._local_calendar_conflict(case["date"], rows, comparisons, 1))
        self.assertIn("endpoint separately", review_signals._checkpoint_boundary_notes(rows))

    def test_saved_review_cannot_substitute_start_clock_for_expiry_day(self):
        case = load_fixture("oct19_timed_source_reviews")[2]
        raw = load_fixture("france_start_time_instead_of_expiry")
        self.api.side_effect = None
        self.api.return_value = response(raw)
        result = review_signals.validate_candidate_signal_semantics(
            case["date"], case["topic"], case["evidence"], case["proposal"])
        self.assertFalse(result["eligible"])
        self.assertEqual("comparison_incomplete", result["reason_code"])
        self.assertIn("source 4", result["reason"])
        self.api.assert_called_once()  # Replay only; no retries or replacement.

    def test_checkpoint_time_coverage_is_not_story_or_date_specific(self):
        for day in ("2026-10-19", "2031-09-01"):
            row = {"published_at": day, "description":
                   "Access was restricted from 10pm. Both restrictions will end Tuesday morning."}
            wrong = [self.comparison(timing="10pm", qualifies=False)]
            right = [self.comparison(timing="Tuesday", qualifies=False)]
            self.assertTrue(review_signals._source_comparison_coverage_gap(day, [row], wrong))
            self.assertEqual("", review_signals._source_comparison_coverage_gap(day, [row], right))

    def test_local_calendar_equal_days_and_rejections_remain_unchanged(self):
        self.api.side_effect = None
        self.api.return_value = response({**self.approval, "comparisons": [self.comparison(), self.comparison(2)]})
        self.assertTrue(self.review(rows=[self.row, self.row])["eligible"])
        self.api.return_value = response({**self.approval, "eligible": False, "reason": "Unrelated checkpoint."})
        self.assertFalse(self.review()["eligible"])

    def test_saved_supported_wording_survives_without_reconstruction(self):
        cases = [
            ("Kelly scheduled a hearing on the CNN/MS NOW/Politico suit for Wednesday afternoon.",
             "Kelly scheduled a hearing on the CNN/MS NOW/Politico suit for Wednesday afternoon, and the Justice Department replied late Tuesday to the suit, saying the ban was at least partly on national security grounds."),
            ("A hearing is scheduled for October 7.", "A hearing is scheduled for October 7."),
            ("A 45-year-old man charged with murder will appear in court Monday.",
             "He was charged with murder and will appear in court Monday."),
        ]
        self.api.side_effect = None
        for (signal, source), timing in zip(cases, ("Wednesday afternoon", "October 7", "Monday")):
            with self.subTest(signal=signal):
                self.api.return_value = response({**self.approval, "signal": "An unwanted rewrite.", "source_index": 99,
                                                 "comparisons": [self.comparison(timing=timing)]})
                row = {**self.row, "description": source, "grounded_sentence": source}
                result = self.review({**self.proposal, "signal": signal}, [row])
                self.assertTrue(result["eligible"])
                self.assertEqual(signal, result["signal"])
                self.assertEqual(1, result["source_index"])
        self.assertEqual(3, self.api.call_count)

    def test_present_due_schedule_across_subjects_and_years(self):
        from datetime import timedelta
        self.api.side_effect = None
        for reference in (date(2026, 10, 13), date(2027, 12, 29), date(2030, 2, 20)):
            timing = (reference + timedelta(days=6)).strftime("%B %d, %Y")
            for clause in ("The defendant is due to appear in court",
                           "The researchers are due to publish their findings",
                           "The spacecraft is due to arrive",
                           "I am due to testify"):
                signal = f"{clause} on {timing}."
                row = {"description": signal, "grounded_sentence": signal,
                       "published_at": reference.isoformat()}
                with self.subTest(reference=reference, clause=clause):
                    self.assertTrue(review_signals._model_evidence_candidates(reference.isoformat(), [row]))
                    self.api.return_value = response({**self.approval,
                        "comparisons": [self.comparison(timing=timing)]})
                    result = review_signals.validate_candidate_signal_semantics(
                        reference.isoformat(), self.topic, [row], {**self.proposal, "signal": signal})
                    self.assertTrue(result["eligible"])
                    self.assertEqual(signal, result["signal"])
        self.assertEqual(12, self.api.call_count)

    def test_due_schedule_keeps_past_dates_and_causal_phrases_out_of_publication(self):
        for signal in ("The defendant was due to appear on October 19, 2026.",
                       "The defendant is due to appear on October 12, 2026.",
                       "The hearing ended due to rain on October 19, 2026.",
                       "Due to rain, the hearing ended on October 19, 2026."):
            row = {"description": signal, "grounded_sentence": signal, "published_at": "2026-10-13"}
            with self.subTest(signal=signal):
                self.assertFalse(review_signals.validate_generated_signal_grounding("2026-10-13", signal, row)[0])
        self.api.assert_not_called()
        # Present-tense causal language is ambiguous locally; it must still be reviewed.
        signal = "The closure is due to rain on October 19, 2026."
        row = {"description": signal, "grounded_sentence": signal, "published_at": "2026-10-13"}
        self.api.side_effect = None
        self.api.return_value = response({"eligible": False, "reason": "Cause, not a scheduled next step.",
            "comparisons": [self.comparison(timing="October 19, 2026", qualifies=False)]})
        result = review_signals.validate_candidate_signal_semantics(
            "2026-10-13", self.topic, [row], {**self.proposal, "signal": signal})
        self.assertFalse(result["eligible"])
        self.api.assert_called_once()

    def test_saved_october_hearing_reaches_reviewer_without_changing_source(self):
        path = "next_signal_report_2026-10-13.json"
        saved = load_fixture(path)[5]
        proposal = saved["triage_decision"]
        signal = proposal["signal"]
        self.assertIn(signal, [x.get("excerpt") for x in saved["sidecar_attempted"]])
        row = {"description": signal, "grounded_sentence": signal, "published_at": "2026-10-13"}
        self.api.side_effect = None
        self.api.return_value = response({**self.approval,
            "comparisons": [self.comparison(timing="Oct. 19")]})
        result = review_signals.validate_candidate_signal_semantics("2026-10-13", self.topic, [row], proposal)
        self.assertTrue(result["eligible"])
        self.assertEqual(signal, result["signal"])
        self.api.assert_called_once()
        context = review_signals._signal_review_context(proposal)
        current = review_signals._cache_key("2026-10-13", self.topic, [row], "candidate_validation", context)
        with patch.object(review_signals, "VALIDATION_PROMPT_VERSION", "next-signal-validation-v14-countdown-scope-2026-10-12"):
            self.assertNotEqual(current, review_signals._cache_key("2026-10-13", self.topic, [row], "candidate_validation", context))

    def test_october_saved_proposals_only_hearing_changes_local_gate(self):
        import re
        root = Path(__file__).resolve().parents[1]
        report = load_fixture("next_signal_report_2026-10-13.json")
        old = re.compile(review_signals._FUTURE_FRAMING_RE.pattern.replace(
            r"(?:am|is|are)\s+due\s+to|", ""), re.I)
        changes = []
        for item in report:
            proposal = item.get("triage_decision") or {}
            if proposal.get("status") != "SIGNAL":
                continue
            # Exact proposals and their saved excerpts; no fresh writer calls.
            rows = [x["excerpt"] for x in item["sidecar_attempted"] if x.get("excerpt")]
            outcomes = []
            for pattern in (old, review_signals._FUTURE_FRAMING_RE):
                with patch.object(review_signals, "_FUTURE_FRAMING_RE", pattern):
                    outcomes.append([review_signals.validate_generated_signal_grounding(
                        "2026-10-13", proposal["signal"],
                        {"grounded_sentence": text, "published_at": "2026-10-13"}) for text in rows])
            if outcomes[0] != outcomes[1]:
                changes.append(item["topic_index"])
        self.assertEqual([6], changes)
        self.api.assert_not_called()

    def test_review_input_is_neutral_and_excludes_writer_rationale(self):
        self.api.side_effect = None
        self.api.return_value = response(self.approval)
        self.review()
        prompt = self.api.call_args.kwargs["messages"][1]["content"]
        self.assertNotIn("WRITER_RATIONALE_SENTINEL", prompt)
        self.assertNotIn("WRITER_COMPARISON_SENTINEL", prompt)
        self.assertNotIn("Grounded future sentence", prompt)
        self.assertIn("Candidate source sentence: " + self.row["description"], prompt)
        self.assertIn("Detected time references (unverified):", prompt)
        self.assertEqual(review_signals.REVIEW_MODEL, self.api.call_args.kwargs["model"])
        self.assertEqual(470, self.api.call_args.kwargs["max_completion_tokens"])

    def test_structural_rejections_spend_no_review_call(self):
        source = "The justices halted a federal appeals court ruling for the November elections, which will determine control of Congress."
        row = {**self.row, "description": source, "grounded_sentence": source}
        bad = [
            ({**self.proposal, "signal": "The justices halted a federal appeals court ruling for the November elections."}, row),
            ({**self.proposal, "signal": "The panel may release its findings Monday and Wednesday."}, self.row),
            ({**self.proposal, "source_index": True}, self.row),
            ({**self.proposal, "source_index": 2}, self.row),
            ({**self.proposal, "signal": "word " * 23}, self.row),
        ]
        for proposal, evidence in bad:
            with self.subTest(proposal=proposal):
                result = self.review(proposal, [evidence])
                self.assertFalse(result["eligible"])
                self.assertFalse(result["review_model_called"])
        self.api.assert_not_called()

    def test_fronted_date_and_qualified_language_reach_review(self):
        self.api.side_effect = None
        self.api.return_value = response(self.approval)
        for text in ("On Monday, the panel will publish its findings.",
                     "The panel may publish its findings Monday if the review ends."):
            with self.subTest(text=text):
                row = {**self.row, "description": text, "grounded_sentence": text}
                self.assertTrue(self.review({**self.proposal, "signal": text}, [row])["eligible"])
        self.assertEqual(2, self.api.call_count)

    def test_separate_clause_uncertainty_can_reach_review(self):
        signal = "The panel will release findings Monday."
        sources = [
            "The panel will release findings Monday, though later trials may take longer.",
            "The panel will release findings Monday; later trials may take longer.",
            "The panel will release findings Monday, with possible delays to later trials.",
        ]
        self.api.side_effect = None
        # A local pass still needs semantic approval. Simulate rejection to check
        # that narrowing the local gate cannot bypass that existing review.
        self.api.return_value = response({**self.approval, "eligible": False,
                                         "reason": "The checkpoint is not material to this story."})
        for source in sources:
            with self.subTest(source=source):
                row = {**self.row, "description": source, "grounded_sentence": source}
                self.assertEqual((True, ""), review_signals.validate_generated_signal_grounding(self.day, signal, row))
                result = self.review({**self.proposal, "signal": signal}, [row])
                self.assertTrue(result["review_model_called"])
                self.assertFalse(result["eligible"])
                self.assertEqual("", result["signal"])
                self.assertIn(source, self.api.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(len(sources), self.api.call_count)

    def test_same_event_and_ambiguous_uncertainty_remain_blocked_locally(self):
        signal = "The panel will release findings Monday."
        sources = [
            f"The panel {modal} release findings Monday."
            for modal in ("may", "might", "could", "is expected to", "is likely to", "is forecast to")
        ] + [
            "The panel may release findings Monday, though delegates will meet Monday.",
            "The panel will release findings Monday, though the release may be delayed.",
            "The panel will release findings Monday; it may be delayed.",
            "The panel will release findings Monday, with uncertainty about that schedule.",
            "It is possible the panel will release findings Monday, while delegates will meet Tuesday.",
            "The panel will release findings Monday if the review ends, which may take longer.",
        ]
        for source in sources:
            with self.subTest(source=source):
                row = {**self.row, "description": source, "grounded_sentence": source}
                result = self.review({**self.proposal, "signal": signal}, [row])
                self.assertFalse(result["eligible"])
                self.assertFalse(result["review_model_called"])
                self.assertIn("uncertainty", result["reason"])
        self.api.assert_not_called()

    def test_negative_triage_discards_stray_sentence(self):
        for status in ("NO_SIGNAL", "RESEARCH_NEEDED"):
            self.api.side_effect = None
            self.api.return_value = response({"status": status, "signal": "Stray draft",
                "source_index": 1, "reason": "No supported event", "search_queries": []})
            result = review_signals.evaluate_signal_triage(self.day, self.topic, [self.row])
            self.assertFalse(result["eligible"])
            self.assertEqual(result["signal"], "")
            self.assertIsNone(result["source_index"])

    def test_expired_source_day_is_not_next_week(self):
        sentence = "The panel will release its findings Friday night."
        row = {"description": sentence, "published_at": "2026-10-16"}
        self.assertFalse(review_signals.validate_generated_signal_grounding("2026-10-17", sentence, row)[0])
        self.assertTrue(review_signals.validate_generated_signal_grounding("2026-10-16", sentence, row)[0])
        future = "The panel will release its findings Sunday."
        self.assertTrue(review_signals.validate_generated_signal_grounding("2026-10-17", future,
            {"description": future, "published_at": "2026-10-16"})[0])

    def test_past_plan_cannot_be_rewritten_as_future(self):
        for past in ("The panel members were to meet Friday night.",
                     "The panel was to meet Friday night."):
            row = {"grounded_sentence": past, "published_at": "2026-10-17"}
            ok, reason = review_signals.validate_generated_signal_grounding(
                "2026-10-17", "The panel will meet Friday night.", row)
            self.assertFalse(ok)
            self.assertIn("past plan", reason)
        future = "The panel will meet Sunday."
        row = {"grounded_sentence": future, "description": "They were to meet Friday. " + future,
               "published_at": "2026-10-17"}
        self.assertTrue(review_signals.validate_generated_signal_grounding("2026-10-17", future, row)[0])

    def test_future_obligations_and_announced_plans_reach_semantic_review(self):
        cases = [
            ("The new division must be established within 90 days.",
             "The order signed today requires the new division to be established within 90 days."),
            ("The operator is required to submit its findings by November 5.",
             "The operator is required to submit its findings by November 5."),
            ("Unions have called for strikes and rallies on 5 November.",
             "Unions called for strikes and rallies on 5 November."),
        ]
        self.api.side_effect = None
        for signal, sentence in cases:
            row = {"description": sentence, "grounded_sentence": sentence, "published_at": "2026-10-19"}
            self.assertTrue(review_signals.validate_generated_signal_grounding("2026-10-19", signal, row)[0])
            timing = "within 90 days" if "90" in sentence else "November 5" if "November 5" in sentence else "5 November"
            self.api.return_value = response({"eligible": False, "reason": "Not a material checkpoint.",
                "comparisons": [self.comparison(timing=timing, qualifies=False)]})
            result = review_signals.validate_candidate_signal_semantics("2026-10-19", self.topic, [row],
                {"eligible": True, "signal": signal, "source_index": 1})
            self.assertTrue(result['review_model_called'])
            self.assertFalse(result['eligible'])

    def test_obligation_gate_keeps_conditional_completed_and_cancelled_cases_closed(self):
        cases = [
            ("The division must be established within 90 days.",
             "If approved, the division must be established within 90 days."),
            ("The operator must have submitted its findings on November 5.",
             "The operator must have submitted its findings on November 5."),
            ("Unions have not called for strikes on 5 November.",
             "Unions have not called for strikes on 5 November."),
            ("Unions have called for strikes on 5 November.",
             "Unions have called for strikes on 5 November."),
        ]
        for i, (signal, sentence) in enumerate(cases):
            day = "2026-11-06" if i == 3 else "2026-10-19"
            self.assertFalse(review_signals.validate_generated_signal_grounding(day, signal,
                {"description": sentence, "published_at": "2026-10-19"})[0])
        self.api.assert_not_called()

    def test_quoted_commitment_requires_attribution(self):
        row = {"title": "Founder pledges shares to employees", "published_at": "2026-10-19",
               "grounded_sentence": "Next year, I will give shares to every employee."}
        ok, reason = review_signals.validate_generated_signal_grounding("2026-10-19",
            "Next year, the founder will give shares to every employee.", row)
        self.assertFalse(ok)
        self.assertIn('attribution', reason)
        for signal in ["The founder pledges to give shares to every employee next year.",
                       "The founder says she will give shares to every employee next year."]:
            self.assertTrue(review_signals.validate_generated_signal_grounding("2026-10-19", signal, row)[0])
        self.api.assert_not_called()

    def test_reported_commitment_cannot_become_a_guaranteed_outcome(self):
        row = {"published_at": "2026-10-19",
               "grounded_sentence": "The director promised to release the results next month."}
        self.assertFalse(review_signals.validate_generated_signal_grounding("2026-10-19",
            "The director will release the results next month.", row)[0])
        self.assertTrue(review_signals.validate_generated_signal_grounding("2026-10-19",
            "The director promised to release the results next month.", row)[0])
        self.api.assert_not_called()

    def test_day_first_calendar_dates_preserve_day_year_and_expiry(self):
        for wording in ("5 November", "5 November 2026", "5th November 2026", "5th of November 2026", "November 5, 2026"):
            sentence = "The board will meet on " + wording + "."
            row = {"description": sentence, "published_at": "2026-10-19"}
            anchors = review_signals._future_anchor_records(sentence, date(2026, 10, 19))
            self.assertIn('date:2026-11-05', [a['key'] for a in anchors])
            self.assertTrue(review_signals.validate_generated_signal_grounding('2026-10-19', sentence, row)[0])
            self.assertFalse(review_signals.validate_generated_signal_grounding('2026-11-06', sentence, row)[0])
            self.assertFalse(review_signals.validate_generated_signal_grounding('2026-10-19',
                'The board will meet on 6 November 2026.', row)[0])
        self.api.assert_not_called()

    def test_project_noun_does_not_turn_an_obligation_into_a_forecast(self):
        source = ("The newly minted project, spearheaded by the first lady, requires the new health division "
                  "to be established within 90 days.")
        signal = "The new health division must be established within 90 days."
        self.assertTrue(review_signals.validate_generated_signal_grounding('2026-10-19', signal,
            {'description': source, 'published_at': '2026-10-19'})[0])
        for source in ("The division is projected to open next month.",
                       "Forecasters are projecting the division will open next month."):
            self.assertFalse(review_signals.validate_generated_signal_grounding('2026-10-19',
                'The division will open next month.', {'description': source, 'published_at': '2026-10-19'})[0])
        self.api.assert_not_called()

    def test_announced_plan_cannot_bypass_an_earlier_material_checkpoint(self):
        rows = [
            {"description": "Unions called for strikes on 5 November.", "published_at": "2026-10-19"},
            {"description": "The curfew is scheduled to expire on October 20.", "published_at": "2026-10-19"},
        ]
        self.api.side_effect = None
        self.api.return_value = response({"eligible": True, "reason": "Supported.", "comparisons": [
            self.comparison(index=1, timing="5 November", event="Strikes"),
            self.comparison(index=2, timing="October 20", event="Curfew expiry", before=True),
        ]})
        result = review_signals.validate_candidate_signal_semantics("2026-10-19", self.topic, rows,
            {"eligible": True, "signal": "Unions have called for strikes on 5 November.", "source_index": 1})
        self.assertFalse(result['eligible'])
        self.assertIn('Earlier qualifying', result['reason'])

    def test_countdown_cannot_invent_start_date_from_publication(self):
        row = {"published_at": "2026-10-17", "description":
               "The defence team indicated its intention to appeal against the sentence and have to do so within 10 days."}
        for start in ("October 17, 2026", "2026-10-17", "October 16, 2026"):
            ok, reason = review_signals.validate_generated_signal_grounding("2026-10-17",
                "The defence team will appeal within 10 days of " + start + ".", row)
            self.assertFalse(ok)
            self.assertIn("calendar date", reason)

    def test_saved_storm_reply_reaches_review_without_bypassing_its_verdict(self):
        path = "focused_review.json"
        saved = load_fixture(path)["required_comparison_storm_review"]
        call = saved["calls"][0]
        raw = json.loads(call["response"])
        self.api.side_effect = None
        self.api.return_value = response(raw, call["finish_reason"])
        proposal = review_signals.evaluate_signal_triage(saved["date"], saved["topic"], saved["evidence"])
        self.assertTrue(proposal["eligible"])
        expected = saved["evidence"][raw["source_index"] - 1]["grounded_sentence"]
        self.assertEqual(expected, proposal["signal"])
        self.assertEqual(raw["source_index"], proposal["source_index"])
        self.assertNotIn("comparisons", proposal)
        self.assertEqual([], proposal["search_queries"])
        # Replaying the old proposal does not establish better model selection.
        comparisons = [{**item, "before_selected": False} for item in raw["comparisons"]]
        self.api.return_value = response({"eligible": False, "comparisons": comparisons,
                                         "reason": "Simulated rejection of the proposed checkpoint."})
        result = review_signals.validate_candidate_signal_semantics(
            saved["date"], saved["topic"], saved["evidence"], proposal,
        )
        self.assertEqual(2, self.api.call_count)  # Mocked writer plus existing reviewer.
        self.assertTrue(result["review_model_called"])
        self.assertFalse(result["eligible"])
        self.assertEqual("", result["signal"])
        self.assertEqual(expected, result["proposed_signal"])
        prompt = self.api.call_args.kwargs["messages"][1]["content"]
        self.assertIn(saved["evidence"][0]["grounded_sentence"], prompt)
        self.assertNotIn("No explicit future event mentioned.", prompt)

    def test_rejection_is_retained_and_invalid_verdict_is_an_error(self):
        self.api.side_effect = None
        for code in ("actor", "action", "timing", "condition", "relevance", "earlier_checkpoint"):
            self.api.return_value = response({**self.approval, "eligible": False, "reason_code": code, "reason": "Unsupported."})
            result = self.review()
            self.assertEqual("", result["signal"])
            self.assertEqual(self.proposal["signal"], result["proposed_signal"])
        for payload, finish in ((self.approval, "length"), ({}, "stop"),
                                ({**self.approval, "eligible": "true"}, "stop"),
                                ({**self.approval, "eligible": 1}, "stop"),
                                ({**self.approval, "eligible": None}, "stop")):
            self.api.return_value = response(payload, finish)
            with self.assertRaises(RuntimeError):
                self.review()

    def test_optional_metadata_cannot_change_a_boolean_verdict(self):
        self.api.side_effect = None
        for eligible in (True, False):
            for extra in ({}, {"reason": None}, {"reason": ["invalid metadata"]},
                          {"reason": "  "}, {"reason_code": "actor|action|timing|none"}):
                with self.subTest(eligible=eligible, extra=extra):
                    self.api.return_value = response({"eligible": eligible, "comparisons": [self.comparison()], **extra})
                    result = self.review()
                    self.assertIs(result["eligible"], eligible)
                    self.assertEqual(self.proposal["signal"] if eligible else "", result["signal"])
                    self.assertIn("without an explanation", result["reason"])
                    self.assertEqual("none" if eligible else "review_rejection", result["reason_code"])

    def test_older_paid_replies_cannot_bypass_required_comparisons(self):
        # Old verdict-only replies lack the newly required comparison. They are
        # retained on disk, but cannot masquerade as responses to this contract.
        path = "focused_review.json"
        saved = load_fixture(path)["fixed_proposal_review"]
        calls = {item["case"]: item for item in saved["calls"]}
        expected = {"Lawsuit hearing": True, "Estate hearing": False, "Tasia appearance": False,
                    "Iran participants": False, "Australian disclosure": False, "Sheeran tour": False}
        self.api.side_effect = None
        for case in saved["cases"]:
            if case["case"] not in expected:
                continue
            with self.subTest(case=case["case"]):
                call = calls[case["case"]]
                self.api.return_value = response(json.loads(call["response"]), call["finish_reason"])
                with self.assertRaisesRegex(RuntimeError, "compare every source"):
                    review_signals.validate_candidate_signal_semantics(
                        case["date"], case["topic"], case["evidence"], case["proposal"],
                    )
        self.assertEqual(6, self.api.call_count)

    def test_cited_passage_is_first_without_renumbering_other_sources(self):
        self.api.side_effect = None
        rows = [{**self.row, "title": title} for title in ("FIRST_PASSAGE", "CITED_PASSAGE", "THIRD_PASSAGE")]
        self.api.return_value = response({**self.approval, "comparisons": [self.comparison(i) for i in (1, 2, 3)]})
        result = self.review({**self.proposal, "source_index": 2}, rows)
        prompt = self.api.call_args.kwargs["messages"][1]["content"]
        cited = prompt.split("CITED PASSAGE —", 1)[1].split("Check the cited passage:", 1)[0]
        others = prompt.split("OTHER PASSAGES —", 1)[1].split("Compare events before comparing dates:", 1)[0]
        self.assertIn("[2]", cited)
        self.assertIn("CITED_PASSAGE", cited)
        self.assertNotIn("FIRST_PASSAGE", cited)
        self.assertNotIn("THIRD_PASSAGE", cited)
        self.assertIn("Reference date for relative timing: " + self.day, cited)
        self.assertIn("[1]", others)
        self.assertIn("FIRST_PASSAGE", others)
        self.assertIn("[3]", others)
        self.assertIn("THIRD_PASSAGE", others)
        self.assertNotIn("[2]", others)
        self.assertNotIn("CITED_PASSAGE", others)
        self.assertEqual(2, result["source_index"])

    def test_cache_context_tracks_wording_and_source_only(self):
        key = review_signals._signal_review_context(self.proposal)
        self.assertEqual(key, review_signals._signal_review_context({**self.proposal, "reason": "changed"}))
        for field, value in (("signal", "The panel may publish its findings Monday."), ("source_index", 2)):
            self.assertNotEqual(key, review_signals._signal_review_context({**self.proposal, field: value}))
        self.assertEqual(key, review_signals._signal_review_context({**self.proposal, "comparisons": []}))

    def test_saved_storm_timing_comparison_reaches_each_existing_stage(self):
        path = "focused_review.json"
        saved = load_fixture(path)["daily_signal_review_2026_09_25"]
        title = "Nor'easter Hits East Coast Causing Widespread Damage and Disruptions"
        rows = saved["model_evidence"][title]
        before = deepcopy(rows)
        comparison = review_signals._checkpoint_timing_comparison(self.day, rows)
        self.assertIn('[3] "Saturday night" => 2026-09-26 if upcoming; "Sunday" => 2026-09-27 if upcoming', comparison)
        self.assertIn('[2] "Monday" => 2026-09-28 if upcoming', comparison)
        self.assertLess(comparison.index("[3]"), comparison.index("[2]"))
        self.assertIn('[4] "within 30 days" => unresolved timing', comparison)
        signal = "The storm is expected to turn toward New Jersey Saturday night into Sunday."
        comparisons = [
            self.comparison(1, "Overnight into Sunday morning", event="Rain and wind continue", reason="Overlapping forecast window."),
            self.comparison(2, "into Monday", event="Possible lingering impacts", reason="Later than the selected turn."),
            self.comparison(3, "Saturday night into Sunday", event="Expected turn toward New Jersey", reason="Earlier material change in storm path."),
            self.comparison(4, "within 30 days", False, None, event="Ticket refunds", reason="Ancillary to the storm's forecast path."),
        ]
        proposal = {**self.proposal, "signal": signal, "source_index": 3, "comparisons": comparisons}
        topic = {"topic_title": title, "summary": "The storm is causing damage and disruptions."}
        self.api.side_effect = None
        writer_reply = {**proposal, "comparisons": self.writer_assessments(comparisons)}
        for function, args, reply, cap, per_source in (
            (review_signals.evaluate_signal_triage, (), writer_reply, 260, 0),
            (review_signals.evaluate_next_signal, (), writer_reply, 220, 0),
            (review_signals.validate_candidate_signal_semantics, (proposal,), {**self.approval, "comparisons": comparisons}, 380, 90),
        ):
            with self.subTest(stage=function.__name__):
                self.api.reset_mock()
                self.api.return_value = response(reply)
                result = function(self.day, topic, rows, *args)
                self.api.assert_called_once()
                request = self.api.call_args.kwargs
                expected_comparison = review_signals._checkpoint_timing_comparison(
                    self.day, rows, include_day_offsets=function == review_signals.validate_candidate_signal_semantics)
                self.assertIn(expected_comparison, request["messages"][1]["content"])
                self.assertEqual(review_signals.REVIEW_MODEL if "max_completion_tokens" in request else review_signals.MODEL, request["model"])
                status_tokens = 120 * (len(rows) - 1) if function == review_signals.validate_candidate_signal_semantics else 0
                self.assertEqual(cap + per_source * len(rows) + status_tokens, request.get("max_tokens", request.get("max_completion_tokens")))
                self.assertTrue(request["response_format"]["json_schema"]["strict"])
                expected = signal if function == review_signals.validate_candidate_signal_semantics else rows[2]["grounded_sentence"]
                self.assertEqual(expected, result["signal"])
                self.assertEqual(3, result["source_index"])
        self.assertEqual(before, rows)

    def test_calendar_comparison_preserves_equal_dates_and_source_references(self):
        texts = ["A hearing is scheduled for October 7.", "A hearing is scheduled for Oct. 7.",
                 "The panel will report Monday.", "The panel will report Monday."]
        rows = [{**self.row, "grounded_sentence": text} for text in texts]
        rows[3]["grounded_reference_date"] = "2026-09-29"
        rows[3]["published_at"] = "2026-09-29"
        comparison = review_signals._checkpoint_timing_comparison(self.day, rows)
        self.assertIn('[1] "October 7" => 2026-10-07', comparison)
        self.assertIn('[2] "Oct. 7" => 2026-10-07', comparison)
        self.assertIn('[3] "Monday" => 2026-09-28', comparison)
        self.assertIn('[4] "Monday" => 2026-10-05', comparison)
        self.assertLess(comparison.index("[3]"), comparison.index("[4]"))
        self.assertLess(comparison.index("[4]"), comparison.index("[1]"))

    def test_explicit_deadlines_reach_review_with_original_dates(self):
        cases = [
            ("The judge gave both sides until Oct. 5 to file their arguments over whether he should consider issuing a longer-lasting ruling called a preliminary injunction.",
             "Both sides must file arguments by October 5 on whether to issue a preliminary injunction.", "Oct. 5"),
            ("The teams have until October 7 to submit their revised designs for the next assessment.",
             "The teams have until October 7 to submit their revised designs.", "October 7"),
            ("The company must notify affected customers by October 9 before making the planned service changes.",
             "The company must notify affected customers by October 9.", "October 9"),
        ]
        self.api.side_effect = None
        for source, signal, timing in cases:
            with self.subTest(source=source):
                row = {**self.row, "description": source, "grounded_sentence": source}
                evidence = review_signals._model_evidence_candidates(self.day, [row])
                self.assertEqual([source], [r["grounded_sentence"] for r in evidence])
                self.api.return_value = response({**self.approval, "comparisons": [self.comparison(timing=timing)]})
                result = self.review({**self.proposal, "signal": signal}, evidence)
                self.assertTrue(result["eligible"])
                self.assertEqual(signal, result["signal"])
        self.assertEqual(len(cases), self.api.call_count)

    def test_deadline_exception_does_not_start_duration_clocks_or_revive_past_dates(self):
        for source in (
            "The rule requires applicants to notify the agency within 30 days.",
            "The parties must have filed arguments by October 5.",
            "The judge never gave both sides until October 5 to file arguments.",
            "The team had until October 5 to submit its report.",
            "The team has until October 5 to submit its report, but the requirement was rescinded.",
            "The judge gave both sides until September 22 to file their arguments on the longer-lasting order.",
        ):
            with self.subTest(source=source):
                row = {**self.row, "description": source, "grounded_sentence": source}
                result = self.review({**self.proposal, "signal": source}, [row])
                self.assertFalse(result["eligible"])
                self.assertFalse(result["review_model_called"])
        self.api.assert_not_called()

    def test_same_day_forecasts_preserve_period_and_uncertainty(self):
        cases = [
            ("Moderate to major coastal flooding is possible during high tide later Saturday evening from the southern Chesapeake Bay to the Jersey Shore and up to Raritan Bay.",
             "Coastal flooding is possible during high tide later Saturday evening.", "later Saturday evening"),
            ("On Saturday afternoon and evening, rounds of heavy rain and strong wind gusts will continue from Maryland to southern New York and begin stretching into southern New England.",
             "Heavy rain and strong wind gusts will continue Saturday afternoon and evening.", "Saturday afternoon and evening"),
            ("The panel may release its findings Saturday night.", "The panel may release its findings Saturday night.", "Saturday night"),
        ]
        self.api.side_effect = None
        for source, signal, timing in cases:
            with self.subTest(source=source):
                row = {**self.row, "description": source, "grounded_sentence": source,
                       "published_at": "2026-09-26T15:50:15Z"}
                evidence = review_signals._model_evidence_candidates(self.day, [row])
                self.assertEqual([source], [r["grounded_sentence"] for r in evidence])
                self.assertTrue(all(k.startswith("window:2026-09-26:") for k in evidence[0]["grounded_time_keys"]))
                comparison = review_signals._checkpoint_timing_comparison(self.day, evidence)
                self.assertIn('"' + timing + '" => 2026-09-26 if upcoming', comparison)
                self.api.return_value = response({**self.approval, "comparisons": [self.comparison(timing=timing)]})
                self.assertTrue(self.review({**self.proposal, "signal": signal}, evidence)["eligible"])
        self.assertEqual(len(cases), self.api.call_count)

    def test_same_day_exception_excludes_past_ambiguous_and_stale_schedules(self):
        texts = [
            "The panel met Friday evening.",
            "The panel was expected to meet Friday evening.",
            "The panel was previously scheduled to meet Friday night.",
            "The panel might have met Friday evening.",
            "The panel is scheduled for Friday evening, but the meeting was cancelled.",
            "The panel will report Friday.",  # No time of day or explicit later-today cue.
            "The panel met last Friday evening.",
        ]
        for text in texts:
            with self.subTest(text=text):
                row = {**self.row, "description": text}
                self.assertEqual([], review_signals._model_evidence_candidates(self.day, [row]))
        self.assertEqual([], review_signals._future_anchor_records("Friday, September 18, 2026", date(2026, 9, 25)))
        stale = {**self.row, "published_at": "2026-09-18", "description": "The panel will meet Friday evening.",
                 "grounded_sentence": "The panel will meet Friday evening."}
        self.assertEqual([], review_signals._model_evidence_candidates(self.day, [stale]))
        self.assertFalse(review_signals.validate_generated_signal_grounding(self.day, stale["description"], stale)[0])
        self.api.assert_not_called()

    def test_same_day_rewrites_cannot_move_the_window_or_remove_uncertainty(self):
        source = "Coastal flooding is possible during high tide later Friday evening."
        row = {**self.row, "description": source, "grounded_sentence": source}
        for signal in (
            "Coastal flooding is possible during high tide Friday morning.",
            "Coastal flooding is possible during high tide Friday night.",
            "Coastal flooding will occur during high tide later Friday evening.",
        ):
            with self.subTest(signal=signal):
                self.assertFalse(self.review({**self.proposal, "signal": signal}, [row])["eligible"])
        self.api.assert_not_called()

    def test_saved_issue_evidence_recovers_deadline_same_day_and_due_schedule(self):
        root = Path(__file__).resolve().parents[1]
        load = load_fixture
        saved = load("focused_review.json")["daily_signal_review_2026_09_25"]
        topics = load("topic_summaries_2026-09-25.json")
        clusters = load("grouped_articles_final_global_receipts_2026-09-25.json")
        pool = load("articles_with_bias_2026-09-25.json")
        cache = load("next_signal_sidecar_cache.json")
        def cached_only(url, unused):
            entry = cache[review_signals._canonical_url(url)]  # A miss fails; never fetch.
            return entry.get("text", ""), dict(entry), False
        changed, rebuilt = [], {}
        with patch.object(review_signals, "_fetch_article_text", side_effect=cached_only):
            for index, old in enumerate(saved["report"], 1):
                topic = next(t for t in topics if t["topic_title"] == old["topic_title"])
                cluster = clusters[old["cluster_index"]]
                local = review_signals.select_signal_evidence(cluster)
                additions, meta, dirty = review_signals.build_signal_sidecar(self.day, topic, cluster, local, pool, cache)
                rows = review_signals._model_evidence_candidates(self.day, local + additions, topic=topic, cluster=cluster)
                rebuilt[index] = rows
                self.assertFalse(dirty)
                self.assertEqual(0, meta["network_fetches"])
                self.assertLessEqual(len(rows), 8)
                if {r["grounded_sentence"] for r in rows} != {r["grounded_sentence"] for r in saved["model_evidence"].get(old["topic_title"], [])}:
                    changed.append(index)
        self.assertEqual([3, 8, 9], changed)
        self.assertTrue(any("until Oct. 5 to file" in r["grounded_sentence"] for r in rebuilt[3]))
        self.assertTrue(any("possible during high tide later Saturday evening" in r["grounded_sentence"] for r in rebuilt[8]))
        self.assertTrue(any("expected to make a sharp turn" in r["grounded_sentence"] for r in rebuilt[8]))
        self.assertTrue(any("November referendum" in r["grounded_sentence"] for r in rebuilt[7]))
        self.assertTrue(any("is due to attend a prayer vigil" in r["grounded_sentence"] for r in rebuilt[9]))
        self.api.assert_not_called()

    def test_calendar_comparison_leaves_uncertain_windows_unresolved(self):
        texts = [
            "The panel will report within 7 days if the application is accepted.",
            "The panel will meet next Monday.",
            "The panel will report in November.",
            "The panel will meet Friday.",  # Same day: do not roll ahead a week.
            "The panel met Tuesday, Sept. 22, 2026.",
        ]
        rows = [{**self.row, "grounded_sentence": text} for text in texts]
        comparison = review_signals._checkpoint_timing_comparison(self.day, rows)
        for text in ("within 7 days", "next Monday", "November"):
            self.assertIn('"' + text + '" => unresolved timing', comparison)
        self.assertNotIn("if upcoming", comparison)
        self.assertEqual(2, comparison.count("No comparable future calendar match"))

    def test_weekday_hints_do_not_redate_completed_or_stale_passages(self):
        rows = [
            {**self.row, "grounded_sentence": "Australia made a dramatic revelation on Wednesday during the United Nations General Assembly."},
            {**self.row, "grounded_sentence": "The panel will report Monday.", "published_at": "2026-09-20"},
            {**self.row, "grounded_sentence": "The panel met last Monday and will publish a report."},
        ]
        comparison = review_signals._checkpoint_timing_comparison(self.day, rows)
        self.assertNotIn("if upcoming", comparison)
        self.assertEqual(3, comparison.count("unresolved timing; check source"))

    def test_earlier_calendar_match_does_not_automatically_reject_a_later_event(self):
        # Calendar order alone cannot determine relevance. The model's judgment
        # remains necessary; this simulated approval does not establish accuracy.
        rows = [
            {**self.row, "grounded_sentence": "Our newsletter will arrive Saturday."},
            self.row,
        ]
        self.api.side_effect = None
        self.api.return_value = response({**self.approval, "comparisons": [
            self.comparison(1, "Saturday", False, True, event="Newsletter delivery", reason="Unrelated recurring newsletter."),
            self.comparison(2),
        ]})
        result = self.review({**self.proposal, "source_index": 2}, rows)
        self.api.assert_called_once()
        self.assertTrue(result["eligible"])
        self.assertEqual(2, result["source_index"])
        self.assertEqual(self.proposal["signal"], result["signal"])

    def test_cache_key_includes_reference_date_used_for_calendar_comparison(self):
        changed = {**self.row, "grounded_reference_date": "2026-09-29"}
        for stage in ("triage", "candidate_validation", "final"):
            with self.subTest(stage=stage):
                before = review_signals._cache_key(self.day, self.topic, [self.row], stage=stage)
                after = review_signals._cache_key(self.day, self.topic, [changed], stage=stage)
                self.assertNotEqual(before, after)

    def test_required_comparisons_reject_incomplete_or_fabricated_entries(self):
        rows = [self.row, {**self.row, "grounded_sentence": "The panel may publish findings Wednesday."}]
        valid = [self.comparison(), self.comparison(2, "Wednesday")]
        invalid = [None, [], valid[:1], [valid[0], valid[0]],
                   [valid[0], {**valid[1], "source_index": 3}],
                   [{**valid[0], "source_index": True}, valid[1]],
                   [{**valid[0], "qualifies": "true"}, valid[1]],
                   [valid[0], {**valid[1], "before_selected": 0}],
                   [valid[0], {**valid[1], "before_selected": "false"}],
                   [{**valid[0], "event": ""}, valid[1]],
                   [{**valid[0], "timing": ""}, valid[1]],
                   [{**valid[0], "timing": "Friday"}, valid[1]],
                   [{**valid[0], "reason": " "}, valid[1]],
                   [valid[0], {k: v for k, v in valid[1].items() if k != "before_selected"}]]
        self.api.side_effect = None
        for comparisons in invalid:
            with self.subTest(comparisons=comparisons):
                self.api.reset_mock()
                self.api.return_value = response({**self.approval, "comparisons": comparisons})
                with self.assertRaises(RuntimeError):
                    self.review(rows=rows)
                self.api.assert_called_once()  # No repair/retry call.

    def test_known_paid_storm_reviewer_still_requires_comparisons(self):
        path = "focused_review.json"
        saved = load_fixture(path)["storm_timing_comparison_review"]
        self.api.side_effect = None
        for function, args, call in (
            (review_signals.evaluate_signal_triage, (), saved["calls"][0]),
            (review_signals.validate_candidate_signal_semantics, (saved["proposal"],), saved["calls"][1]),
        ):
            self.api.reset_mock()
            self.api.return_value = response(json.loads(call["response"]), call["finish_reason"])
            if function == review_signals.evaluate_signal_triage:
                result = function(saved["date"], saved["topic"], saved["evidence"], *args)
                self.assertTrue(result["eligible"])
                self.assertNotIn("comparisons", result)
            else:
                with self.assertRaisesRegex(RuntimeError, "compare every source"):
                    function(saved["date"], saved["topic"], saved["evidence"], *args)
            self.api.assert_called_once()

    def test_only_review_order_flags_can_block_a_later_selection(self):
        later = {**self.row, "description": "The panel may publish findings Wednesday.",
                 "grounded_sentence": "The panel may publish findings Wednesday."}
        rows = [self.row, later]
        comparisons = [self.comparison(before=True, reason="Earlier direct step in the same process."),
                       self.comparison(2, "Wednesday")]
        proposal = {**self.proposal, "source_index": 2, "signal": later["description"], "comparisons": comparisons}
        self.api.side_effect = None
        for function, args, reply in (
            (review_signals.evaluate_signal_triage, (), proposal),
            (review_signals.evaluate_next_signal, (), proposal),
            (review_signals.validate_candidate_signal_semantics, (proposal,), {**self.approval, "comparisons": comparisons}),
        ):
            with self.subTest(stage=function.__name__):
                self.api.reset_mock()
                self.api.return_value = response(reply)
                result = function(self.day, self.topic, rows, *args)
                self.api.assert_called_once()
                if function != review_signals.validate_candidate_signal_semantics:
                    self.assertTrue(result["eligible"])  # Still only a proposal.
                    self.assertEqual(proposal["signal"], result["signal"])
                    self.assertNotIn("comparisons", result)
                    continue
                self.assertFalse(result["eligible"])
                self.assertEqual("", result["signal"])
                self.assertEqual("comparison_conflict", result["reason_code"])
                self.assertIn("source 1", result["reason"])
                self.assertEqual([comparisons[0], self.writer_assessments([comparisons[1]])[0]], result["comparisons"])
                self.assertEqual(proposal["signal"], result["proposed_signal"])

    def test_duplicate_or_uncertain_order_does_not_invent_an_earlier_event(self):
        self.api.side_effect = None
        for before in (False, None):
            with self.subTest(before=before):
                self.api.return_value = response({**self.approval, "comparisons": [
                    self.comparison(), self.comparison(2, before=before,
                                                      reason="Duplicate checkpoint or uncertain order."),
                ]})
                self.assertTrue(self.review(rows=[self.row, self.row])["eligible"])

    def test_non_signal_writer_decisions_need_no_comparisons_but_cannot_select_a_sentence(self):
        self.api.side_effect = None
        for function in (review_signals.evaluate_signal_triage, review_signals.evaluate_next_signal):
            payload = {"status": "NO_SIGNAL", "eligible": False, "signal": "", "source_index": None,
                       "reason": "Background only."}
            self.api.return_value = response(payload)
            result = function(self.day, self.topic, [self.row])
            self.assertFalse(result["eligible"])
            self.assertNotIn("comparisons", result)
            for changed in ({"source_index": 1}, {"signal": self.proposal["signal"]}):
                self.api.return_value = response({**payload, **changed})
                if function == review_signals.evaluate_signal_triage or "signal" in changed:
                    result = function(self.day, self.topic, [self.row])
                    self.assertFalse(result["eligible"])
                    self.assertEqual(result["signal"], "")
                    self.assertIsNone(result["source_index"])
                else:
                    with self.assertRaises(RuntimeError):
                        function(self.day, self.topic, [self.row])

    def test_writer_refusal_truncation_and_invalid_source_are_errors(self):
        self.api.side_effect = None
        refused = response(self.proposal)
        refused.choices[0].message["refusal"] = "Cannot complete this request."
        for function in (review_signals.evaluate_signal_triage, review_signals.evaluate_next_signal):
            for reply in (refused, response(self.proposal, "length"),
                          response({**self.proposal, "source_index": True}),
                          response({**self.proposal, "source_index": "1"}),
                          response({**self.proposal, "source_index": 9})):
                self.api.reset_mock()
                self.api.return_value = reply
                with self.assertRaises(RuntimeError):
                    function(self.day, self.topic, [self.row])
                self.api.assert_called_once()

    def test_response_contract_bounds_comparison_coverage_and_output(self):
        for stage, base in (("triage", 260), ("candidate_validation", 380), ("final", 220)):
            for count in (1, 4, 8):
                selected = count if stage == "candidate_validation" else None
                options = review_signals._comparison_request_options(stage, count, selected)
                schema = options["response_format"]["json_schema"]
                self.assertTrue(schema["strict"])
                body = schema["schema"]
                self.assertFalse(body["additionalProperties"])
                self.assertEqual(set(body["properties"]), set(body["required"]))
                if stage != "candidate_validation":
                    self.assertNotIn("signal", body["properties"])
                    self.assertEqual([None, *range(1, count + 1)], body["properties"]["source_index"]["enum"])
                    self.assertNotIn("comparisons", body["properties"])
                    self.assertNotIn("before_selected", body["properties"])
                    self.assertEqual(base, options["max_tokens"])
                    continue
                    self.assertEqual("calendar_check", next(iter(body["properties"])))
                comparisons = body["properties"]["comparisons"]
                self.assertEqual(count, comparisons["minItems"])
                self.assertEqual(count, comparisons["maxItems"])
                item = comparisons["items"]
                variants = item.get("anyOf", [item])
                covered = []
                for variant in variants:
                    self.assertFalse(variant["additionalProperties"])
                    self.assertEqual(set(variant["properties"]), set(variant["required"]))
                    indices = variant["properties"]["source_index"]["enum"]
                    covered.extend(indices)
                    needs_order = stage == "candidate_validation" and selected not in indices
                    self.assertEqual(needs_order, "before_selected" in variant["required"])
                    self.assertEqual(needs_order, "checkpoint_state" in variant["required"])
                    self.assertEqual(needs_order, "state_evidence_index" in variant["required"])
                    if needs_order:
                        names = list(variant["properties"])
                        self.assertLess(names.index("state_evidence_index"), names.index("checkpoint_state"))
                    if stage == "candidate_validation" and not needs_order:
                        self.assertEqual([selected], indices)
                self.assertEqual(list(range(1, count + 1)), sorted(covered))
                self.assertEqual(base + 90 * count + 120 * (count - 1), options["max_tokens"])
                self.assertLessEqual(options["max_tokens"], 1940)
                self.assertNotIn("signal", body["properties"])
                self.assertNotIn("source_index", body["properties"])
        for count in (0, 9):
            with self.assertRaises(ValueError):
                review_signals._comparison_request_options("triage", count)
        for selected in (None, True, 0, 3):
            with self.assertRaises(ValueError):
                review_signals._comparison_request_options("candidate_validation", 2, selected)

    def test_saved_writer_failures_stay_blocked_without_spending_a_review_call(self):
        path = "focused_review.json"
        saved = load_fixture(path)["deadline_same_day_issue_validation"]
        for index in (4, 6):
            case = saved["cases"][index]
            with self.subTest(title=case["title"]):
                result = review_signals.validate_candidate_signal_semantics(
                    saved["date"], case["topic"], case["evidence"], case["proposal"],
                )
                self.assertFalse(result["eligible"])
                self.assertFalse(result["review_model_called"])
                self.assertEqual(case["review"]["reason"], result["reason"])
        self.api.assert_not_called()

    def test_conditional_countdown_uses_source_even_when_condition_is_removed(self):
        for condition in ("if approval is granted", "unless the permit is refused",
                          "once approval arrives", "when the permit arrives",
                          "pending approval", "subject to approval",
                          "contingent upon approval", "provided that approval arrives"):
            source = "The port will reopen in 7 days " + condition + "."
            row = {**self.row, "description": source, "grounded_sentence": source}
            for signal in (source, "The port will reopen in 7 days."):
                with self.subTest(condition=condition, signal=signal):
                    result = self.review({**self.proposal, "signal": signal}, [row])
                    self.assertFalse(result["eligible"])
                    self.assertFalse(result["review_model_called"])
                    self.assertIn("start is not established", result["reason"])
        self.api.assert_not_called()

    def test_countdown_guard_preserves_calendar_conditions_and_unconditional_durations(self):
        cases = [
            "The port will reopen Monday if approval is granted.",
            "The port will reopen October 7 if approval is granted.",
            "The port may reopen tomorrow if approval is granted.",
            "The port will reopen in 7 days.",
            "Approval was granted Friday; the port will reopen within 7 days.",
        ]
        for text in cases:
            with self.subTest(text=text):
                row = {**self.row, "description": text, "grounded_sentence": text}
                self.assertEqual((True, ""), review_signals.validate_generated_signal_grounding(self.day, text, row))
        # A different dated event is not proof that a conditional duration started.
        text = "Officials met Monday. The port will reopen in 7 days if approval arrives."
        row = {**self.row, "description": text, "grounded_sentence": text}
        self.assertFalse(self.review({**self.proposal, "signal": "The port will reopen in 7 days."}, [row])["eligible"])
        self.api.assert_not_called()

    def test_saved_ten_story_replay_rejects_countdown_and_incomplete_reviews(self):
        path = "focused_review.json"
        saved = load_fixture(path)["simple_writer_saved_issue_final_validation"]
        changed = []
        kept = []
        self.api.side_effect = None
        for index, case in enumerate(saved["cases"]):
            proposal = case.get("proposal")
            if not proposal or not proposal.get("eligible"):
                continue
            # These are saved model verdicts, not new judgments or new API calls.
            self.api.return_value = response(case["review"])
            before_calls = self.api.call_count
            result = review_signals.validate_candidate_signal_semantics(
                saved["date"], case["topic"], case["evidence"], proposal,
            )
            if result["signal"] != case["signal"]:
                changed.append(index)
                if index == 0:
                    self.assertEqual(before_calls, self.api.call_count)
                    self.assertIn("start is not established", result["reason"])
                else:
                    self.assertEqual(before_calls + 1, self.api.call_count)
                    self.assertEqual("comparison_incomplete", result["reason_code"])
            else:
                kept.append(index)
        self.assertEqual([0, 7, 8], changed)
        self.assertEqual([2, 4, 6], kept)
        self.assertEqual(5, self.api.call_count)

    def test_saved_countdown_boundary_cases_match_expected_local_behavior(self):
        path = "focused_review.json"
        saved = load_fixture(path)["conditional_countdown_boundary_tests_2026_10_12"]
        for case in saved["cases"]:
            with self.subTest(case=case["case"]):
                row = {**self.row, "grounded_sentence": case["source"], "description": case["source"]}
                passed, _ = review_signals.validate_generated_signal_grounding(self.day, case["signal"], row)
                self.assertEqual(case["should_reach_semantic_review"], passed)
        self.api.assert_not_called()

    def test_countdown_dependencies_across_subjects_and_reference_dates(self):
        actions = ("The laboratory will publish results", "The company will launch its service",
                   "The factory will resume production", "The museum will open")
        for day in ("2026-10-12", "2027-01-01", "2030-12-31"):
            for number, action in enumerate(actions, start=2):
                signal = f"{action} within {number} days."
                for source in (f"{action} within {number} days after approval.",
                               f"{action} within {number} days following approval.",
                               f"Upon approval, {action[0].lower() + action[1:]} within {number} days."):
                    with self.subTest(day=day, source=source):
                        row = {"grounded_sentence": source, "description": source, "published_at": day}
                        passed, reason = review_signals.validate_generated_signal_grounding(day, signal, row)
                        self.assertFalse(passed)
                        self.assertIn("start is not established", reason)
                for source in (f"Officials asked if work was complete. {signal}",
                               f"Officials asked if work was complete; {signal}",
                               f"Approval was granted today. {signal}", signal):
                    with self.subTest(day=day, source=source):
                        row = {"grounded_sentence": source, "description": source, "published_at": day}
                        self.assertEqual((True, ""), review_signals.validate_generated_signal_grounding(day, signal, row))
        self.api.assert_not_called()

    def test_countdown_scope_keeps_explicit_cross_sentence_prerequisites(self):
        signal = "The port will reopen within 7 days."
        for source in (f"Approval is pending. {signal}",
                       f"{signal} This is subject to approval.",
                       f"The reopening depends on approval; {signal}",
                       f"If approval arrives. {signal}"):
            with self.subTest(source=source):
                row = {**self.row, "grounded_sentence": source, "description": source}
                self.assertFalse(self.review({**self.proposal, "signal": signal}, [row])["eligible"])
        self.api.assert_not_called()

    def test_old_countdown_approval_cache_cannot_bypass_new_guard(self):
        text = "The port will reopen in 7 days if approval arrives."
        self.row = {**self.row, "description": text, "grounded_sentence": text}
        self.proposal = {**self.proposal, "signal": "The port will reopen in 7 days."}
        context = review_signals._signal_review_context(self.proposal)
        with patch.object(review_signals, "VALIDATION_PROMPT_VERSION", "next-signal-validation-v12-alternative-order-2026-09-29"):
            old_key = review_signals._cache_key(self.day, self.topic, [self.row], stage="candidate_validation", context=context)
        new_key = review_signals._cache_key(self.day, self.topic, [self.row], stage="candidate_validation", context=context)
        self.assertNotEqual(old_key, new_key)
        with self.pipeline() as (files, _, triage), patch.object(
            review_signals, "_model_evidence_candidates", return_value=[self.row],
        ):
            old_approval = {"eligible": True, "signal": self.proposal["signal"], "source_index": 1}
            files["cache"] = {old_key: deepcopy(old_approval)}
            review_signals.main()
            self.assertEqual("", files["summary"][0]["next_signal"])
            self.assertFalse(files["cache"][new_key]["eligible"])
            self.assertEqual(old_approval, files["cache"][old_key])
        self.api.assert_not_called()

    def test_saved_missouri_copy_stays_blocked_and_legacy_wrong_quotes_still_fail(self):
        path = "focused_review.json"
        saved = load_fixture(path)["source_wording_alternative_order_review"]
        case = saved["cases"][1]
        raw = json.loads(saved["calls"][2]["response"])
        self.api.side_effect = None
        for writer in (review_signals.evaluate_signal_triage, review_signals.evaluate_next_signal):
            self.api.return_value = response({**raw, "eligible": True})
            proposal = writer(saved["date"], case["topic"], case["evidence"])
            self.assertTrue(proposal["eligible"])
            self.assertEqual(case["evidence"][2]["grounded_sentence"], proposal["signal"])
            self.assertEqual(3, proposal["source_index"])
            self.assertNotIn("comparisons", proposal)
            # The complete sentence contains a completed court action. Unlike
            # the old paraphrase, it fails the unchanged grounding gate locally.
            self.api.reset_mock()
            blocked = review_signals.validate_candidate_signal_semantics(
                saved["date"], case["topic"], case["evidence"], proposal)
            self.assertFalse(blocked["eligible"])
            self.api.assert_not_called()
            # The writer's old audit is ignored. The independent reviewer still
            # must reject incorrect quotes when replaying the old paraphrase.
            self.api.return_value = response({"eligible": True, "comparisons": [
                {**item, "before_selected": False} for item in raw["comparisons"]
            ], "reason": "Simulated review with incorrect source quotes."})
            with self.assertRaisesRegex(RuntimeError, "timing is not quoted"):
                review_signals.validate_candidate_signal_semantics(
                    saved["date"], case["topic"], case["evidence"], {**proposal, "signal": raw["signal"]},
                )
            prompt = self.api.call_args.kwargs["messages"][1]["content"]
            for row in case["evidence"]:
                self.assertIn(row["grounded_sentence"], prompt)

    def test_saved_storm_self_order_is_ignored_but_qualification_still_controls(self):
        path = "focused_review.json"
        saved = load_fixture(path)["deadline_same_day_issue_validation"]
        case = saved["cases"][7]
        raw = json.loads(saved["calls"][7]["response"])
        self.assertIsNone(raw["comparisons"][0]["before_selected"])
        self.api.side_effect = None
        self.api.return_value = response(raw)
        result = review_signals.validate_candidate_signal_semantics(
            saved["date"], case["topic"], case["evidence"], case["proposal"],
        )
        self.assertFalse(result["eligible"])
        self.assertEqual("comparison_incomplete", result["reason_code"])
        # The refund can be excluded, but its explicit timing cannot be omitted.
        raw["comparisons"][3].update(event="Festival ticket refunds", timing="within 30 days",
                                     reason="Refunds do not advance the storm's track.")
        self.api.return_value = response(raw)
        result = review_signals.validate_candidate_signal_semantics(
            saved["date"], case["topic"], case["evidence"], case["proposal"],
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(case["proposal"]["signal"], result["signal"])
        self.assertNotIn("before_selected", result["comparisons"][0])
        self.assertEqual(raw["comparisons"][1:], result["comparisons"][1:])
        # Replaying a saved verdict checks the parser, not better model reasoning.
        # Even with no self-order field, an unqualified selection must fail closed.
        changed = deepcopy(raw)
        changed["comparisons"][0].pop("before_selected")
        changed["comparisons"][0]["qualifies"] = False
        self.api.return_value = response(changed)
        result = review_signals.validate_candidate_signal_semantics(
            saved["date"], case["topic"], case["evidence"], case["proposal"],
        )
        self.assertFalse(result["eligible"])
        self.assertEqual("comparison_conflict", result["reason_code"])

    @contextlib.contextmanager
    def pipeline(self, external=False):
        # Main runs against an in-memory filesystem. No output files are created.
        files = {"summary": [{**self.topic, "next_signal": "previous", "untouched": {"keep": True}}], "clusters": [{}]}
        writes = []
        def save(path, value):
            files[path] = deepcopy(value)
            writes.append(path)
        argv = ["review-draft", "--date", self.day, "--summary-file", "summary",
                "--cluster-file", "clusters", "--article-pool-file", "pool",
                "--report-file", "report", "--cache-file", "cache", "--sidecar-cache-file", "sidecar"]
        if not external:
            argv += ["--no-sidecar", "--no-external"]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(review_signals, "_load_json", side_effect=lambda p, d: deepcopy(files.get(p, d))))
            stack.enter_context(patch.object(review_signals, "_save_json_atomic", side_effect=save))
            stack.enter_context(patch.object(review_signals, "match_topic_to_cluster", return_value=(0, {"url_overlap": 1})))
            stack.enter_context(patch.object(review_signals, "select_signal_evidence", return_value=[self.row]))
            stack.enter_context(patch.object(review_signals, "_rank_sidecar_candidates", return_value=[]))
            triage = stack.enter_context(patch.object(review_signals, "evaluate_signal_triage", return_value=self.proposal))
            yield files, writes, triage

    def test_pipeline_preserves_proposal_and_cache_is_specific_to_it(self):
        self.api.side_effect = None
        self.api.return_value = response({**self.approval, "signal": "Unwanted rewrite.", "source_index": 99})
        with self.pipeline() as (files, _, triage):
            before = deepcopy(files["summary"][0])
            review_signals.main()
            self.assertEqual({**before, "next_signal": self.proposal["signal"]}, files["summary"][0])
            self.assertEqual(2, files["report"][0]["model_call_count"])
            review_signals.main()
            self.assertEqual(0, files["report"][0]["model_call_count"])
            self.assertEqual(1, self.api.call_count)
            self.assertEqual(1, triage.call_count)
            triage_key = next(k for k in files["cache"] if ":triage:" in k)
            changed = "The panel may release findings Monday."
            files["cache"][triage_key]["signal"] = changed
            review_signals.main()
            self.assertEqual(0, files["report"][0]["model_call_count"])
            self.assertEqual(1, self.api.call_count)
            self.assertEqual("", files["summary"][0]["next_signal"])
            self.assertEqual("source_copy", files["report"][0]["candidate_validation_decision"]["reason_code"])

    def test_cached_approval_cannot_bypass_conservative_timing_at_publication(self):
        later = {**self.row, "description": "The panel will release final findings Wednesday.",
                 "grounded_sentence": "The panel will release final findings Wednesday."}
        rows = [self.row, later]
        proposal = {**self.proposal, "source_index": 2, "signal": later["grounded_sentence"]}
        for version in (review_signals.VALIDATION_PROMPT_VERSION, "next-signal-validation-v20-source-copy-2026-10-27"):
            with self.subTest(version=version), self.pipeline() as (files, _, triage), patch.object(
                review_signals, "_model_evidence_candidates", return_value=rows,
            ):
                triage_key = review_signals._cache_key(self.day, self.topic, rows, stage="triage")
                with patch.object(review_signals, "VALIDATION_PROMPT_VERSION", version):
                    approval_key = review_signals._cache_key(self.day, self.topic, rows, stage="candidate_validation",
                                                             context=review_signals._signal_review_context(proposal))
                cached = {"eligible": True, "signal": proposal["signal"], "source_index": 2}
                files["cache"] = {triage_key: deepcopy(proposal), approval_key: deepcopy(cached)}
                review_signals.main()
                self.assertEqual("", files["summary"][0]["next_signal"])
                self.assertIn("earlier or unresolved timing", files["report"][0]["reason"])
                self.assertEqual(0, files["report"][0]["model_call_count"])
                self.assertEqual(cached, files["cache"][approval_key])
                triage.assert_not_called()
        self.api.assert_not_called()

    def test_source_copy_rejects_added_claims_fragments_and_pronouns(self):
        examples = [
            ("The company will report earnings Tuesday.", "The company will report earnings Tuesday, potentially updating the pledge."),
            ("If the permit is approved, construction will start Monday.", "Construction will start Monday."),
            ("The council may vote Monday.", "The council will vote Monday."),
            ("It will reopen Thursday.", "It will reopen Thursday."),
            ("Both will reopen Thursday.", "Both will reopen Thursday."),
            ("Next year I will distribute shares.", "Next year I will distribute shares."),
            ("The bridge and trail are closed. It will reopen Thursday.", "It will reopen Thursday."),
            ("The council will vote Monday if", "The council will vote Monday if"),
        ]
        for source, signal in examples:
            with self.subTest(signal=signal):
                row = {"grounded_sentence": source, "published_at": self.day}
                result = review_signals.review_signal_for_publication(self.day, self.topic, [row],
                    {"eligible": True, "signal": signal, "source_index": 1})
                self.assertFalse(result["eligible"])
                self.assertEqual("", result["signal"])
                self.assertFalse(result["review_model_called"])
        self.api.assert_not_called()

    def test_source_copy_withholds_ellipses_before_paid_review(self):
        for text in ["Koenig died Monday at...",
                     "The court will hear the appeal Friday at...",
                     "The court will hear the appeal Friday…",
                     'The court will hear the appeal Friday at..."',
                     "The court will hear the appeal . . . Friday.",
                     "The court will hear the appeal … Friday."]:
            with self.subTest(text=text):
                result = review_signals.review_signal_for_publication(
                    self.day, self.topic, [{"description": text, "published_at": self.day}],
                    {"eligible": True, "signal": text, "source_index": 1})
                self.assertFalse(result["eligible"])
                self.assertFalse(result["review_model_called"])
                self.assertIn("ellipsis", result["reason"])
        self.api.assert_not_called()
        # Ordinary sentence punctuation and date abbreviations remain usable.
        text = "The court will hear the appeal on Oct. 30."
        self.assertEqual((True, ""), review_signals.validate_signal_source_copy(text, {"description": text}))

    def test_source_copy_preserves_whole_sentence_and_qualifications(self):
        for text in ["The panel may release its findings Monday.",
                     "If the permit is approved, construction will start Monday.",
                     "The company pledges to distribute shares next year."]:
            self.assertEqual((True, ""), review_signals.validate_signal_source_copy(text, {"description": text}))
        long = "The board will meet Monday " + "publicly " * 40 + "."
        self.assertFalse(review_signals.validate_signal_source_copy(long, {"description": long})[0])

    def test_cached_approval_cannot_bypass_source_copy_publication_check(self):
        self.api.side_effect = None
        self.api.return_value = response(self.approval)
        with self.pipeline() as (files, _, triage):
            review_signals.main()
            rows = review_signals._model_evidence_candidates(self.day, [self.row], topic=self.topic, cluster={})
            changed = {**self.proposal, "signal": "The panel may release findings Monday."}
            triage_key = next(k for k in files["cache"] if ":triage:" in k)
            files["cache"][triage_key] = changed
            key = review_signals._cache_key(self.day, self.topic, rows, "candidate_validation",
                                           review_signals._signal_review_context(changed))
            files["cache"][key] = {"eligible": True, "signal": changed["signal"], "source_index": 1}
            review_signals.main()
            self.assertEqual(0, files["report"][0]["model_call_count"])
            self.assertEqual("", files["summary"][0]["next_signal"])
            self.assertIn("copy a complete source sentence", files["report"][0]["reason"])
        self.api.assert_called_once()

    def test_source_copy_does_not_replace_semantic_review(self):
        self.api.side_effect = None
        self.api.return_value = response({"eligible": False, "reason": "Unrelated process.",
            "comparisons": [self.comparison(qualifies=False)]})
        result = review_signals.review_signal_for_publication(self.day, self.topic, [self.row], self.proposal)
        self.assertFalse(result["eligible"])
        self.assertTrue(result["review_model_called"])
        self.api.assert_called_once()

    def test_source_copy_saved_longer_sentence_and_known_omission(self):
        cases = load_fixture("focused_review.json")["simple_writer_saved_issue_final_validation"]["cases"]
        storm = cases[7]["evidence"][0]
        self.assertGreater(len(storm["grounded_sentence"].split()), 22)
        self.assertTrue(review_signals.validate_signal_source_copy(storm["grounded_sentence"], storm)[0])
        # The filing deadline is valid with context, but this conservative mode
        # deliberately omits the sentence's "both"/"he" references.
        court = cases[2]["evidence"][0]
        self.assertFalse(review_signals.validate_signal_source_copy(court["grounded_sentence"], court)[0])

    def test_evidence_merge_includes_research_when_local_slots_are_full(self):
        local = [{"grounded_sentence": f"Local event {i} Monday."} for i in range(8)]
        external = [{"grounded_sentence": "New research event Tuesday."}]
        before = deepcopy((local, external))
        result = review_signals._merge_model_evidence(local, external)
        self.assertEqual(8, len(result))
        self.assertIn(external[0], result)
        self.assertEqual(local[0], result[0])
        self.assertEqual(before, (local, external))
        self.assertEqual([], review_signals._merge_model_evidence(local, external, limit=0))

    def test_evidence_merge_preserves_long_distinctions_and_richer_context(self):
        prefix = "An extended description of the proceedings. " * 8
        a = {"grounded_sentence": prefix + "Hearing Monday.", "description": "short"}
        b = {"grounded_sentence": prefix + "Hearing Tuesday."}
        richer = {**a, "description": "Full identifying context for this proceeding."}
        result = review_signals._merge_model_evidence([a], [b, richer])
        self.assertEqual([richer, b], result)

    def test_external_pipeline_keeps_new_evidence_at_capacity(self):
        local = [{**self.row, "description": f"Panel {i} will meet Monday.",
                  "grounded_sentence": f"Panel {i} will meet Monday."} for i in range(8)]
        external = {**self.row, "description": "The panel will publish findings Wednesday.",
                    "grounded_sentence": "The panel will publish findings Wednesday."}
        with self.pipeline(external=True) as (files, _, triage), \
                patch.object(review_signals, "select_signal_evidence", return_value=local), \
                patch.object(review_signals, "build_signal_sidecar", return_value=([], {}, False)), \
                patch.object(review_signals, "build_external_signal_evidence", return_value=([external], {}, False)), \
                patch.object(review_signals, "evaluate_next_signal", return_value={"eligible": False, "signal": ""}) as final:
            triage.return_value = {"status": "RESEARCH_NEEDED", "eligible": False, "search_queries": ["panel findings"]}
            review_signals.main()
            rows = final.call_args.args[2]
            self.assertEqual(8, len(rows))
            self.assertIn(external["grounded_sentence"], [r["grounded_sentence"] for r in rows])
            self.assertEqual("", files["summary"][0]["next_signal"])
        self.api.assert_not_called()

    def test_pipeline_counts_no_call_for_structural_rejection(self):
        with self.pipeline() as (files, _, triage):
            triage.return_value = {**self.proposal, "signal": "The panel released its findings Monday."}
            review_signals.main()
            self.assertEqual(1, files["report"][0]["model_call_count"])
            self.assertEqual("", files["summary"][0]["next_signal"])
        self.api.assert_not_called()

    def test_reviewer_unqualified_selection_blocks_publication_and_caches_the_audit(self):
        generate = review_signals.evaluate_signal_triage
        later = {**self.row, "description": "The panel may release its findings Wednesday.",
                 "grounded_sentence": "The panel may release its findings Wednesday."}
        comparisons = [
            self.comparison(qualifies=False), self.comparison(2, "Wednesday", before=False),
        ]
        self.api.side_effect = [
            response({"status": "SIGNAL", "source_index": 1}),
            response({"eligible": True, "comparisons": comparisons}),
        ]
        with self.pipeline() as (files, _, triage), \
                patch.object(review_signals, "select_signal_evidence", return_value=[self.row, later]):
            triage.side_effect = generate
            review_signals.main()
            self.assertEqual(2, self.api.call_count)
            self.assertEqual("", files["summary"][0]["next_signal"])
            audit = files["report"][0]
            self.assertEqual(2, audit["model_call_count"])
            self.assertNotIn("comparisons", audit["triage_decision"])
            self.assertEqual("comparison_conflict", audit["candidate_validation_decision"]["reason_code"])
            self.assertEqual(2, len(files["cache"]))
            review_signals.main()
            self.assertEqual(2, self.api.call_count)
            self.assertEqual(0, files["report"][0]["model_call_count"])

    def test_review_failure_preserves_outputs_and_completed_cache_for_retry(self):
        self.api.side_effect = RuntimeError("temporary outage")
        with self.pipeline() as (files, writes, triage):
            before = deepcopy(files["summary"])
            with self.assertRaisesRegex(RuntimeError, "outputs were not replaced"):
                review_signals.main()
            self.assertEqual(before, files["summary"])
            self.assertEqual(["cache"], writes)
            self.assertEqual(1, len(files["cache"]))
            self.api.side_effect = None
            self.api.return_value = response(self.approval)
            review_signals.main()
            self.assertEqual(1, triage.call_count)
            self.assertEqual(1, files["report"][0]["model_call_count"])

    def test_incomplete_first_pass_preserves_existing_outputs_and_cache(self):
        generate = review_signals.evaluate_signal_triage
        self.api.side_effect = None
        self.api.return_value = response({k: v for k, v in self.proposal.items() if k != "comparisons"}, "length")
        with self.pipeline() as (files, writes, triage):
            triage.side_effect = generate
            files["cache"] = {"existing": {"eligible": False}}
            files["report"] = ["existing report"]
            before = deepcopy(files)
            with self.assertRaisesRegex(RuntimeError, "triage failed; newsletter outputs were not replaced"):
                review_signals.main()
            self.assertEqual(before, files)
            self.assertEqual([], writes)
            self.api.assert_called_once()

    def test_incomplete_external_generation_preserves_completed_triage(self):
        external = {**self.row, "description": "The panel will issue findings Wednesday.",
                    "grounded_sentence": "The panel will issue findings Wednesday."}
        self.api.side_effect = None
        self.api.return_value = response({"eligible": True, "signal": external["description"], "source_index": 2}, "length")
        with self.pipeline(external=True) as (files, writes, triage), \
                patch.object(review_signals, "build_signal_sidecar", return_value=([], {}, False)), \
                patch.object(review_signals, "build_external_signal_evidence", return_value=([external], {}, False)):
            triage.return_value = {"status": "RESEARCH_NEEDED", "eligible": False, "search_queries": ["panel findings schedule"]}
            before = deepcopy(files["summary"])
            with self.assertRaisesRegex(RuntimeError, "final failed; newsletter outputs were not replaced"):
                review_signals.main()
            self.assertEqual(before, files["summary"])
            self.assertEqual(["cache"], writes)
            self.assertEqual(1, len(files["cache"]))
            self.assertIn(":triage:", next(iter(files["cache"])))
            self.api.assert_called_once()

    def test_external_path_reviews_exact_external_proposal(self):
        external = {**self.row, "description": "The panel will issue findings Sunday.",
                    "grounded_sentence": "The panel will issue findings Sunday."}
        candidate = {"eligible": True, "signal": external["description"], "source_index": 2}
        self.api.side_effect = None
        self.api.return_value = response({**self.approval, "signal": "Unwanted rewrite.", "source_index": 1,
                                         "comparisons": [self.comparison(1, qualifies=False, before=False,
                                                                         reason="Local passage insufficient for this story."),
                                                         self.comparison(2, "Sunday")]})
        with self.pipeline(external=True) as (files, _, triage), \
                patch.object(review_signals, "build_signal_sidecar", return_value=([], {}, False)), \
                patch.object(review_signals, "build_external_signal_evidence", return_value=([external], {}, False)), \
                patch.object(review_signals, "evaluate_next_signal", return_value=candidate):
            triage.return_value = {"status": "RESEARCH_NEEDED", "eligible": False, "search_queries": ["panel findings schedule"]}
            review_signals.main()
            self.assertEqual(candidate["signal"], files["summary"][0]["next_signal"])
            self.assertEqual(3, files["report"][0]["model_call_count"])
            self.assertEqual("external_enrichment", files["report"][0]["signal_stage"])
            self.assertEqual(external["description"], files["report"][0]["supporting_source"]["grounded_sentence"])
            prompt = self.api.call_args.kwargs["messages"][1]["content"]
            self.assertIn(review_signals._signal_review_context(candidate), prompt)


class DailyPipelineSignalChecks(unittest.TestCase):
    """Exercise orchestration without executing stages, requests, or live writes."""

    def check_path(self, *, global_enabled=False, global_success=False,
                   expanded_age=None, signal_exit=0, dry_run=False, summary_exit=0, global_fail_stage=None,
                   signal_budget="0.25", signal_max_calls=40):
        from types import SimpleNamespace
        import os
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            day = "2030-12-31"
            final = root / f"grouped_articles_final_{day}.json"
            final.write_text("[]")
            (root / f"grouped_articles_filtered_{day}.json").write_text("[]")
            (root / f"gdelt_discovery_candidates_{day}.json").write_text("{}")
            os.utime(final, (200, 200))
            if expanded_age is not None:
                expanded = root / f"grouped_articles_final_expanded_{day}.json"
                expanded.write_text("[]")
                os.utime(expanded, (expanded_age, expanded_age))
            calls = []
            def required(**kwargs):
                calls.append(kwargs)
                if kwargs["script"] == "generate_next_signals.py":
                    return signal_exit
                if kwargs["script"] == "summarize_grouped_topics.py":
                    return summary_exit
                return 0
            def optional(**kwargs):
                calls.append(kwargs)
                if kwargs["script"] == "audit_gdelt_global_discovery.py" and global_success:
                    (root / f"gdelt_discovery_candidates_{day}.json").write_text("{}")
                return global_success and kwargs["script"] != global_fail_stage
            args = SimpleNamespace(date=day, with_gdelt_discovery=global_enabled,
                                   gdelt_discovery_cached_only=False, dry_run=dry_run,
                                   resume_story_review=False, story_review_budget_usd="0.90",
                                   next_signal_budget_usd=signal_budget, next_signal_max_calls=signal_max_calls)
            with patch.object(daily_pipeline, "__file__", str(root / "run_pipeline_by_date.py")), \
                 patch.object(daily_pipeline, "parse_args", return_value=args), \
                 patch.object(daily_pipeline, "run_required", side_effect=required), \
                 patch.object(daily_pipeline, "run_optional", side_effect=optional), \
                 patch.object(daily_pipeline, "manifest_is_usable", return_value=True), \
                 patch.object(daily_pipeline, "outputs_exist", return_value=True), \
                 patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = daily_pipeline.run()
            signal_calls = [c for c in calls if c["script"] == "generate_next_signals.py"]
            if summary_exit:
                self.assertEqual(summary_exit, result)
                self.assertEqual([], signal_calls)
                self.assertEqual(1, len([c for c in calls if c["script"] == "summarize_grouped_topics.py"]))
                return
            self.assertEqual(signal_exit, result)
            self.assertEqual(1, len(signal_calls))
            call = signal_calls[0]
            self.assertEqual(call, calls[-1])
            self.assertEqual("summarize_grouped_topics.py", calls[-2]["script"])
            suffix = "global_receipts" if global_success and not global_fail_stage else (
                "expanded" if expanded_age is not None and expanded_age >= 200 else None)
            expected = f"grouped_articles_final_{suffix}_{day}.json" if suffix else final.name
            self.assertEqual(["--summary-file", f"topic_summaries_{day}.json",
                              "--cluster-file", expected,
                              "--checkpoint-review", "--checkpoint-live",
                              "--checkpoint-budget-usd", signal_budget,
                              "--checkpoint-max-calls", str(signal_max_calls),
                              "--checkpoint-max-output-tokens", "2400",
                              "--checkpoint-cache-file", f"outputs/checkpoint_receipts_{day}.json",
                              "--checkpoint-ledger-file", f"outputs/checkpoint_usage_{day}.json",
                              "--no-external"], call["extra_args"])
            self.assertEqual(dry_run, call["dry_run"])

    def test_baseline_selects_same_fresh_input_as_summarizer(self):
        for age in (None, 100, 200, 300):
            with self.subTest(expanded_age=age):
                self.check_path(expanded_age=age)

    def test_global_summary_failure_does_not_repeat_paid_stage_or_run_signals(self):
        self.check_path(global_enabled=True, global_success=True, summary_exit=7)

    def test_global_receipts_and_discovery_fallback(self):
        self.check_path(global_enabled=True, global_success=True)
        self.check_path(global_enabled=True, global_success=False, expanded_age=300)
        self.check_path(global_enabled=True, global_success=True,
                        global_fail_stage="attach_gdelt_global_receipts.py", expanded_age=300)

    def test_signal_failure_stops_success_and_summary_failure_skips_signal(self):
        self.check_path(signal_exit=7)
        self.check_path(summary_exit=3)

    def test_dry_run_propagates_to_signal(self):
        self.check_path(dry_run=True)
        self.check_path(global_enabled=True, global_success=True, dry_run=True)

    def test_custom_checkpoint_limits_reach_baseline_and_global_stage(self):
        self.check_path(signal_budget="0.06", signal_max_calls=3)
        self.check_path(global_enabled=True, global_success=True,
                        signal_budget="0.06", signal_max_calls=3)

    def test_invalid_limits_stop_before_any_pipeline_stage(self):
        for flag, value in [("--next-signal-budget-usd", v) for v in
                            ("0", "-0.01", "NaN", "Infinity", "-Infinity", "invalid")] + [
                            ("--next-signal-max-calls", v) for v in ("0", "-1", "1.5")]:
            with self.subTest(flag=flag, value=value), \
                 patch.object(sys, "argv", ["runner", "--date", "2030-12-31", flag, value]), \
                 patch.object(daily_pipeline.subprocess, "run") as child, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
                daily_pipeline.run()
            self.assertEqual(2, stop.exception.code)
            child.assert_not_called()

    def test_real_daily_dry_run_selects_new_route_without_executing_stages(self):
        for mode in ([], ["--no-gdelt-discovery"]):
            output = io.StringIO()
            with self.subTest(mode=mode), \
                 patch.object(sys, "argv", ["runner", "--date", "2030-12-31", "--dry-run", *mode]), \
                 patch.object(daily_pipeline.subprocess, "run", side_effect=AssertionError("No subprocess")) as child, \
                 patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(0, daily_pipeline.run())
            child.assert_not_called()
            commands = [line for line in output.getvalue().splitlines()
                        if "DRY RUN:" in line and "generate_next_signals.py" in line]
            self.assertEqual(1, len(commands))
            self.assertIn("--checkpoint-review --checkpoint-live", commands[0])
            self.assertIn("--checkpoint-budget-usd 0.25", commands[0])
            self.assertIn("--checkpoint-max-calls 40", commands[0])
            self.assertIn("--checkpoint-max-output-tokens 2400", commands[0])
            self.assertIn("--no-external", commands[0])


if __name__ == "__main__":
    unittest.main()


class SourceTimingReferenceTests(unittest.TestCase):
    def test_timing_choices_are_bound_to_their_own_source(self):
        rows = [{"grounded_sentence": "Officials will meet next week."},
                {"grounded_sentence": "Negotiations will take place on Nov. 10-11."}]
        quotes = review_signals._source_timing_quotes(rows[1])
        self.assertNotIn("next week", quotes)
        idx = next(i for i, q in enumerate(quotes, 1) if "Nov." in q)
        items = [{"source_index": 1, "event": "Officials will meet next week.",
                  "timing": "next week", "qualifies": True, "reason": "Scheduled"},
                 {"source_index": 2, "event": rows[1]["grounded_sentence"],
                  "timing_index": idx, "qualifies": True, "reason": "Scheduled"}]
        result = review_signals._read_source_comparisons({"comparisons": items}, rows, 1)
        self.assertEqual(quotes[idx-1], result[1]["timing"])
        for invalid in (True, 0, -1, len(quotes)+1, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                review_signals._read_source_comparisons(
                    {"comparisons": [items[0], {**items[1], "timing_index": invalid}]}, rows, 1)
        with self.assertRaises(RuntimeError):
            review_signals._read_source_comparisons(
                {"comparisons": [items[0], {**items[1], "timing": "next week"}]}, rows, 1)

    def test_schema_requires_source_local_timing_ids(self):
        rows = [{"grounded_sentence": "The board will decide on Friday."},
                {"grounded_sentence": "The committee will meet tomorrow."}]
        schema = review_signals._comparison_request_options(
            "candidate_validation", 2, 1, evidence=rows)["response_format"]["json_schema"]["schema"]
        for variant in schema["properties"]["comparisons"]["items"]["anyOf"]:
            fields = variant["properties"]
            index = fields["source_index"]["enum"][0]
            self.assertNotIn("timing", fields)
            self.assertEqual([None, *range(1, len(review_signals._source_timing_quotes(rows[index-1]))+1)],
                             fields["timing_index"]["enum"])

    def test_earlier_step_remains_blocked_despite_model_relevance_exclusion(self):
        rows = [{"grounded_sentence": "The delegations will negotiate next week.", "published_at": "2030-06-03"},
                {"grounded_sentence": "Ahead of next week's negotiations, advisers will meet this week.", "published_at": "2030-06-03"}]
        comparisons = [{"source_index":1,"qualifies":True},
                      {"source_index":2,"qualifies":False,"checkpoint_state":"FUTURE"}]
        self.assertTrue(review_signals._unresolved_future_alternative("2030-06-03",rows,1,comparisons))


class BoundEventTimingTests(unittest.TestCase):
    def test_other_clause_time_cannot_date_the_quoted_event(self):
        row = {"grounded_sentence": "Ahead of next week's negotiations, advisers will meet this week."}
        item = {"source_index": 1, "event": "advisers will meet this week",
                "timing": "next week", "timing_index": 1}
        self.assertTrue(review_signals._comparison_event_binding_gap([row], [item]))
        item["timing"] = "this week"
        self.assertEqual("", review_signals._comparison_event_binding_gap([row], [item]))

    def test_foreign_event_quote_rejected_even_with_correct_source_date(self):
        row = {"grounded_sentence": "The board will vote on Friday."}
        item = {"source_index": 1, "event": "The court will rule on Friday",
                "timing": "Friday", "timing_index": 1}
        self.assertTrue(review_signals._comparison_event_binding_gap([row], [item]))
        item["event"] = "The board will vote on Friday"
        self.assertEqual("", review_signals._comparison_event_binding_gap([row], [item]))
