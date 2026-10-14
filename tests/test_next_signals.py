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
from datetime import date


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
    return SimpleNamespace(choices=[Choice(message={"content": json.dumps(payload)}, finish_reason=finish_reason)])


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
        self.assertEqual(review_signals.MODEL, self.api.call_args.kwargs["model"])
        self.assertEqual(330, self.api.call_args.kwargs["max_tokens"])

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

    def test_saved_storm_reply_reaches_review_without_bypassing_its_verdict(self):
        path = "focused_review.json"
        saved = load_fixture(path)["required_comparison_storm_review"]
        call = saved["calls"][0]
        raw = json.loads(call["response"])
        self.api.side_effect = None
        self.api.return_value = response(raw, call["finish_reason"])
        proposal = review_signals.evaluate_signal_triage(saved["date"], saved["topic"], saved["evidence"])
        self.assertTrue(proposal["eligible"])
        self.assertEqual(raw["signal"], proposal["signal"])
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
        self.assertEqual(raw["signal"], result["proposed_signal"])
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
            (review_signals.validate_candidate_signal_semantics, (proposal,), {**self.approval, "comparisons": comparisons}, 240, 90),
        ):
            with self.subTest(stage=function.__name__):
                self.api.reset_mock()
                self.api.return_value = response(reply)
                result = function(self.day, topic, rows, *args)
                self.api.assert_called_once()
                request = self.api.call_args.kwargs
                self.assertIn(comparison, request["messages"][1]["content"])
                self.assertEqual(review_signals.MODEL, request["model"])
                self.assertEqual(cap + per_source * len(rows), request["max_tokens"])
                self.assertTrue(request["response_format"]["json_schema"]["strict"])
                self.assertEqual(signal, result["signal"])
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
        for stage, base in (("triage", 260), ("candidate_validation", 240), ("final", 220)):
            for count in (1, 4, 8):
                selected = count if stage == "candidate_validation" else None
                options = review_signals._comparison_request_options(stage, count, selected)
                schema = options["response_format"]["json_schema"]
                self.assertTrue(schema["strict"])
                body = schema["schema"]
                self.assertFalse(body["additionalProperties"])
                self.assertEqual(set(body["properties"]), set(body["required"]))
                if stage != "candidate_validation":
                    self.assertNotIn("comparisons", body["properties"])
                    self.assertNotIn("before_selected", body["properties"])
                    self.assertEqual(base, options["max_tokens"])
                    continue
                self.assertEqual("comparisons", next(iter(body["properties"])))
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
                    if stage == "candidate_validation" and not needs_order:
                        self.assertEqual([selected], indices)
                self.assertEqual(list(range(1, count + 1)), sorted(covered))
                self.assertEqual(base + 90 * count, options["max_tokens"])
                self.assertLessEqual(options["max_tokens"], 960)
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

    def test_saved_ten_story_replay_only_removes_unstarted_countdown(self):
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
                self.assertEqual(before_calls, self.api.call_count)
                self.assertIn("start is not established", result["reason"])
            else:
                kept.append(index)
        self.assertEqual([0], changed)
        self.assertEqual([2, 4, 6, 7, 8], kept)
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

    def test_saved_missouri_proposal_reaches_review_but_wrong_source_quotes_still_fail(self):
        path = "focused_review.json"
        saved = load_fixture(path)["source_wording_alternative_order_review"]
        case = saved["cases"][1]
        raw = json.loads(saved["calls"][2]["response"])
        self.api.side_effect = None
        for writer in (review_signals.evaluate_signal_triage, review_signals.evaluate_next_signal):
            self.api.return_value = response({**raw, "eligible": True})
            proposal = writer(saved["date"], case["topic"], case["evidence"])
            self.assertTrue(proposal["eligible"])
            self.assertEqual(raw["signal"], proposal["signal"])
            self.assertEqual(3, proposal["source_index"])
            self.assertNotIn("comparisons", proposal)
            # The writer's old audit is ignored. The independent reviewer still
            # must supply accurate source quotes and cannot inherit these errors.
            self.api.return_value = response({"eligible": True, "comparisons": [
                {**item, "before_selected": False} for item in raw["comparisons"]
            ], "reason": "Simulated review with incorrect source quotes."})
            with self.assertRaisesRegex(RuntimeError, "timing is not quoted"):
                review_signals.validate_candidate_signal_semantics(
                    saved["date"], case["topic"], case["evidence"], proposal,
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
            self.assertEqual(1, files["report"][0]["model_call_count"])
            self.assertEqual(2, self.api.call_count)
            self.assertEqual(changed, files["summary"][0]["next_signal"])

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
            self.comparison(before=True), self.comparison(2, "Wednesday", qualifies=False),
        ]
        self.api.side_effect = [
            response({"status": "SIGNAL", "signal": later["description"], "source_index": 2}),
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
        external = {**self.row, "description": "The panel will issue findings Wednesday.",
                    "grounded_sentence": "The panel will issue findings Wednesday."}
        candidate = {"eligible": True, "signal": external["description"], "source_index": 2}
        self.api.side_effect = None
        self.api.return_value = response({**self.approval, "signal": "Unwanted rewrite.", "source_index": 1,
                                         "comparisons": [self.comparison(1, qualifies=False, before=True,
                                                                         reason="Local passage insufficient for this story."),
                                                         self.comparison(2, "Wednesday")]})
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
                   expanded_age=None, signal_exit=0, dry_run=False, summary_exit=0, global_fail_stage=None):
        from types import SimpleNamespace
        import os
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            day = "2030-12-31"
            final = root / f"grouped_articles_final_{day}.json"
            final.write_text("[]")
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
                return global_success and kwargs["script"] != global_fail_stage
            args = SimpleNamespace(date=day, with_gdelt_discovery=global_enabled,
                                   gdelt_discovery_cached_only=False, dry_run=dry_run)
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
                              "--cluster-file", expected], call["extra_args"])
            self.assertEqual(dry_run, call["dry_run"])

    def test_baseline_selects_same_fresh_input_as_summarizer(self):
        for age in (None, 100, 200, 300):
            with self.subTest(expanded_age=age):
                self.check_path(expanded_age=age)

    def test_global_receipts_and_discovery_fallback(self):
        self.check_path(global_enabled=True, global_success=True)
        self.check_path(global_enabled=True, global_success=False, expanded_age=300)
        self.check_path(global_enabled=True, global_success=True,
                        global_fail_stage="summarize_grouped_topics.py", expanded_age=300)

    def test_signal_failure_stops_success_and_summary_failure_skips_signal(self):
        self.check_path(signal_exit=7)
        self.check_path(summary_exit=3)

    def test_dry_run_propagates_to_signal(self):
        self.check_path(dry_run=True)
        self.check_path(global_enabled=True, global_success=True, dry_run=True)


if __name__ == "__main__":
    unittest.main()
