"""Offline image-only tests; never import the executable summarization pipeline."""
import ast
from pathlib import Path
import re
import types
import unittest
from unittest.mock import patch


def load_image_code():
    path = Path(__file__).resolve().parents[1] / "summarize_grouped_topics.py"
    source = path.read_text(encoding="utf-8")
    start = source.index("_STOP = {")
    end = source.index("BIAS_LABELS =")
    module = types.ModuleType("image_selection_only")
    module.re = re
    exec(compile(ast.parse(source[start:end]), str(path), "exec"), module.__dict__)
    return module


class ImageIdentityChecks(unittest.TestCase):
    def setUp(self):
        self.images = load_image_code()
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")))

    def test_activity_caption_vetoes_location_classification(self):
        metrics = {"anchor_score": 4, "anchor_type": "institution",
                   "scene_type": "institution", "topic_semantic": .8,
                   "anchor_semantic": .8, "title_focus": 1,
                   "title_anchor_match": True, "caption_scene_type": "event"}
        self.assertFalse(self.images._context_candidate_is_safe(metrics))
        metrics["caption_scene_type"] = "institution"
        self.assertTrue(self.images._context_candidate_is_safe(metrics))
        metrics.update(anchor_type="event", scene_type="event", caption_scene_type="event")
        self.assertTrue(self.images._context_candidate_is_safe(metrics))
        metrics.update(anchor_type="person", scene_type="person")
        self.assertTrue(self.images._context_candidate_is_safe(metrics))

    def test_commons_credit_markup_keeps_names_and_notices(self):
        raw = '[<a href="https://example.org/author">https://example.org/author</a> Renée &amp; Jo] / Courtesy Archive'
        self.assertEqual('Renée & Jo / Courtesy Archive', self.images._commons_credit_text(raw))

    def test_final_rights_withholds_unreadable_attribution(self):
        candidate, record = self.rights_fixture()
        record['imageinfo'][0]['extmetadata']['Artist']['value'] = 'Ren\ufffde'
        with patch.object(self.images, '_image_rights_record', return_value=record):
            result, review = self.images.final_image_rights_check(candidate)
        self.assertIsNone(result)
        self.assertEqual('withheld', review['status'])
        self.assertIn('Unreadable', review['reason'])

    def test_observed_commons_license_metadata(self):
        # Metadata sampled from Commons on 2026-10-13. These exercise the
        # existing policy, including restrictions independent of copyright.
        samples = (
            ("CC BY 2.0", "https://creativecommons.org/licenses/by/2.0", "", True),
            ("CC BY-SA 2.0", "https://creativecommons.org/licenses/by-sa/2.0", "", False),
            ("Public domain", "", "personality", False),
            ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0", "personality", False),
        )
        for short, url, restrictions, expected in samples:
            with self.subTest(short=short, restrictions=restrictions):
                self.assertEqual(expected, self.images._commons_license_is_approved(
                    short, url, {"Restrictions": {"value": restrictions}}))

    def test_license_policy_preserves_allowed_and_excluded_terms(self):
        for short, url, expected in (
            ("Public domain", "", True),
            ("CC0", "", True),
            ("CC BY 4.0", "", True),
            ("CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0", False),
            ("CC BY-ND 4.0", "https://creativecommons.org/licenses/by-nd/4.0", False),
            ("GFDL", "", False),
            ("Unknown", "", False),
            ("", "", False),
        ):
            with self.subTest(short=short):
                self.assertEqual(expected, self.images._commons_license_is_approved(short, url, {}))

    def test_openverse_accepts_only_verifiable_origin_and_license(self):
        record = {"source": "wikimedia", "license": "by",
                  "license_url": "https://creativecommons.org/licenses/by/2.0/",
                  "foreign_landing_url": "https://commons.wikimedia.org/w/index.php?curid=123"}
        self.assertEqual(123, self.images._openverse_commons_id(record))
        for change in (
            {"source": "flickr"}, {"license": "by-sa"}, {"mature": True},
            {"license_url": "https://example.com/creativecommons.org/licenses/by/2.0/"},
            {"license_url": "https://creativecommons.org/licenses/by-nc/4.0/"},
            {"foreign_landing_url": "https://commons.wikimedia.org.evil.test/?curid=123"},
            {"foreign_landing_url": "https://commons.wikimedia.org/?curid=123&curid=456"},
            {"foreign_landing_url": "http://127.0.0.1/?curid=123"},
        ):
            with self.subTest(change=change):
                self.assertIsNone(self.images._openverse_commons_id({**record, **change}))

    def test_openverse_rechecks_current_license_before_relevance(self):
        from types import SimpleNamespace
        m = self.images
        m.IMAGES_OFF = False
        # The index says CC BY, but the original now disallows reuse, or has
        # a separate restriction. Neither may reach the relevance scorer.
        record = {"source": "wikimedia", "license": "by",
                  "license_url": "https://creativecommons.org/licenses/by/2.0/",
                  "foreign_landing_url": "https://commons.wikimedia.org/?curid=123"}
        for short, restrictions in (("CC BY-SA 2.0", ""), ("Public domain", "personality")):
            m._OPENVERSE_CACHE.clear()
            calls = []
            def get(url, **kwargs):
                calls.append((url, kwargs))
                payload = {"results": [record]} if 'openverse.org' in url else {
                    "query": {"pages": {"123": {"pageid": 123, "title": "File:Example.jpg",
                        "imageinfo": [{"thumburl": "https://example.test/image.jpg", "extmetadata": {
                            "LicenseShortName": {"value": short},
                            "Restrictions": {"value": restrictions}}}]}}}}
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
            m.requests = SimpleNamespace(get=get)
            with patch.object(m, "_candidate_context_metrics", side_effect=AssertionError("Rights must reject first")):
                self.assertEqual([], m.fetch_openverse_image("Example", all_candidates=True))
            self.assertEqual("123", calls[1][1]["params"]["pageids"])
            self.assertNotIn("generator", calls[1][1]["params"])

    def test_openverse_failure_stops_further_requests_and_images_off_skips(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        m = self.images
        get = Mock(side_effect=RuntimeError("429"))
        m.requests = SimpleNamespace(get=get)
        m.IMAGES_OFF = True
        self.assertIsNone(m.fetch_openverse_image("Example"))
        get.assert_not_called()
        m.IMAGES_OFF = False
        self.assertEqual([], m.fetch_openverse_image("Example", all_candidates=True))
        self.assertIsNone(m.fetch_openverse_image("Another"))
        self.assertEqual(1, get.call_count)

    def test_openverse_keeps_complete_verified_attribution(self):
        from types import SimpleNamespace
        m = self.images
        m.IMAGES_OFF = False
        m._OPENVERSE_CACHE["Example Park"] = [123]
        ext = {"Artist": {"value": "Example Photographer"},
               "LicenseShortName": {"value": "CC BY 4.0"},
               "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
               "Attribution": {"value": "Courtesy Example Collection"}}
        payload = {"query": {"pages": {"123": {"pageid": 123, "title": "File:Example Park.jpg",
                   "imageinfo": [{"thumburl": "https://example.test/image.jpg", "mime": "image/jpeg", "extmetadata": ext}]}}}}
        m.requests = SimpleNamespace(get=lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload))
        metrics = {"anchor_score": 4, "anchor_type": "geographic", "scene_type": "geographic",
                   "topic_semantic": .8, "title_focus": .8, "foreign_ratio": 0}
        with patch.object(m, "classify_candidate_medium", return_value="photo"), \
             patch.object(m, "_candidate_context_metrics", return_value=metrics), \
             patch.object(m, "_context_candidate_is_safe", return_value=True), \
             patch.object(m, "_context_quality_score", return_value=3), \
             patch.object(m, "_selection_confidence", return_value=.8), \
             patch.object(m, "_commons_image_year", return_value=2026), \
             patch.object(m, "_temporal_context", return_value={"allowed": True, "freshness": 1., "image_year": 2026, "age_years": 0}):
            result = m.fetch_openverse_image("Example Park")
            self.assertEqual("wikimedia", result["source"])
            self.assertEqual("openverse", result["discovery_source"])
            self.assertIn("Example Park.jpg", result["credit"])
            self.assertIn("Example Photographer", result["credit"])
            self.assertIn("Courtesy Example Collection", result["credit"])
            self.assertEqual("https://commons.wikimedia.org/?curid=123", result["source_url"])
            # Missing attribution must fail closed even for a relevant photo.
            ext.pop("Artist")
            self.assertIsNone(m.fetch_openverse_image("Example Park"))

    def rights_fixture(self):
        candidate = {"source": "wikimedia", "source_url": "https://commons.wikimedia.org/?curid=123",
                     "url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/ab/Example.jpg/600px-Example.jpg",
                     "credit": "Old credit", "license": "CC BY 4.0"}
        record = {"title": "File:Example.jpg", "templates": [], "imageinfo": [{
            "url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Example.jpg",
            "extmetadata": {k: {"value": v} for k, v in {
                "Artist": "Example Creator", "Credit": "Own work", "LicenseShortName": "CC BY 4.0",
                "LicenseUrl": "https://creativecommons.org/licenses/by/4.0/",
                "Attribution": "Example Collection", "Restrictions": ""}.items()}}]}
        return candidate, record

    def test_final_rights_check_passes_and_preserves_evidence_and_credit(self):
        candidate, record = self.rights_fixture()
        with patch.object(self.images, "_image_rights_record", return_value=record):
            result, review = self.images.final_image_rights_check(candidate)
        self.assertEqual("passed", review["status"])
        self.assertEqual(record, review["source_record"])
        self.assertIn("Example Creator", result["credit"])
        self.assertIn("Example Collection", result["credit"])
        self.assertEqual("Old credit", candidate["credit"])

    def test_final_rights_fetch_distinguishes_image_history_from_missing_notices(self):
        from types import SimpleNamespace
        candidate, record = self.rights_fixture()
        record['ns'] = 6
        payload = {'query': {'pages': {'123': record}}, 'continue': {'iistart': '2013-01-01', 'continue': '||templates'}}
        self.images.requests = SimpleNamespace(get=lambda *a, **kw: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: payload))
        self.assertEqual(record, self.images._image_rights_record(candidate))
        payload['continue']['tlcontinue'] = '123|10|Example'
        with self.assertRaises(ValueError):
            self.images._image_rights_record(candidate)

    def test_final_rights_check_withholds_disputes_missing_credit_and_restrictions(self):
        import copy
        candidate, baseline = self.rights_fixture()
        for case in ("dispute", "restrictions", "creator", "source", "license", "wrong_image"):
            record = copy.deepcopy(baseline)
            ext = record['imageinfo'][0]['extmetadata']
            if case == "dispute": record['templates'] = [{"title": "Template:No permission since"}]
            elif case == "restrictions": ext['Restrictions']['value'] = 'personality'
            elif case == "creator": ext.pop('Artist')
            elif case == "source": ext.pop('Credit')
            elif case == "license": ext['LicenseShortName']['value'] = 'CC BY-SA 4.0'
            else: record['imageinfo'][0]['url'] = 'https://upload.wikimedia.org/wikipedia/commons/a/ab/Other.jpg'
            with self.subTest(case=case), patch.object(self.images, '_image_rights_record', return_value=record):
                result, review = self.images.final_image_rights_check(candidate)
                self.assertIsNone(result)
                self.assertEqual('withheld', review['status'])

    def test_final_rights_check_handles_no_image_and_provider_failure(self):
        with patch.object(self.images, '_image_rights_record', side_effect=RuntimeError('connection failed')) as fetch:
            self.assertEqual('not_applicable', self.images.final_image_rights_check(None)[1]['status'])
            fetch.assert_not_called()
            self.assertEqual('withheld', self.images.final_image_rights_check(self.rights_fixture()[0])[1]['status'])

    def test_final_rights_unsplash_checks_asset_and_retains_provider_attribution(self):
        candidate = {'source': 'unsplash', 'url': 'https://images.unsplash.com/photo-example?w=900'}
        record = {'urls': {'raw': 'https://images.unsplash.com/photo-example'},
                  'links': {'html': 'https://unsplash.com/photos/example-abcdefghijk'},
                  'user': {'name': 'Example Creator', 'links': {'html': 'https://unsplash.com/@example'}}}
        with patch.object(self.images, '_image_rights_record', return_value=record):
            result, review = self.images.final_image_rights_check(candidate)
            self.assertEqual('passed', review['status'])
            self.assertEqual('Unsplash License', result['license'])
            self.assertIn('utm_source=nominal_news', result['photographer_url'])
            record['premium'] = True
            self.assertIsNone(self.images.final_image_rights_check(candidate)[0])

    def queries(self, titles, accepted, limit=6):
        # Deliberately permit the fragments too: filtering must preserve identity
        # even when the classifier regards a surname/common word as a subject.
        with patch.object(self.images, "_image_named_entities",
                          side_effect=lambda title, description="": [(phrase, accepted[phrase.lower()])
                              for phrase in self.images._extract_proper_phrases(title)
                              if phrase.lower() in accepted]):
            return self.images.build_context_image_queries(
                titles[0], [{"title": title} for title in titles], max_queries=limit)

    def test_full_person_and_venue_displace_nested_fragments(self):
        for full, kind in (("Christa Pike", "person"), ("Six Flags", "institution"),
                           ("Jordan Banks", "person"), ("Silver Harbor", "institution")):
            words = full.lower().split()
            accepted = {full.lower(): kind, **{word: kind for word in words}}
            with self.subTest(full=full):
                result = self.queries([f"{full} faces questions", f"Report examines {full}"], accepted)
                self.assertEqual([full], result)

    def test_standalone_subjects_survive_and_generic_headlines_do_not_suppress(self):
        result = self.queries(["Iceland hosts summit", "Officials visit Iceland"], {"iceland": "geographic"})
        self.assertEqual(["Iceland"], result)
        result = self.queries(["Selma Blair Faces Charges", "Selma Blair appears in court"],
                              {"selma blair": "person", "blair": "person"})
        self.assertEqual(["Selma Blair"], result)

    def test_ambiguous_short_form_is_withheld_but_distinct_place_survives(self):
        result = self.queries(["Jordan Banks visits Jordan", "Jordan welcomes Jordan Banks"],
                              {"jordan banks": "person", "jordan": "geographic", "banks": "person"})
        # The ambiguous short place/name form remains withheld conservatively.
        self.assertEqual(["Jordan Banks"], result)
        result = self.queries(["Jordan Banks visits Iceland", "Iceland welcomes Jordan Banks"],
                              {"jordan banks": "person", "jordan": "person", "banks": "person", "iceland": "geographic"})
        self.assertEqual({"Jordan Banks", "Iceland"}, set(result))

    def test_unsupported_generated_names_cannot_become_queries(self):
        with patch.object(self.images, "_image_named_entities", return_value=[("Selma Blair", "person")]):
            queries = self.images.build_context_image_queries(
                "Invented Person speaks", [{"title": "Selma Blair speaks"}, {"title": "Selma Blair arrives"}])
        self.assertEqual(["Selma Blair"], queries)

    def test_contextual_entity_span_excludes_title_words(self):
        with patch.object(self.images, "_image_named_entities", return_value=[("Selma Blair", "person")]):
            queries = self.images.build_context_image_queries(
                "Actress Selma Blair Faces Charges", [{"title": "Actress Selma Blair faces charges"}])
        self.assertEqual(["Selma Blair"], queries)

    def test_missing_local_entity_model_fails_closed(self):
        self.images._IMAGE_NLP_ATTEMPTED = True
        self.images._IMAGE_NLP = None
        self.assertEqual([], self.images.build_context_image_queries(
            "Example Person speaks", [{"title": "Example Person speaks"}]))
        self.assertTrue(any(x["reason"] == "local_entity_model_unavailable" for x in self.images._IMAGE_AUDIT))

    def test_context_type_conflicts_withhold_subject(self):
        with patch.object(self.images, "_image_named_entities", side_effect=[[("Jordan", "person")], [("Jordan", "geographic")]]):
            result = self.images.build_context_image_queries("Jordan", [{"title": "Jordan speaks"}, {"title": "Jordan votes"}])
        self.assertEqual([], result)

    def candidate(self, url, score, confidence):
        return {"url": url, "query": "Example Person", "_selection_score": score, "_confidence": confidence}

    def test_eligible_runner_up_survives_low_confidence_winner(self):
        rejected = self.candidate("bad", 100, .47)
        valid = self.candidate("good", 90, .70)
        self.assertIs(valid, self.images.choose_best_context_candidate([rejected, valid], "wikimedia"))
        self.assertTrue(any(x["reason"] == "confidence_floor" for x in self.images._IMAGE_AUDIT))

    def test_ambiguity_margin_is_unchanged_for_distinct_images(self):
        self.assertIsNone(self.images.choose_best_context_candidate([
            self.candidate("one", 100, .55), self.candidate("two", 99, .55)], "wikimedia"))
        self.assertEqual("ambiguous_candidates", self.images._IMAGE_AUDIT[-1]["reason"])

    def test_same_photo_from_two_queries_is_not_two_competing_images(self):
        first = self.candidate("photo?query=1", 100, .55)
        second = self.candidate("photo?query=2", 99, .55)
        self.assertIs(first, self.images.choose_best_context_candidate([first, second], "wikimedia"))

    def test_provider_keeps_multiple_hard_checked_candidates(self):
        from types import SimpleNamespace
        m = self.images
        m.IMAGES_OFF = False
        def page(number, title):
            return {"pageid": number, "title": "File: " + title, "imageinfo": [{
                "thumburl": "https://example.test/" + str(number), "mime": "image/jpeg",
                "extmetadata": {"ImageDescription": {"value": title},
                    "LicenseShortName": {"value": "CC BY 2.0"},
                    "LicenseUrl": {"value": "https://example.test/license"}}}]}
        payload = {"query": {"pages": {str(i): page(i, title) for i, title in enumerate(
            ["Example Person portrait one", "Example Person portrait two", "Unrelated subject"])}}}
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
        m.requests = SimpleNamespace(get=lambda *args, **kwargs: response)
        def metrics(anchor, topic, title, desc):
            return {"anchor_score": 4 if title.startswith("Example Person") else 0,
                    "anchor_type": "person", "scene_type": "person", "topic_semantic": .8,
                    "anchor_semantic": .8, "title_focus": .6, "title_anchor_match": True,
                    "title_foreign": 0, "title_extensions": 0, "desc_foreign": 0,
                    "desc_extensions": 0, "foreign_ratio": 0}
        with patch.object(m, "_commons_license_is_approved", return_value=True), \
             patch.object(m, "_wiki_media_is_photo_candidate", return_value=True), \
             patch.object(m, "classify_candidate_medium", return_value="photo"), \
             patch.object(m, "_candidate_context_metrics", side_effect=metrics), \
             patch.object(m, "_commons_image_year", return_value=2026), \
             patch.object(m, "_temporal_context", return_value={"allowed": True, "freshness": 1., "image_year": 2026, "age_years": 0}):
            results = m.fetch_wikimedia_image("Example Person", all_candidates=True)
        self.assertEqual(2, len(results))
        self.assertEqual({"https://example.test/0", "https://example.test/1"}, {x["url"] for x in results})
        self.assertTrue(any(x["reason"] == "missing_subject_identity" for x in m._IMAGE_AUDIT))

    def test_retrieval_and_high_similarity_cannot_replace_subject_identity(self):
        metrics = {"anchor_type": "event", "scene_type": "event", "anchor_score": 0,
                   "topic_semantic": .99, "anchor_semantic": .99, "title_focus": .9,
                   "foreign_ratio": 0, "title_foreign": 0, "title_extensions": 0}
        self.assertFalse(self.images._context_candidate_is_safe(metrics))
        # Supported named events remain eligible when metadata identifies them.
        metrics.update(anchor_score=4, title_anchor_match=True)
        self.assertTrue(self.images._context_candidate_is_safe(metrics))

    def test_word_boundaries_and_order_are_required(self):
        for anchor, text, expected in (("Ann Lee", "Portrait of Joann Lee", 0),
                                      ("Ann Lee", "Portrait of Ann Lee", 4),
                                      ("Six Flags", "Flags at six locations", 0),
                                      ("Six Flags", "Six Flags entrance", 4),
                                      ("Christa Pike", "Pike caught in a lake", 0),
                                      ("Iceland", "Iceland landscape", 3)):
            with self.subTest(anchor=anchor, text=text):
                self.assertEqual(expected, self.images.anchor_match_score(anchor, text))

    def test_no_fragment_fallback_when_full_subject_has_no_image(self):
        queries = self.queries(["Christa Pike awaits review", "Review of Christa Pike"],
                              {"christa pike": "person", "christa": "person", "pike": "person"}, limit=3)
        self.assertEqual(["Christa Pike"], queries)
        self.assertIsNone(self.images.choose_best_context_candidate([], source="wikimedia"))


if __name__ == "__main__":
    unittest.main()
