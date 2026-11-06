"""Publication must preserve the last edition until review and ranking finish."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'final_cohesion_check.py'
sys.path.insert(0, str(PATH.parent))
import run_pipeline_by_date as runner


class ReviewCompletion(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('socket.socket.connect', side_effect=AssertionError('Offline only')))
        self.tree = ast.parse(PATH.read_text(encoding='utf-8'))
        names = {'_publication_review_status', '_require_complete_review', '_load_event_cache', '_save_event_cache'}
        nodes = [n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.ns = dict(Path=Path, json=json, os=os, tempfile=tempfile, _EVENT_SPENT_USD=.2, EVENT_BUDGET_USD=.75)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), self.ns)

    def status(self, count=10, reason='queue_exhausted_rank_complete', unresolved=None):
        return self.ns['_publication_review_status'](10, count, reason, 60, 60, unresolved or [])

    def test_count_and_rank_completion_are_both_required(self):
        self.assertEqual('COMPLETE', self.status()['status'])
        self.assertEqual('INCOMPLETE', self.status(count=9)['status'])
        for reason in ('budget_reached_before_rank_secured', 'live_call_cap_reached_before_rank_secured',
                       'queue_exhausted_with_unresolved_candidates', 'queue_exhausted_before_target', 'not_run'):
            with self.subTest(reason=reason):
                self.assertEqual('INCOMPLETE', self.status(count=15, reason=reason)['status'])

    def test_incomplete_review_preserves_existing_edition_and_reports_resume_state(self):
        with tempfile.TemporaryDirectory(dir=PATH.parent / 'tests') as directory:
            edition = Path(directory) / 'edition.json'
            original = b'[{"previous": "finished edition"}]'
            edition.write_bytes(original)
            status = self.status(reason='budget_reached_before_rank_secured', unresolved=[{'action': 'DEFER'}])
            with self.assertRaises(SystemExit) as failure:
                self.ns['_require_complete_review'](status, str(edition))
            self.assertEqual(2, failure.exception.code)
            self.assertEqual(original, edition.read_bytes())
            saved = json.loads(edition.with_suffix('.review.json').read_text())
            self.assertEqual('INCOMPLETE', saved['status'])
            self.assertEqual([{'action': 'DEFER'}], saved['unresolved'])

    def test_guard_runs_before_either_edition_output_is_written(self):
        main = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        guard = next(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == '_require_complete_review')
        writes = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Call) and
                  ('Path(gdelt_global_output_file).write_text' in ast.unparse(n.func)
                   or isinstance(n.func, ast.Name) and n.func.id == 'open'
                   and n.args and ast.unparse(n.args[0]) == 'output_file')]
        self.assertEqual(2, len(writes))
        self.assertTrue(all(line > guard for line in writes))

    def test_secured_bounds_allow_irrelevant_unresolved_candidates(self):
        status = self.status(reason='top_k_local_and_global_rank_secured', unresolved=[{'topic': 'below cutoff'}])
        self.assertEqual('COMPLETE', status['status'])

    def test_cache_round_trip_and_failed_replace_preserve_completed_work(self):
        with tempfile.TemporaryDirectory(dir=PATH.parent / 'tests') as directory:
            path = Path(directory) / 'cache.json'
            self.ns['EVENT_CACHE_FILE'] = str(path)
            self.assertEqual({}, self.ns['_load_event_cache']())
            old = {'checked': {'action': 'KEEP'}}
            self.ns['_save_event_cache'](old)
            self.assertEqual(old, self.ns['_load_event_cache']())
            before = path.read_bytes()
            with patch('os.replace', side_effect=PermissionError('Simulated failed replacement')):
                with self.assertRaises(PermissionError):
                    self.ns['_save_event_cache']({'new': {'action': 'REJECT'}})
            self.assertEqual(before, path.read_bytes())
            self.assertEqual([path], list(Path(directory).iterdir()))

    def test_damaged_cache_stops_instead_of_silently_repeating_paid_work(self):
        with tempfile.TemporaryDirectory(dir=PATH.parent / 'tests') as directory:
            path = Path(directory) / 'cache.json'
            self.ns['EVENT_CACHE_FILE'] = str(path)
            for text in ('unfinished {', '[]'):
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.ns['_load_event_cache']()


class RunnerReviewStops(unittest.TestCase):
    """Run the real orchestration, replacing only stage subprocesses with offline stubs."""

    def setUp(self):
        self.enterContext(patch('socket.socket.connect', side_effect=AssertionError('Offline only')))
        directory = self.enterContext(tempfile.TemporaryDirectory(dir=PATH.parent / 'tests'))
        self.root = Path(directory)
        self.day = '2034-06-12'
        self.calls = []
        self.review_exit = 2
        self.discovery_exit = 0
        self.write_outputs = True
        for name in ('fetch_articles_by_date.py', 'normalize_and_dedupe_articles.py', 'bias_labeler.py',
                     'generate_market_overview.py', 'cluster_articles_by_embedding.py',
                     'audit_gdelt_global_discovery.py', 'inject_gdelt_discovery_candidates.py',
                     'merge_similar_clusters.py', 'filter_outlier_articles.py', 'report_cluster_cohesion.py',
                     'final_cohesion_check.py', 'enrich_gdelt_discovery_articles.py',
                     'expand_cluster_coverage_gdelt.py', 'attach_gdelt_global_receipts.py',
                     'summarize_grouped_topics.py', 'generate_next_signals.py'):
            (self.root / name).touch()
        self.originals = {}
        for stem in ('topic_summaries', 'grouped_articles_final', 'grouped_articles_final_global',
                     'gdelt_global_ranking', 'grouped_articles_final_global_receipts'):
            path = self.root / f'{stem}_{self.day}.json'
            path.write_text('[{"previous": "edition"}]')
            self.originals[path] = path.read_bytes()
        self.cache = self.root / 'eventness_cache.json'
        self.cache.write_text('{"saved-review":"verified"}')
        self.enterContext(patch.object(runner, '__file__', str(self.root / 'run_pipeline_by_date.py')))
        self.enterContext(patch.object(runner, 'manifest_is_usable', return_value=True))
        self.enterContext(patch.object(runner.subprocess, 'run', side_effect=self.child))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def child(self, command, **kwargs):
        name = Path(command[1]).name
        self.calls.append((command, kwargs))
        if name == 'audit_gdelt_global_discovery.py':
            (self.root / f'gdelt_discovery_candidates_{self.day}.json').write_text('{"candidates":[]}')
            return subprocess.CompletedProcess(command, self.discovery_exit)
        if name == 'report_cluster_cohesion.py':
            (self.root / f'grouped_articles_filtered_{self.day}.json').write_text('[{"articles":[]}]')
        if name == 'final_cohesion_check.py':
            self.assertEqual('{"saved-review":"verified"}', self.cache.read_text())
            (self.root / f'grouped_articles_final_{self.day}.review.json').write_text(json.dumps({
                'status': 'INCOMPLETE' if self.review_exit else 'COMPLETE',
                'estimated_spend_or_reservation_usd': .4,
                'stop_reason': 'budget_reached_before_rank_secured' if self.review_exit else 'top_k_rank_secured'}))
            if not self.review_exit and self.write_outputs:
                for stem in ('grouped_articles_final', 'grouped_articles_final_global', 'gdelt_global_ranking'):
                    (self.root / f'{stem}_{self.day}.json').write_text('[]')
            return subprocess.CompletedProcess(command, self.review_exit)
        if '--output-file' in command:
            (self.root / command[command.index('--output-file') + 1]).write_text('[]')
        return subprocess.CompletedProcess(command, 0)

    def run_pipeline(self, *flags):
        with patch.object(sys, 'argv', ['runner', '--date', self.day, *flags]):
            return runner.run()

    def reviews(self):
        return [call for call in self.calls if Path(call[0][1]).name == 'final_cohesion_check.py']

    def test_incomplete_and_failed_global_reviews_stop_once_and_preserve_edition(self):
        for code in (2, 7):
            with self.subTest(exit_code=code):
                self.review_exit = code
                flags = ['--resume-story-review'] if self.reviews() else []
                self.calls.clear()
                self.assertEqual(code, self.run_pipeline(*flags, '--story-review-budget-usd', '0.45'))
                self.assertEqual(1, len(self.reviews()))
                self.assertIn('--gdelt-ranking-shadow', self.reviews()[0][0])
                self.assertEqual('final_cohesion_check.py', Path(self.calls[-1][0][1]).name)
                self.assertEqual('0.45', self.reviews()[0][1]['env']['NN_EVENT_BUDGET_USD'])
                for path, original in self.originals.items():
                    self.assertEqual(original, path.read_bytes())

    def test_resume_skips_upstream_and_keeps_global_or_baseline_review_arguments(self):
        for global_mode in (True, False):
            with self.subTest(global_mode=global_mode):
                self.day = '2034-06-13' if global_mode else '2034-06-14'
                self.discovery_exit = 0 if global_mode else 1
                self.review_exit = 2
                self.calls.clear()
                self.assertEqual(2, self.run_pipeline('--story-review-budget-usd', '0.4'))
                original_command = self.reviews()[0][0]
                self.calls.clear()
                self.review_exit = 0
                self.assertEqual(0, self.run_pipeline('--resume-story-review', '--story-review-budget-usd', '0.1'))
                self.assertEqual(original_command, self.calls[0][0])
                self.assertEqual(1, len(self.reviews()))
                self.assertEqual('0.1', self.reviews()[0][1]['env']['NN_EVENT_BUDGET_USD'])
                names = [Path(c[0][1]).name for c in self.calls]
                self.assertNotIn('fetch_articles_by_date.py', names)
                self.assertNotIn('merge_similar_clusters.py', names)
                self.assertEqual('generate_next_signals.py', names[-1])
                state = json.loads(runner.review_state_path(self.root, self.day).read_text())
                self.assertEqual('complete', state['status'])
                self.assertEqual(['0.4', '0.1'], [a['allowance_usd'] for a in state['attempts']])

    def test_restart_cannot_bypass_pending_review_or_change_path(self):
        self.assertEqual(2, self.run_pipeline())
        for flags in ([], ['--no-gdelt-discovery'],
                      ['--resume-story-review', '--no-gdelt-discovery', '--story-review-budget-usd', '0.1']):
            self.calls.clear()
            self.assertEqual(1, self.run_pipeline(*flags))
            self.assertEqual([], self.calls)

    def test_resume_requires_explicit_budget_even_when_environment_has_one(self):
        with patch.dict(os.environ, {'NN_EVENT_BUDGET_USD':'0.7'}), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_pipeline('--resume-story-review')
        self.assertEqual([], self.calls)

    def test_changed_evidence_code_or_settings_blocks_resume_without_calls(self):
        self.assertEqual(2, self.run_pipeline())
        for name in (f'grouped_articles_filtered_{self.day}.json',
                     f'gdelt_discovery_candidates_{self.day}.json', 'final_cohesion_check.py'):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b'\n')
            self.calls.clear()
            self.assertEqual(1, self.run_pipeline('--resume-story-review', '--story-review-budget-usd', '0.1'))
            self.assertEqual([], self.calls)
            path.write_bytes(original)
        with patch.dict(os.environ, {'NN_EVENT_MODEL':'different-model'}):
            self.assertEqual(1, self.run_pipeline('--resume-story-review', '--story-review-budget-usd', '0.1'))
            self.assertEqual([], self.calls)

    def test_missing_review_outputs_stop_instead_of_using_baseline(self):
        for path in self.originals:
            if 'topic_summaries' not in path.name:
                path.unlink()
        self.review_exit = 0
        self.write_outputs = False
        self.assertEqual(1, self.run_pipeline())
        self.assertEqual(1, len(self.reviews()))
        self.assertEqual('final_cohesion_check.py', Path(self.calls[-1][0][1]).name)

    def test_resume_dry_run_does_not_change_receipts_or_start_subprocesses(self):
        self.assertEqual(2, self.run_pipeline())
        path = runner.review_state_path(self.root, self.day)
        before = path.read_bytes()
        self.calls.clear()
        self.assertEqual(0, self.run_pipeline('--resume-story-review', '--story-review-budget-usd', '0.1', '--dry-run'))
        self.assertEqual([], self.calls)
        self.assertEqual(before, path.read_bytes())

    def test_budget_environment_is_scoped_to_review_and_cli_overrides_inherited_value(self):
        with patch.dict(os.environ, {'NN_EVENT_BUDGET_USD':'0.7'}):
            self.assertEqual(2, self.run_pipeline('--story-review-budget-usd', '0.2'))
            self.assertEqual('0.7', os.environ['NN_EVENT_BUDGET_USD'])
        self.assertEqual('0.2', self.reviews()[0][1]['env']['NN_EVENT_BUDGET_USD'])
        self.assertTrue(all('env' not in kwargs for command, kwargs in self.calls
                            if Path(command[1]).name != 'final_cohesion_check.py'))

    def test_invalid_budget_or_missing_resume_receipt_starts_nothing(self):
        for value in ('0', '-1', 'NaN', 'Infinity', '-Infinity', 'invalid', '1e10000', '1e-10000'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_pipeline('--story-review-budget-usd', value)
        self.assertEqual(1, self.run_pipeline('--resume-story-review', '--story-review-budget-usd', '0.1'))
        self.assertEqual([], self.calls)


if __name__ == '__main__':
    unittest.main()
