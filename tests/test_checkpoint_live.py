"""Checkpoint early withholding and live I/O, using only fake HTTP responses."""
from copy import deepcopy
from decimal import Decimal
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import generate_next_signals as rules
import next_signal_checkpoint_review as review
import next_signal_checkpoint_io as live
from test_checkpoint_review import fixture, receipt, memory_run


def evaluate(worker, case, cache, **kwargs):
    return worker.evaluate(case['edition'], case['topic'], case['evidence'], cache=cache, **kwargs)


def raw_reply(payload, **overrides):
    result = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(payload)}}],
              'usage': {'prompt_tokens': 100, 'completion_tokens': 10}, 'service_tier': 'default'}
    result.update(overrides)
    return io.BytesIO(json.dumps(result).encode('utf-8'))


class EarlyWithholding(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline only')))
        self.worker = review.ReviewAdapter(rules)
        self.case = fixture()

    def pair(self):
        case = deepcopy(self.case)
        case['evidence'].append({**case['evidence'][0],
                                'grounded_sentence':'The panel will meet next week.',
                                'description':'The panel will meet next week.'})
        return case, self.worker.candidates(case)

    def test_no_resolved_display_date_needs_no_model_or_receipts(self):
        case = self.case
        case['evidence'][0].update(description='The panel will meet next week.',
                                   grounded_sentence='The panel will meet next week.')
        transport = Mock(side_effect=AssertionError('Unnecessary review'))
        result = evaluate(self.worker, case, {}, transport=transport, budget=review.Budget(max_calls=1,max_usd=1))
        self.assertFalse(result.decision['eligible'])
        self.assertEqual('no_resolved_display_time', result.decision['withholding_basis'])
        self.assertFalse(result.decision['all_candidates_reviewed'])
        transport.assert_not_called()

    def test_one_exact_cached_blocker_ends_review_with_missing_other_receipts(self):
        case, rows = self.pair()
        row = next(r for r in rows if r['window'] is None)
        cache = {}
        self.worker.remember(cache,case,row,receipt(self.worker,case,row))
        result = evaluate(self.worker,case,cache)
        self.assertFalse(result.decision['eligible'])
        self.assertEqual('blocking_review',result.decision['withholding_basis'])
        self.assertEqual(row['candidate_id'],result.decision['blocking_candidate_id'])
        self.assertEqual(1,result.decision['reviewed_candidate_count'])
        self.assertEqual(2,result.candidate_count)

    def test_first_live_blocker_saves_calls_even_with_one_call_allowance(self):
        case, rows = self.pair()
        unknown = next(r for r in rows if r['window'] is None)
        transport = Mock(return_value={'response':receipt(self.worker,case,unknown),
                                      'usage':{'prompt_tokens':100,'completion_tokens':10},'finish_reason':'stop'})
        result = evaluate(self.worker,case,{},transport=transport,budget=review.Budget(max_calls=1,max_usd=1))
        self.assertFalse(result.decision['eligible'])
        self.assertEqual(1,result.review_calls)
        self.assertFalse(result.decision['all_candidates_reviewed'])
        transport.assert_called_once()

    def test_unrelated_candidate_cannot_stand_in_for_complete_review(self):
        case, rows = self.pair()
        unknown = next(r for r in rows if r['window'] is None)
        answer = receipt(self.worker,case,unknown)
        answer['assessments'][0]['story_connection'].update(kind='SHARED_SUBJECT_ONLY',evidence_id=1)
        cache = {}
        self.worker.remember(cache,case,unknown,answer)
        with self.assertRaises(review.CacheMiss):
            evaluate(self.worker,case,cache)

    def test_missing_positive_review_cannot_be_hidden_by_call_limit(self):
        case, rows = self.pair()
        unknown = next(r for r in rows if r['window'] is None)
        answer = receipt(self.worker,case,unknown)
        answer['assessments'][0]['story_connection'].update(kind='SHARED_SUBJECT_ONLY',evidence_id=1)
        transport = Mock(return_value={'response':answer,'finish_reason':'stop',
                                      'usage':{'prompt_tokens':100,'completion_tokens':10}})
        cache = {}
        with self.assertRaises(review.ReviewLimitReached):
            evaluate(self.worker,case,cache,transport=transport,budget=review.Budget(max_calls=1,max_usd=1))
        self.assertEqual(1,len(cache))
        transport.assert_called_once()

    def test_corrupt_receipt_is_not_hidden_by_a_valid_blocker(self):
        case, rows = self.pair()
        cache = {}
        for row in rows:
            self.worker.remember(cache,case,row,receipt(self.worker,case,row))
        key = self.worker.cache_identity(case, next(r for r in rows if r['window']))[0]
        cache[key]['response']['assessments'][0]['claims'].pop()
        with self.assertRaises(review.ReviewError):
            evaluate(self.worker,case,cache)


class LiveSessionChecks(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(socket.socket,'connect',side_effect=AssertionError('No real network')))
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.cache_file = Path(self.temp)/'receipts.json'
        self.ledger_file = Path(self.temp)/'usage.json'
        self.case = fixture()
        self.worker = review.ReviewAdapter(rules)
        self.row = self.worker.candidates(self.case)[0]
        self.answer = receipt(self.worker,self.case,self.row)
        self.key = 'FAKE_KEY_NEVER_LOG_THIS'

    def session(self, **overrides):
        args = dict(edition=self.case['edition'],cache_file=self.cache_file,ledger_file=self.ledger_file,
                    max_calls=2,max_usd='0.20',api_key=self.key)
        args.update(overrides)
        return live.CheckpointSession(**args)

    def run_case(self, session):
        return evaluate(self.worker,self.case,session.cache,transport=session.send,budget=session.budget)

    def read_ledger(self):
        return json.loads(self.ledger_file.read_text(encoding='utf-8'))

    def test_transport_has_no_environment_proxy_and_verifies_tls(self):
        import ssl
        from urllib.request import HTTPSHandler, ProxyHandler
        with patch.dict('os.environ', {'HTTPS_PROXY': 'http://127.0.0.1:8888'}):
            with self.session() as session:
                handlers = session._opener.handlers
                self.assertFalse(any(isinstance(h, ProxyHandler) for h in handlers))
                https = next(h for h in handlers if isinstance(h, HTTPSHandler))
                self.assertTrue(https._context.check_hostname)
                self.assertEqual(https._context.verify_mode, ssl.CERT_REQUIRED)
                redirect = next(h for h in handlers if isinstance(h, live._NoRedirect))
                self.assertIsNone(redirect.redirect_request(None, None, 302, '', {}, 'https://elsewhere.example/'))

    def test_reservation_precedes_http_and_receipts_resume_without_spending(self):
        with self.session(max_calls=1) as session:
            def send(req, timeout):
                state=self.read_ledger()
                self.assertEqual(1,state['calls'])
                self.assertTrue(state['pending']['sent'])
                self.assertGreater(Decimal(state['accounted_usd']),0)
                self.assertEqual('https://api.openai.com/v1/chat/completions',req.full_url)
                self.assertEqual('Bearer '+self.key,req.get_header('Authorization'))
                body=json.loads(req.data)
                self.assertEqual('default',body.pop('service_tier'))
                self.assertEqual(self.worker.request(self.case,self.row),body)
                self.assertEqual(45,timeout)
                return raw_reply(self.answer)
            session._opener.open=Mock(side_effect=send)
            self.assertTrue(self.run_case(session).decision['eligible'])
            self.assertEqual(Decimal('0.000120'),session.budget.accounted_usd)
            self.assertIsNone(session.budget.pending)
            session._opener.open.assert_called_once()
        with self.session(max_calls=1) as resumed:
            resumed._opener.open=Mock(side_effect=AssertionError('Receipt should be reused'))
            result=self.run_case(resumed)
            self.assertTrue(result.decision['eligible'])
            self.assertEqual(1,result.cache_hits)
            self.assertEqual(1,resumed.budget.calls)
            self.assertEqual(Decimal('0.000120'),resumed.budget.accounted_usd)
            resumed._opener.open.assert_not_called()
        for path in (self.cache_file,self.ledger_file):
            self.assertNotIn(self.key,path.read_text(encoding='utf-8'))

    def test_budget_prevents_network_before_first_request(self):
        with self.session(max_usd='0.00001') as session:
            session._opener.open=Mock()
            with self.assertRaises(review.ReviewLimitReached):self.run_case(session)
            session._opener.open.assert_not_called()
            self.assertEqual(0,session.budget.calls)

    def test_http_failure_has_no_retry_and_blocks_automatic_restart(self):
        with self.session() as session:
            session._opener.open=Mock(side_effect=HTTPError('https://api.openai.com',429,'failed',{},None))
            with self.assertRaisesRegex(review.ReviewError,'429'):self.run_case(session)
            session._opener.open.assert_called_once()
            self.assertTrue(session.budget.halted)
        state=self.read_ledger()
        self.assertGreater(Decimal(state['accounted_usd']),0)
        self.assertIsNotNone(state['pending'])
        with self.assertRaisesRegex(review.ReviewError,'unfinished or failed'):self.session()
        self.assertEqual(state,self.read_ledger())

    def test_interruption_retains_reservation_without_reissuing_request(self):
        with self.session() as session:
            session.budget.reserve(self.worker.request(self.case,self.row))
        before=self.read_ledger()
        with self.assertRaisesRegex(review.ReviewError,'unfinished or failed'):self.session()
        self.assertEqual(before,self.read_ledger())

    def test_truncation_and_invalid_json_are_not_cached_as_editorial_omissions(self):
        for content,finish in (('{}','length'),('{','stop')):
            with self.subTest(finish=finish), tempfile.TemporaryDirectory() as temp:
                with self.session(cache_file=Path(temp)/'cache.json',ledger_file=Path(temp)/'ledger.json') as session:
                    session._opener.open=Mock(return_value=raw_reply(self.answer,choices=[{
                        'finish_reason':finish,'message':{'content':content}}]))
                    with self.assertRaises(review.ReviewError):self.run_case(session)
                    self.assertEqual({},dict(session.cache))
                    self.assertTrue(session.budget.halted)
                    self.assertEqual(Decimal('0.000120'),session.budget.accounted_usd)
                    session._opener.open.assert_called_once()

    def test_invalid_usage_retains_full_reservation(self):
        with self.session() as session:
            session._opener.open=Mock(return_value=raw_reply(self.answer,usage=None))
            with self.assertRaisesRegex(review.ReviewError,'usage'):self.run_case(session)
            self.assertEqual(Decimal(session.budget.pending['reservation_usd']),session.budget.accounted_usd)
            self.assertTrue(session.budget.halted)
            self.assertEqual({},dict(session.cache))

    def test_cache_write_failure_keeps_cost_and_blocks_a_duplicate_paid_retry(self):
        save=live._save
        def fail_cache(path,value):
            if Path(path)==self.cache_file:raise PermissionError('Simulated write failure')
            return save(path,value)
        with self.session() as session:
            session._opener.open=Mock(return_value=raw_reply(self.answer))
            with patch.object(live,'_save',side_effect=fail_cache),self.assertRaises(review.ReviewError):
                self.run_case(session)
            self.assertTrue(session.budget.pending['settled'])
            self.assertEqual(Decimal('0.000120'),session.budget.accounted_usd)
        with self.assertRaises(review.ReviewError):self.session()

    def test_reservation_write_failure_never_reaches_network(self):
        with self.session() as session:
            session._opener.open=Mock()
            with patch.object(live,'_save',side_effect=PermissionError('Simulated disk failure')):
                with self.assertRaises(PermissionError):self.run_case(session)
            session._opener.open.assert_not_called()

    def test_concurrent_runs_cannot_share_either_budget_or_cache(self):
        with self.session():
            for change in ({},{'cache_file':Path(self.temp)/'second-cache.json'},
                           {'ledger_file':Path(self.temp)/'second-ledger.json'}):
                with self.subTest(change=change),self.assertRaises(OSError):self.session(**change)

    def test_corrupt_ledger_and_foreign_edition_fail_closed(self):
        self.ledger_file.write_text('{',encoding='utf-8')
        with self.assertRaises(ValueError):self.session()
        live._save(self.ledger_file,{'context':{'edition':'foreign'}})
        with self.assertRaisesRegex(review.ReviewError,'different edition'):self.session()

    def test_redirects_are_disabled(self):
        self.assertIsNone(live._NoRedirect().redirect_request(None,None,302,'',{},'https://elsewhere.test'))

    def test_spending_limit_does_not_reset_on_resume(self):
        with self.session(max_calls=1) as session:
            session._opener.open=Mock(return_value=raw_reply(self.answer))
            self.run_case(session)
        self.case['topic']['summary']='Changed evidence requiring a fresh review.'
        with self.session(max_calls=1) as session:
            session._opener.open=Mock()
            with self.assertRaises(review.ReviewLimitReached):self.run_case(session)
            session._opener.open.assert_not_called()
            self.assertEqual(1,session.budget.calls)

    def test_invalid_finite_limits_do_not_allow_requests(self):
        for value in ('NaN','Infinity','-1'):
            with self.subTest(value=value),self.assertRaises((ValueError,ArithmeticError)):
                self.session(max_usd=value)

    def test_main_live_route_uses_explicit_limits_and_preserves_other_fields(self):
        args=['--checkpoint-live','--checkpoint-budget-usd','0.20','--checkpoint-max-calls','1',
              '--checkpoint-cache-file',str(self.cache_file),'--checkpoint-ledger-file',str(self.ledger_file),
              '--checkpoint-max-output-tokens','2400']
        with memory_run([self.case],{},extra=args) as (files,writes,api,external):
            before=deepcopy(files['summary'])
            actual=deepcopy(self.case);actual['topic']=deepcopy(before[0])
            worker=review.ReviewAdapter(rules,review.Settings(max_completion_tokens=2400))
            row=worker.candidates(actual)[0]
            answer=receipt(worker,actual,row)
            opener=Mock();opener.open.return_value=raw_reply(answer)
            with patch.dict('os.environ',{'OPENAI_API_KEY':self.key}),patch.object(live,'build_opener',return_value=opener):
                rules.main()
            before[0]['next_signal']=self.case['evidence'][0]['grounded_sentence']
            self.assertEqual(before,files['summary'])
            self.assertEqual(['summary','report'],writes)
            self.assertEqual(1,files['report'][0]['model_call_count'])
            opener.open.assert_called_once();api.assert_not_called();external.assert_not_called()
        # Resource scope releases locks even when main() is invoked in-process.
        with self.session(max_calls=1) as resumed:self.assertEqual(1,resumed.budget.calls)

    def test_live_flag_requires_both_budget_limits(self):
        with memory_run([self.case],{},extra=['--checkpoint-live']) as (_,writes,_,_):
            with patch.object(live,'CheckpointSession',side_effect=AssertionError('No authorized limit')):
                with self.assertRaises(SystemExit) as error:rules.main()
            self.assertEqual(2,error.exception.code)
            self.assertEqual([],writes)


if __name__=='__main__':
    unittest.main()
