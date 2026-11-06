"""Offline post-purification identity checks; no pipeline import or API calls."""
import ast
import copy
from datetime import datetime
import json
from pathlib import Path
import re
from itertools import permutations
import unittest
from unittest.mock import Mock
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def load_check():
    path = Path(__file__).resolve().parents[1] / 'final_cohesion_check.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'suppress_repeated_approved_events', '_possible_repeated_event'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'re': re, 'datetime': datetime, 'ENGLISH_STOP_WORDS': ENGLISH_STOP_WORDS,
             '_same_approved_event': Mock(side_effect=AssertionError('Unexpected identity check'))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['suppress_repeated_approved_events']


class ApprovedEventDuplicates(unittest.TestCase):
    def setUp(self):
        self.check = load_check()
        self.title = 'Regional transport authority approves new rail expansion after public consultation on Tuesday'

    def article(self, title=None, day='2031-04-08'):
        return {'title': title or self.title, 'published_at': day + 'T12:00:00Z'}

    def cluster(self, core, related=()):
        return {'articles': list(core), 'related_articles': list(related),
                'publication_purification': {'action': 'KEEP'}, 'attention_score': 7}

    def test_syndicated_core_already_in_verified_related_coverage(self):
        retained = self.cluster([self.article('Independent report '+str(i)) for i in range(5)], [self.article()])
        duplicate = self.cluster([self.article(self.title+s) for s in ('', ' – Daily News', ' – Regional Post')])
        data = [retained, duplicate]; before = copy.deepcopy(data)
        self.assertEqual([retained], self.check(data))
        self.assertEqual(before, data)  # Never combine or rewrite approved cores.
        self.assertEqual([retained], self.check(list(reversed(data))))

    def test_shared_entities_are_not_same_event(self):
        a = self.cluster([self.article(), self.article()])
        b = self.cluster([self.article(self.title.replace('approves', 'rejects'))]*2)
        self.assertEqual([a,b], self.check([a,b]))

    def test_shared_background_article_is_insufficient(self):
        a = self.cluster([self.article()])
        b = self.cluster([self.article(), self.article('A different hearing'), self.article('Another distinct decision')])
        self.assertEqual([a,b], self.check([a,b]))

    def test_dates_numbers_missing_dates_and_short_titles_do_not_merge(self):
        for left,right in (
            (self.article(), self.article(day='2032-04-08')),
            (self.article(), {'title':self.title}),
            (self.article('City approves 12 new rail stations after public consultation on Tuesday'),
             self.article('City approves 15 new rail stations after public consultation on Tuesday')),
            (self.article('Daily market update'), self.article('Daily market update')),
        ):
            a,b=self.cluster([left,copy.deepcopy(left)]),self.cluster([right,copy.deepcopy(right)])
            with self.subTest(right=right): self.assertEqual([a,b],self.check([a,b]))

    def test_unreviewed_groups_are_unchanged(self):
        a,b=self.cluster([self.article()]*2),self.cluster([self.article()]*2)
        b.pop('publication_purification')
        self.assertEqual([a,b],self.check([a,b]))

    def test_disjoint_article_urls_need_verified_event_identity_not_word_overlap(self):
        a = {**self.cluster([self.article('First account')]*4),
             'canonical_event': 'A passenger jet crashed near the regional airport.'}
        b = {**self.cluster([self.article('Different wording')]*4),
             'canonical_event': 'A jet crashed in a neighborhood near the regional airport.'}
        reviewer = Mock(return_value=False)
        self.check.__globals__['_same_approved_event'] = reviewer
        self.assertEqual([a,b], self.check([a,b]))
        reviewer.assert_called_once()
        reviewer.return_value = True
        before = copy.deepcopy([a,b])
        self.assertEqual([a], self.check([a,b]))
        self.assertEqual(before, [a,b])

    def test_duplicate_is_removed_before_top_ten_and_next_approved_event_fills_slot(self):
        groups = [{**self.cluster([self.article('Report '+str(i))]*4),
                   'canonical_event': 'Independent occurrence '+str(i)} for i in range(11)]
        # Reviewer confirms only this pair; other shared words are not identity.
        self.check.__globals__['_same_approved_event'] = lambda a,b: {
            a['canonical_event'], b['canonical_event']} == {'Independent occurrence 2', 'Independent occurrence 3'}
        result = self.check(groups)[:10]
        self.assertEqual(10, len(result))
        self.assertIn(groups[10], result)
        self.assertNotIn(groups[3], result)

    def test_unresolved_identity_is_not_counted_as_two_distinct_events(self):
        a = {**self.cluster([self.article('One report')]*4), 'canonical_event': 'Council approved rail extension'}
        b = {**self.cluster([self.article('Another report')]*4), 'canonical_event': 'Council approved a rail extension'}
        self.check.__globals__['_same_approved_event'] = Mock(side_effect=ValueError('Unresolved identity'))
        with self.assertRaises(ValueError): self.check([a,b])

    def test_opposite_labels_are_reviewed_using_approved_source_evidence(self):
        a = {**self.cluster([self.article('Residents win appeal against development')]*4),
             'canonical_event': 'Residents won their appeal'}
        b = {**self.cluster([dict(self.article('Developer loses building permit'),
             description='Residents won their appeal, overturning the building permit.')]*4),
             'canonical_event': 'Developer lost building permission'}
        reviewer = Mock(return_value=False)
        self.check.__globals__['_same_approved_event'] = reviewer
        self.assertEqual([a,b], self.check([a,b]))
        reviewer.assert_called_once()
        reviewer.return_value = True
        self.assertEqual([a], self.check([a,b]))

    def test_background_mention_only_requests_review_not_suppression(self):
        a = {**self.cluster([self.article('Council approves a bridge')]*4),
             'canonical_event': 'Council approves bridge'}
        b = {**self.cluster([dict(self.article('Mayor announces resignation'),
             description='Council approves bridge; mayor announces resignation afterward.')]*4),
             'canonical_event': 'Mayor announces resignation'}
        reviewer = Mock(return_value=False)
        self.check.__globals__['_same_approved_event'] = reviewer
        self.assertEqual([a,b], self.check([a,b]))
        reviewer.assert_called_once()

    def test_related_text_and_scattered_words_do_not_expand_shortlist(self):
        possible = self.check.__globals__['_possible_repeated_event']
        a = {'canonical_event': 'Council approves bridge', 'articles': []}
        b = {'canonical_event': 'Mayor announces resignation',
             'articles': [self.article('Council'), self.article('bridge')],
             'related_articles': [self.article('Council approves bridge')]}
        self.assertFalse(possible(a,b))

    def paraphrased_groups(self):
        retained = {**self.cluster([self.article('Appeal judgment')]*5),
                    'canonical_event': 'Residents won their appeal'}
        alternate = {**self.cluster([dict(self.article('Development permit annulled'),
                     description='Residents won their appeal, overturning the permit.')]*4),
                     'canonical_event': 'Developer lost building permission'}
        later = {**self.cluster([dict(self.article('Building permission withdrawn'),
                 description='The developer lost building permission.')]*3),
                 'canonical_event': 'Permit for the building was withdrawn'}
        return retained, alternate, later

    def test_removed_wording_can_shortlist_but_direct_identity_is_still_required(self):
        retained,alternate,later = self.paraphrased_groups()
        possible = self.check.__globals__['_possible_repeated_event']
        self.assertFalse(possible(retained,later))
        self.assertTrue(possible(alternate,later))
        for direct_same in (False,True):
            reviewer = Mock(side_effect=lambda a,b: a is alternate or (a is later and b is retained and direct_same))
            self.check.__globals__['_same_approved_event'] = reviewer
            before = copy.deepcopy([retained,alternate,later])
            self.assertEqual([retained] if direct_same else [retained,later],
                             self.check([retained,alternate,later]))
            self.assertEqual(before,[retained,alternate,later])
            self.assertEqual([(alternate,retained),(later,retained)],
                             [call.args for call in reviewer.call_args_list])

    def test_paraphrase_chain_is_not_accepted_transitively(self):
        retained,alternate,later = self.paraphrased_groups()
        # Even if a reviewer would match the two later labels, only a direct
        # match with the retained occurrence can remove the third group.
        reviewer = Mock(side_effect=lambda a,b: b is alternate or a is alternate)
        self.check.__globals__['_same_approved_event'] = reviewer
        self.assertEqual([retained,later],self.check([retained,alternate,later]))
        reviewer.assert_any_call(later,retained)
        self.assertNotIn((later,alternate),[call.args for call in reviewer.call_args_list])

    def test_paraphrases_are_found_independent_of_input_order(self):
        groups = self.paraphrased_groups()
        self.check.__globals__['_same_approved_event'] = Mock(return_value=True)
        for order in permutations(groups):
            self.assertEqual([groups[0]],self.check(list(order)))

    def test_uncertain_direct_match_remains_unresolved(self):
        retained,alternate,later = self.paraphrased_groups()
        def review(a,b):
            if a is later:raise ValueError('Uncertain direct occurrence')
            return True
        self.check.__globals__['_same_approved_event'] = review
        with self.assertRaisesRegex(ValueError,'Uncertain direct occurrence'):
            self.check([retained,alternate,later])

    def test_daily_loop_remembers_removed_wording_between_candidate_reviews(self):
        # Execute the real post-review loop block, not a duplicate implementation.
        path = Path(__file__).resolve().parents[1] / 'final_cohesion_check.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        loop = next(n for n in ast.walk(main) if isinstance(n,ast.For) and ast.unparse(n.target)=='(queue_pos, c)')
        start = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr)
                     and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='vetted.append')
        end = next(i for i in range(start,len(loop.body)) if isinstance(loop.body[i],ast.If)
                   and ast.unparse(loop.body[i].test)=='purity_report')
        initializers = [copy.deepcopy(n) for n in main.body if isinstance(n,ast.Assign)
                        and isinstance(n.value,ast.List) and not n.value.elts]
        wrapper = ast.parse('def replay(groups):\n    for published in groups:\n        pass\n    return vetted')
        wrapper.body[0].body[0].body = copy.deepcopy(loop.body[start:end])
        wrapper.body[0].body[0:0] = initializers
        scope={'suppress_repeated_approved_events':self.check,'print':Mock()}
        exec(compile(ast.fix_missing_locations(wrapper),str(path),'exec'),scope)
        groups = self.paraphrased_groups()
        self.check.__globals__['_same_approved_event'] = Mock(return_value=True)
        self.assertEqual([groups[0]],scope['replay'](groups))

    def test_identity_reviewer_can_cite_late_core_evidence_but_not_related(self):
        path=Path(__file__).resolve().parents[1]/'final_cohesion_check.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_same_approved_event')
        left={**self.cluster([self.article('First account')]*4+[self.article('Decisive fifth account')],
                            [self.article('Related-only statement')]),'canonical_event':'A fixed occurrence'}
        right={**self.cluster([self.article('Another account')]*2),'canonical_event':'A differently worded occurrence'}
        def review(prompt,properties,name,limit):
            data=json.loads(prompt[prompt.index('{"left":'):])
            self.assertEqual(5,len(data['left']['evidence']))
            self.assertEqual('Decisive fifth account',data['left']['evidence']['5']['title'])
            self.assertNotIn('Related-only statement',prompt)
            self.assertIn('5',properties['left_evidence_id']['enum'])
            return {'reason':'Same concrete occurrence','relation':'SAME_EVENT',
                    'left_evidence_id':'5','right_evidence_id':'2'}
        scope={'json':json,'_publication_excerpt':lambda x:x,'_event_json_request':review,
               '_remember_event_response':Mock()}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        self.assertTrue(scope['_same_approved_event'](left,right))
