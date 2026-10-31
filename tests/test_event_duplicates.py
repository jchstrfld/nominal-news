"""Offline post-purification identity checks; no pipeline import or API calls."""
import ast
import copy
from datetime import datetime
from pathlib import Path
import re
import unittest


def load_check():
    path = Path(__file__).resolve().parents[1] / 'final_cohesion_check.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'suppress_repeated_approved_events')
    scope = {'re': re, 'datetime': datetime}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[node.name]


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
