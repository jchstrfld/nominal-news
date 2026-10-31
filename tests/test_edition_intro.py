import unittest
from datetime import date, timedelta

from edition_intro import PHRASES, introduction_for_date


class EditionIntroTests(unittest.TestCase):
    def test_unique_month_and_repeat_interval_across_calendar_boundaries(self):
        self.assertEqual(len(PHRASES), 31)
        self.assertEqual(len(set(PHRASES)), 31)
        for start in (date(2026, 10, 14), date(2026, 12, 20), date(2028, 2, 15)):
            phrases = [introduction_for_date((start + timedelta(days=i)).isoformat())
                       for i in range(62)]
            self.assertEqual(len(set(phrases[:31])), 31)
            self.assertEqual(phrases[:31], phrases[31:])

    def test_reopening_archive_is_stable(self):
        original = introduction_for_date('2026-10-14')
        for other_date in ('2027-01-01', '2026-10-13', '2028-02-29'):
            introduction_for_date(other_date)
            self.assertEqual(introduction_for_date('2026-10-14'), original)
        self.assertEqual(original, PHRASES[0])

    def test_invalid_date_has_stable_fallback(self):
        for invalid in ('', 'not-a-date', '2026-02-30', None):
            self.assertEqual(introduction_for_date(invalid), PHRASES[0])


if __name__ == '__main__':
    unittest.main()
