"""Approved introductions, in a stable 31-day rotation with no external calls."""
from datetime import date

# Keep this order and anchor fixed so archived editions retain their phrase.
ANCHOR = date(2026, 10, 14)
PHRASES = (
    "Stay informed without staying online. Here’s the short version. If today’s already full, we’ll be here tomorrow.",
    "A little news, then back to your day. No endless feed. No need to check again at lunch.",
    "The news can be heavy. We’ll keep the reading light. Take in what you have room for.",
    "You don’t have to keep refreshing to keep up. Here are the basics, whenever you have the bandwidth.",
    "Enough news to stay informed. Enough space to step away. Some days, stepping away comes first.",
    "Catch up without getting caught in the feed. We’ll keep it short.",
    "The world has updates. You don’t need to follow them all. Here’s the short version.",
    "Being informed doesn’t have to be an all-day job. A few minutes here, then you’re good to go.",
    "Some days, the news asks too much. Read what you can, or give yourself the day off.",
    "A few minutes of news can be enough. No need to keep the feed open.",
    "Your attention deserves a break. We’ll keep the news brief so there’s room for the rest of your day.",
    "A short read, with an actual ending. No more news required today.",
    "Read a little. Step away. Caring about the world doesn’t mean watching it around the clock.",
    "Today’s news, in a few minutes. If those minutes are better spent elsewhere, we understand.",
    "Here’s enough to catch up without settling in for a long scroll.",
    "Whenever this finds you, we’ll keep it brief. If today’s already full, come back another day.",
    "A little context without a day of scrolling. Take what’s useful, then carry on.",
    "Keeping up shouldn’t leave you worn out. Here’s the short version.",
    "There’s more to your day than the news. Let’s leave room for it.",
    "Check in with the world, then check out for a while. You don’t have to follow every development.",
    "The news can be difficult. You don’t have to read it all to stay informed.",
    "One short read. Fewer reasons to keep checking your phone.",
    "You don’t owe the news your whole morning. A few minutes will do.",
    "Stay curious. Keep some breathing room. The news doesn’t need all your attention.",
    "We’ll cover the basics. You decide whether today has room for them.",
    "Read enough to feel informed, without spending the day in the headlines.",
    "Here’s a short update, without the constant updates.",
    "A brief look at what’s happening, then back to what’s happening in your life.",
    "Some days have room for the news. Some don’t. We’ll keep it short whenever you’re ready.",
    "Staying informed shouldn’t mean feeling overwhelmed. Read a little, then take a break.",
    "That endless feed can wait. Here’s a little news with a clear stopping point.",
)


def introduction_for_date(edition_date: str) -> str:
    """Use edition date, never the current clock or page refresh count.

    Continues across month/year boundaries; each phrase repeats every 31 days.
    Invalid dates get a stable fallback without changing existing route handling.
    """
    try:
        day = date.fromisoformat(edition_date)
    except (ValueError, TypeError):
        return PHRASES[0]
    return PHRASES[(day - ANCHOR).days % len(PHRASES)]
