"""Approved edition greetings, sign-offs, and banner palettes; no external calls."""
from datetime import date

# Monday through Sunday. Keep this mapping fixed for archived editions.
# Each pair is (banner background, accent); cream lettering stays unchanged.
WEEKDAY_PALETTES = (
    ("#1644BD", "#FFE578"),  # Cobalt & Lemon
    ("#6424B5", "#D6F779"),  # Violet & Lime
    ("#A61B60", "#FFD0E5"),  # Raspberry & Blush
    ("#075B91", "#9CEAFF"),  # Ocean & Ice
    ("#075E63", "#FFE397"),  # Lagoon & Buttercream
    ("#076247", "#A3F1C6"),  # Emerald & Mint
    ("#3832AC", "#E2CAFF"),  # Indigo & Lavender
)

# Same weekdays, softened toward #242424: backgrounds 15%, accents 10%.
DARK_WEEKDAY_PALETTES = (
    ("#183FA6", "#E9D270"),  # Cobalt & Lemon
    ("#5A249F", "#C4E271"),  # Violet & Lime
    ("#931C57", "#E9BFD2"),  # Raspberry & Blush
    ("#0B5381", "#90D6E9"),  # Ocean & Ice
    ("#0B555A", "#E9D08C"),  # Lagoon & Buttercream
    ("#0B5942", "#96DDB6"),  # Emerald & Mint
    ("#353098", "#CFB9E9"),  # Indigo & Lavender
)


def banner_palette_for_date(edition_date: str, theme: str = "light") -> tuple[str, str]:
    """Use the edition's weekday; the display theme only changes its shades."""
    palettes = DARK_WEEKDAY_PALETTES if theme == "dark" else WEEKDAY_PALETTES
    try:
        weekday = date.fromisoformat(edition_date).weekday()
    except (ValueError, TypeError):
        return palettes[0]
    return palettes[weekday]


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


SIGN_OFFS = (
    "That’s enough news for now. Take a little time for yourself.",
    "A little more context, a little less reason to keep scrolling.",
    "That’s the short version. The rest of your day is yours.",
    "You can close this tab. Caring doesn’t require constant updates.",
    "The news keeps moving. You’re allowed to pause.",
    "That’s our update. No refresh needed.",
    "Now there’s room for something other than headlines.",
    "Enough headlines for a while. Take some breathing room.",
    "You don’t have to follow every development to care.",
    "The scroll stops here. Thanks for spending a little time with us.",
    "Whether you read every word or skimmed, that’s enough for now.",
    "You’ve given the news some attention. It doesn’t need all of it.",
    "Staying informed can include knowing when to stop.",
    "End of edition. No endless feed attached.",
    "Some news is a lot to take in. Give yourself a little space.",
    "That’s what we have for today. Come back when you have room.",
    "You’ve checked in with the world. Now back to yours.",
    "No need to keep a tab open on everything.",
    "The headlines can wait while you do something else.",
    "You can step away and still care about what happens next.",
    "That’s the roundup. Your attention is yours again.",
    "A few minutes of news can be enough.",
    "We promised a stopping point. Here it is.",
    "Close the tab whenever you’re ready. We’ll be here another day.",
    "That’s enough checking in for now.",
    "There’s always more news. There doesn’t have to be more scrolling.",
    "Take what was useful. Give yourself time with the rest.",
    "The world is worth understanding. Your day is worth returning to.",
    "A little distance from the news is welcome, too.",
    "Thanks for reading. No need to stay connected all day.",
    "That’s all from us. Leave a little bandwidth for yourself.",
)


def sign_off_for_date(edition_date: str) -> str:
    """Keep the approved sign-off fixed per edition in a continuous 31-day cycle."""
    try:
        day = date.fromisoformat(edition_date)
    except (ValueError, TypeError):
        return SIGN_OFFS[0]
    return SIGN_OFFS[(day - ANCHOR).days % len(SIGN_OFFS)]


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
