# bias_labeler.py — Merged mappings, canonical labels, guaranteed bias, and unmapped export
import os
import json
from urllib.parse import urlparse
import sys
from datetime import datetime
from collections import defaultdict

# ----------------------------
# Canonical bias labels (match UI color map)
# ----------------------------
CANONICAL = {
    "far-left": "Far Left",
    "far left": "Far Left",
    "left": "Left",
    "center-left": "Left",
    "center": "Center",
    "lean-left": "Left",
    "lean left": "Left",
    "center-right": "Right",
    "right": "Right",
    "lean-right": "Right",
    "lean right": "Right",
    "far-right": "Far Right",
    "far right": "Far Right",
    "unknown": "Unknown",
    "": "Unknown",
    None: "Unknown",
}

def canonicalize(raw: str | None) -> str:
    if raw is None:
        return "Unknown"
    v = str(raw).strip().lower()
    return CANONICAL.get(v, raw.title() if raw else "Unknown")

# ----------------------------
# Original NAME-BASED map (kept intact)
# Values may be lowercased or dashed; we canonicalize on output.
# ----------------------------
bias_map = {
    # -------------------------
    # FAR-LEFT
    # -------------------------
    "AlterNet": "far-left",
    "Democracy Now": "far-left",
    "Jacobin": "far-left",

    # -------------------------
    # LEFT  (includes Lean Left)
    # -------------------------
    "AJC": "left",
    "Al Jazeera": "left",                       # often rated Lean Left
    "Amnesty International": "left",
    "The Atlantic": "left",
    "Bloomberg Opinion": "left",
    "BuzzFeed News": "left",
    "CBC News": "left",
    "Chicago Reader": "left",
    "City Limits": "left",
    "CNN": "left",
    "Current Affairs": "left",
    "Daily Mirror": "left",
    "El País": "left",
    "Global Voices": "left",
    "Grist": "left",
    "The Guardian": "left",
    "Haaretz": "left",
    "HuffPost": "left",
    "Human Rights Watch": "left",
    "Iowa Capital Dispatch": "left",
    "LA Times": "left",
    "Mediaite": "left",
    "The Nation": "left",                        # moved from far-left → left (consensus Lean Left/Left)
    "New York Times": "left",
    "NPR": "left",
    "Philadelphia Inquirer": "left",
    "Politico": "left",
    "ProPublica": "left",
    "Rolling Stone": "left",
    "Slate": "left",                             # moved from far-left → left (consensus Lean Left)
    "The Hindu": "left",
    "The Independent": "left",
    "The Intercept": "left",                     # moved from far-left → left (consensus Left/Lean Left)
    "The New Yorker": "left",
    "Time": "left",
    "Vox": "left",
    "Washington Post": "left",
    "Yahoo News": "left",

    # -------------------------
    # CENTER
    # -------------------------
    "ABC News": "center",
    "AFP": "center",
    "AP": "center",
    "Associated Press": "center",                # moved from left → center (consensus Center)
    "Axios": "center",
    "Barron's": "center",
    "BBC News": "center",
    "Bloomberg": "center",
    "Business Insider": "center",
    "CBS News": "center",
    "Christian Science Monitor": "center",
    "CNBC": "center",
    "CTV News": "center",
    "DW": "center",
    "Denver Post": "center",
    "Deutsche Welle": "center",
    "Financial Times": "center",
    "Forbes": "center",
    "France 24": "center",
    "Gothamist": "center",
    "Japan Times": "center",
    "MarketWatch": "center",
    "MSN": "center",
    "NBC News": "center",
    "NewsNation": "center",
    "Newsweek": "center",
    "New York Daily News": "center",
    "New Zealand Herald": "center",
    "Nikkei Asia": "center",
    "NPR News": "center",
    "PBS NewsHour": "center",
    "Phoenix New Times": "center",
    "Quartz": "center",
    "Reason": "center",
    "Reuters": "center",
    "RTÉ": "center",
    "Scientific American": "center",
    "Semafor": "center",
    "Sky News": "center",
    "South China Morning Post": "center",
    "STAT News": "center",
    "Straight Arrow News": "center",
    "The Economist": "center",
    "The Hill": "center",
    "The Times of India": "center",
    "USA Today": "center",
    "Wall Street Journal": "center",             # news side (opinion has separate entries below)
    "Westword": "center",
    "WGN - Chicago": "center",
    "WUSA9": "center",
    "Yahoo Finance": "center",

    # -------------------------
    # RIGHT  (includes Lean Right)
    # -------------------------
    "AEI": "right",
    "Blaze Media": "right",
    "Boston Herald": "right",
    "CBN": "right",
    "City Journal": "right",
    "Daily Caller": "right",
    "Daily Mail": "right",
    "The Daily Telegraph (Australia)": "right",
    "The Dispatch": "right",
    "The Federalist": "right",
    "Financial Post": "right",
    "Fox Business": "right",
    "Fox News": "right",
    "The Jerusalem Post": "right",
    "Just the News": "right",                    # moved from center → right (consensus Right/Lean Right)
    "National Post": "right",
    "National Review": "right",
    "Newsmax": "right",
    "New York Post": "right",
    "RealClearPolitics": "right",
    "The Spectator": "right",
    "The Straits Times": "center",
    "The Telegraph": "right",
    "The Times (UK)": "right",
    "The Wall Street Journal": "right",          # opinion page/brand
    "Upward News": "right",
    "Washington Examiner": "right",
    "Washington Free Beacon": "right",
    "Washington Times": "right",
    "ZeroHedge": "right",

    # -------------------------
    # FAR-RIGHT
    # -------------------------
    "Breitbart": "far-right",
    "OAN": "far-right",                           # moved from right → far-right (align OANN)
    "OANN": "far-right",
    "The Daily Signal": "far-right",
    "The Daily Wire": "far-right",
    "The Epoch Times": "right",                   # moved from center → right (consensus Right/Far-Right)
    "The Gateway Pundit": "far-right",
    "The Post Millennial": "far-right",

    # -------------------------
    # GLOBAL / SPECIALTY ADDS (keep conservative, many rate these as center-ish)
    # -------------------------
    "Ars Technica": "left",
    "Der Spiegel": "center",
    "El Mundo": "center",
    "Engadget": "center",
    "Financial News (Dow Jones)": "center",
    "Fortune": "center",
    "Le Figaro": "right",
    "Le Monde": "left",
    "NPR Science": "center",
    "STAT": "center",
    "TechCrunch": "center",
    "The Globe and Mail": "center",
    "The Information": "center",
    "The Japan News (Yomiuri)": "right",
    "The Times of Israel": "center",
    "Wired": "left"
}

# ----------------------------
# Provide a DOMAIN→display-name map.
# Used when converting a domain to the name-based map above (name_bias_map)
# ----------------------------
domain_aliases = {
    # --- A ---
    "abc.net.au": "ABC News",
    "abcnews.go.com": "ABC News",
    "aei.org": "American Enterprise Institute (AEI)",
    "ajc.com": "Atlanta Journal-Constitution (AJC)",
    "aljazeera.com": "Al Jazeera",
    "allafrica.com": "AllAfrica",
    "apnews.com": "Associated Press",
    "arstechnica.com": "Ars Technica",
    "asia.nikkei.com": "Nikkei Asia",
    "axios.com": "Axios",

    # --- B ---
    "barrons.com": "Barron's",
    "bbc.co.uk": "BBC News",
    "bbc.com": "BBC News",
    "billypenn.com": "Billy Penn",
    "bloomberg.com": "Bloomberg",
    "bostonherald.com": "Boston Herald",
    "breitbart.com": "Breitbart",
    "businessinsider.com": "Business Insider",
    "buzzfeednews.com": "BuzzFeed News",

    # --- C ---
    "cbc.ca": "CBC News",
    "cbsnews.com": "CBS News",
    "cbn.com": "CBN",
    "chicago.suntimes.com": "Chicago Sun-Times",
    "chicagoreader.com": "Chicago Reader",
    "chron.com": "Houston Chronicle",
    "city-journal.org": "City Journal",
    "cnbc.com": "CNBC",
    "cnn.com": "CNN",
    "commentary.org": "Commentary Magazine",
    "csmonitor.com": "Christian Science Monitor",
    "ctvnews.ca": "CTV News",
    "currentaffairs.org": "Current Affairs",

    # --- D ---
    "dailycaller.com": "Daily Caller",
    "dailytelegraph.com.au": "The Daily Telegraph (Australia)",
    "dailymail.co.uk": "Daily Mail",
    "dailymaillive.co.uk": "Daily Mail",
    "dailysignal.com": "The Daily Signal",
    "dailywire.com": "The Daily Wire",
    "denverpost.com": "Denver Post",
    "dw.com": "DW (Deutsche Welle)",

    # --- E ---
    "economist.com": "The Economist",
    "elmundo.es": "El Mundo",
    "elpais.com": "El País",
    "engadget.com": "Engadget",
    "euronews.com": "Euronews",

    # --- F ---
    "finance.yahoo.com": "Yahoo Finance",
    "financialnews.com": "Financial News (Dow Jones)",
    "financialpost.com": "Financial Post",
    "fnlondon.com": "Financial News (Dow Jones)",
    "forbes.com": "Forbes",
    "fortune.com": "Fortune",
    "foxbusiness.com": "Fox Business",
    "foxnews.com": "Fox News",
    "france24.com": "France 24",
    "ft.com": "Financial Times",

    # --- G ---
    "globeandmail.com": "The Globe and Mail",
    "gothamist.com": "Gothamist",
    "guardian.co.uk": "The Guardian",
    "haaretz.com": "Haaretz",

    # --- H ---
    "huffpost.com": "HuffPost",

    # --- I ---
    "ijr.com": "Independent Journal Review",
    "independent.co.uk": "The Independent",
    "informationweek.com": "InformationWeek",
    "inquirer.com": "Philadelphia Inquirer",
    "iowacapitaldispatch.com": "Iowa Capital Dispatch",

    # --- J ---
    "japantimes.co.jp": "Japan Times",
    "jpost.com": "The Jerusalem Post",

    # --- K ---
    "kabc.com": "KABC - Los Angeles",
    "kff.org": "Kaiser Family Foundation (KFF)",
    "kiro7.com": "KIRO7 - Seattle",
    "khou.com": "KHOU - Houston",
    "kqed.org": "KQED - San Francisco",

    # --- L ---
    "lefigaro.fr": "Le Figaro",
    "lemonde.fr": "Le Monde",
    "latimes.com": "LA Times",

    # --- M ---
    "marketwatch.com": "MarketWatch",
    "mediaite.com": "Mediaite",
    "mirror.co.uk": "Daily Mirror",
    "motherjones.com": "Mother Jones",
    "msn.com": "MSN",
    "msnbc.com": "MSNBC",

    # --- N ---
    "nationalobserver.com": "National Observer",
    "nationalpost.com": "National Post",
    "nationalreview.com": "National Review",
    "nbcnews.com": "NBC News",
    "nbcwashington.com": "NBC4 Washington",
    "newyorker.com": "The New Yorker",
    "news.sky.com": "Sky News",
    "news.yahoo.com": "Yahoo News",
    "newsday.com": "Newsday",
    "newsnationnow.com": "NewsNation",
    "newsweek.com": "Newsweek",
    "npr.org": "NPR",
    "nydailynews.com": "New York Daily News",
    "nypost.com": "New York Post",
    "nyt.com": "New York Times",
    "nytimes.com": "New York Times",
    "nzherald.co.nz": "New Zealand Herald",

    # --- O ---
    "oann.com": "One America News Network (OANN)",
    "opinionjournal.com": "The Wall Street Journal",  # opinion-branded subdomain

    # --- P ---
    "pbs.org": "PBS NewsHour",
    "pewresearch.org": "Pew Research Center",
    "phoenixnewtimes.com": "Phoenix New Times",
    "pitchfork.com": "Pitchfork",
    "politico.com": "Politico",
    "postmillennial.com": "The Post Millennial",
    "propublica.org": "ProPublica",

    # --- Q ---
    "qz.com": "Quartz",

    # --- R ---
    "rand.org": "RAND Corporation",
    "rawstory.com": "Raw Story",
    "reason.com": "Reason",
    "realclearpolitics.com": "RealClearPolitics",
    "reuters.com": "Reuters",
    "rollingstone.com": "Rolling Stone",
    "rte.ie": "RTÉ",
    "rt.com": "RT",

    # --- S ---
    "scientificamerican.com": "Scientific American",
    "scmp.com": "South China Morning Post",
    "sfchronicle.com": "San Francisco Chronicle",
    "semafor.com": "Semafor",
    "sky.com": "Sky News",
    "slate.com": "Slate",
    "spectator.co.uk": "The Spectator",
    "spiegel.de": "Der Spiegel",
    "statnews.com": "STAT News",
    "startribune.com": "Star Tribune",
    "straitstimes.com": "The Straits Times",
    "straightarrownews.com": "Straight Arrow News",

    # --- T ---
    "techcrunch.com": "TechCrunch",
    "telegraph.co.uk": "The Telegraph",
    "the-japan-news.com": "The Japan News (Yomiuri)",
    "theinformation.com": "The Information",
    "theatlantic.com": "The Atlantic",
    "thedispatch.com": "The Dispatch",
    "theepochtimes.com": "Epoch Times",
    "thegatewaypundit.com": "The Gateway Pundit",
    "theglobeandmail.com": "The Globe and Mail",
    "theguardian.com": "The Guardian",
    "thehindu.com": "The Hindu",
    "thehill.com": "The Hill",
    "thenation.com": "The Nation",
    "thetimes.co.uk": "The Times (UK)",
    "thestar.com.my": "The Star (Malaysia)",
    "timesofisrael.com": "The Times of Israel",
    "timesofindia.indiatimes.com": "The Times of India",
    "time.com": "Time",

    # --- U ---
    "upward.news": "Upward News",
    "usatoday.com": "USA Today",

    # --- V ---
    "vox.com": "Vox",
    "venturebeat.com": "VentureBeat",

    # --- W ---
    "washingtonexaminer.com": "Washington Examiner",
    "washingtonpost.com": "Washington Post",
    "washingtontimes.com": "Washington Times",
    "westword.com": "Denver Westword",
    "wgntv.com": "WGN - Chicago",
    "willametteweek.com": "Willamette Week",
    "wired.com": "Wired",
    "wsj.com": "Wall Street Journal",
    "wusa9.com": "WUSA9 - Washington, D.C.",
    "yomiuri.co.jp": "The Japan News (Yomiuri)",
}

# ----------------------------
# Internal DOMAIN overrides (priority if present, canonical labels)
# ----------------------------
domain_bias_overrides = {
    "breitbart.com": "Far Right",
    "oann.com": "Far Right",
    "theepochtimes.com": "Far Right",
    "thegatewaypundit.com": "Far Right",
    "foxnews.com": "Right",
    "abc.net.au": "Center",
    "aljazeera.com": "Left",
    "euronews.com": "Center",
    "independent.co.uk": "Left",
    "ft.com": "Center",
    "reuters.com": "Center",
    "apnews.com": "Center",
    "bbc.co.uk": "Center",
    "dw.com": "Center",
    "nbcnews.com": "Center",
    "nytimes.com": "Left",
    "theguardian.com": "Left",
    "cnn.com": "Left",
    "msnbc.com": "Left",
    "foxnews.com": "Right"
}

# ----------------------------
# Helper: extract domain from URL
# ----------------------------
def get_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

# ----------------------------
# Build a canonical NAME -> CANONICAL label map from bias_map
# ----------------------------
def build_name_bias_map():
    out = {}
    for name, raw in bias_map.items():
        out[name] = canonicalize(raw)
    return out

name_bias_map = build_name_bias_map()

# ----------------------------
# Lookup helpers
# ----------------------------
def lookup_bias_by_name(name: str) -> str | None:
    # Exact match
    raw = name_bias_map.get(name)
    if raw:
        return canonicalize(raw)
    # Case-insensitive fallback
    for k, v in name_bias_map.items():
        if k.lower() == name.lower():
            return canonicalize(v)
    # Domain-like keys in name map
    raw = name_bias_map.get(name.lower())
    if raw:
        return canonicalize(raw)
    return None

def lookup_display_name_by_domain(domain: str) -> str | None:
    # Direct domain alias
    if domain in domain_aliases:
        return domain_aliases[domain]
    # Try without leading 'www.'
    if domain.startswith("www."):
        d2 = domain[4:]
        if d2 in domain_aliases:
            return domain_aliases[d2]
    return None

def lookup_bias_by_domain(url: str) -> str | None:
    domain = get_domain(url)
    if not domain:
        return None

    # 1) Explicit domain override
    if domain in domain_bias_overrides:
        return domain_bias_overrides[domain]

    # 2) Domain alias -> display name -> name map
    alias_name = domain_aliases.get(domain)
    if alias_name:
        bias = lookup_bias_by_name(alias_name)
        if bias:
            return bias

    # 3) Try domain directly in name map (if any keys are domains)
    raw = name_bias_map.get(domain)
    if raw:
        return canonicalize(raw)

    return None

def label_article_bias(input_path, output_path, default_bias="Center", unmapped_output="unmapped_bias.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    unmapped = []
    for a in articles:
        bias = None

        # Prefer explicit bias in article if present
        if a.get("bias"):
            bias = canonicalize(a["bias"])
        else:
            # Try domain-based
            bias = lookup_bias_by_domain(a.get("url", ""))

            # Try name-based (source, outlet)
            if not bias:
                for key in ("source", "outlet", "publisher", "site_name"):
                    val = a.get(key)
                    if val:
                        bias = lookup_bias_by_name(val)
                        if bias:
                            break

        if not bias:
            bias = default_bias if default_bias else "Unknown"

        a["bias"] = bias

        # collect unmapped
        if bias == "Unknown":
            unmapped.append({
                "url": a.get("url"),
                "domain": get_domain(a.get("url", "")),
                "source": a.get("source") or a.get("outlet"),
                "title": a.get("title")
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    # Export unmapped for later inspection
    if unmapped:
        with open("unmapped_bias.json", "w", encoding="utf-8") as f:
            json.dump(unmapped, f, indent=2, ensure_ascii=False)

    print(f"✅ Labeled {len(articles)} articles with bias; {len(unmapped)} unknowns exported to unmapped_bias.json")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    args = sys.argv[1:]
    if "--date" in args:
        idx = args.index("--date") + 1
        if idx < len(args):
            date_str = args[idx]
        else:
            print("❌ No date provided after --date")
            sys.exit(1)
    else:
        date_str = datetime.today().strftime("%Y-%m-%d")

    input_path_raw = f"articles_raw_{date_str}.json"
    input_path_norm = f"articles_raw_normalized_{date_str}.json"
    input_path = input_path_norm if os.path.exists(input_path_norm) else input_path_raw

    output_path = f"articles_with_bias_{date_str}.json"
    label_article_bias(input_path, output_path)