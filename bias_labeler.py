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
    "centre-left": "Left",
    "center": "Center",
    "centre": "Center",
    "neutral": "Center",
    "right": "Right",
    "center-right": "Right",
    "centre-right": "Right",
    "far-right": "Far Right",
    "far right": "Far Right",
}

def canonicalize(label: str, default="Center") -> str:
    if not label:
        return default
    key = label.strip().lower()
    return CANONICAL.get(key, default)

# ----------------------------
# Your original NAME-BASED map (kept intact)
# Values may be lowercased or dashed; we canonicalize on output.
# ----------------------------
bias_map = {
    # Far‑Left
    "Democracy Now": "far-left",
    "Jacobin": "far-left",
    "The Intercept": "far-left",
    "SF Chronicle": "far-left",         # Manually set via AllSides

    # Left
    "CNN": "left",
    "The Guardian": "left",
    "MSNBC": "left",
    "HuffPost": "left",
    "New York Times": "left",
    "NPR": "left",
    "The Independent": "left",
    "Politico": "left",
    "Rolling Stone": "left",
    "Amnesty International": "left",
    "Human Rights Watch": "left",
    "The Baffler": "left",
    "Star Tribune": "left",             # Manually via AllSides
    "AJC": "left",                      # Manually via AllSides
    "Philadelphia Inquirer": "left",    # Manually via AllSides
    "City Limits": "left",              # Manually via AllSides
    "Chicago Reader": "left",           # Manually via mediabiasfactcheck.com
    "Grist": "left",                    # Manually via AllSides
    "Global Voices": "left",            # Manually via mediabiasfactcheck.com
    "Westword": "center",               # Manually via AllSides

    # Center
    "Reuters": "center",
    "Associated Press": "center",
    "BBC News": "center",
    "DW": "center",
    "Deutsche Welle": "center",
    "Al Jazeera": "center",
    "AFP": "center",
    "NPR News": "center",
    "USA Today": "center",
    "Axios": "center",
    "The Hill": "center",
    "NBC News": "center",
    "Pew Research Center": "center",
    "RAND Corporation": "center",
    "Urban Institute": "center",
    "KFF": "center",
    "Science News": "center",
    "Johns Hopkins Hub": "center",
    "The Conversation": "center",
    "Harvard Gazette": "center",
    "Princeton News": "center",
    "University of Michigan News": "center",
    "Penn State News": "center",
    "UC Berkeley News": "center",
    "University of Chicago News": "center",
    "UNC Chapel Hill News": "center",
    "University of Washington News": "center",
    "University of Texas News": "center",
    "University at Buffalo News": "center",
    "World Health Organization": "center",
    "UN News": "center",
    "OECD": "center",
    "NASA": "center",
    "NOAA": "center",
    "CDC": "center",
    "Foreign Policy": "center",
    "Smithsonian Magazine": "center",
    "Stat News": "center",
    "Education Week": "center",
    "Inside Climate News": "center",
    "Pitchfork": "center",                      # Unable to source for bias
    "AllAfrica": "center",

    # Local / Regional (defaulted center)
    "WGN": "center",
    "KHOU": "center",
    "KDFW": "center",
    "KABC": "center",
    "WXYZ": "center",
    "WPVI": "center",
    "KIRO": "center",
    "WCVB": "center",
    "KTLA": "center",
    "Chicago Sun-Times": "center",
    "Houston Chronicle": "center",
    "Newsday": "center",
    "NBC4 Washington": "center",
    "WUSA9": "center",
    "KIRO7": "center",
    "Billy Penn": "center",                 # Unable to source for bias
    "Phoenix New Times": "center",
    "Gothamist": "center",
    "Willamette Week": "center",            # Unable to source for bias

    # Right
    "The Wall Street Journal": "right",
    "Fox News": "right",
    "Washington Examiner": "right",
    "WSJ.com": "right",
    "AEI": "right",
    "City Journal": "right",
    "RealClearPolitics": "right",
    "The Dispatch": "right",
    "The Bulwark": "right",
    "National Review": "right",
    "Commentary Magazine": "right",
    "Washington Times": "right",

    # Far‑Right
    "Breitbart": "far-right",
    "OANN": "far-right",
    "Epoch Times": "far-right",
    "The Gateway Pundit": "far-right"
}

# Normalize NAME keys for quick lookup by source name
name_bias_map = {k.strip().lower(): v for k, v in bias_map.items()}

# ----------------------------
# Domain aliases (domain -> Display Name)
# Kept exactly as you provided
# ----------------------------
domain_aliases = {
    "bbc.co.uk": "BBC News",
    "cnn.com": "CNN",
    "msnbc.com": "MSNBC",
    "theguardian.com": "The Guardian",
    "huffpost.com": "HuffPost",
    "nytimes.com": "New York Times",
    "npr.org": "NPR",
    "usatoday.com": "USA Today",
    "axios.com": "Axios",
    "thehill.com": "The Hill",
    "nbcnews.com": "NBC News",
    "pewresearch.org": "Pew Research Center",
    "rand.org": "RAND Corporation",
    "urban.org": "Urban Institute",
    "kff.org": "Kaiser Family Foundation (KFF)",
    "sciencenews.org": "Science News",
    "hub.jhu.edu": "Johns Hopkins Hub",
    "theconversation.com": "The Conversation",
    "news.harvard.edu": "Harvard Gazette",
    "princeton.edu": "Princeton News",
    "news.umich.edu": "University of Michigan News",
    "news.psu.edu": "Penn State News",
    "news.berkeley.edu": "UC Berkeley News",
    "news.uchicago.edu": "University of Chicago News",
    "unc.edu": "UNC Chapel Hill News",
    "washington.edu": "University of Washington News",
    "utexas.edu": "University of Texas News",
    "buffalo.edu": "University at Buffalo News",
    "worldhealthorganization.int": "World Health Organization",
    "un.org": "United Nations News",
    "oecd.org": "OECD",
    "nasa.gov": "NASA",
    "noaa.gov": "NOAA",
    "cdc.gov": "CDC",
    "foreignpolicy.com": "Foreign Policy",
    "globalvoices.org": "Global Voices",
    "smithsonianmag.com": "Smithsonian Magazine",
    "statnews.com": "Stat News",
    "edweek.org": "Education Week",
    "insideclimatenews.org": "Inside Climate News",
    "grist.org": "Grist",
    "pitchfork.com": "Pitchfork",
    "allafrica.com": "AllAfrica",
    "wgntv.com": "WGN - Chicago",
    "khou.com": "KHOU - Houston",
    "fox4news.com": "KDFW - Dallas/Fort Worth",
    "kabc.com": "KABC - Los Angeles",
    "wxyz.com": "WXYZ - Detroit",
    "wcvb.com": "WCVB - Boston",
    "ktla.com": "KTLA - Los Angeles",
    "chicago.suntimes.com": "Chicago Sun-Times",
    "chron.com": "Houston Chronicle",
    "newsday.com": "Newsday",
    "nbcwashington.com": "NBC4 Washington",
    "wusa9.com": "WUSA9 - Washington, D.C.",
    "kiro7.com": "KIRO - Seattle",
    "westword.com": "Denver Westword",
    "citylimits.org": "City Limits",
    "billypenn.com": "Billy Penn",
    "phoenixnewtimes.com": "Phoenix New Times",
    "chicagoreader.com": "Chicago Reader",
    "gothamist.com": "Gothamist",
    "startribune.com": "Star Tribune",
    "wweek.com": "Willamette Week",
    "ajc.com": "Atlanta Journal-Constitution (AJC)",
    "sfchronicle.com": "San Francisco Chronicle",
    "inquirer.com": "Philadelphia Inquirer",
    "aei.org": "American Enterprise Institute (AEI)",
    "city-journal.org": "City Journal",
    "realclearpolitics.com": "RealClearPolitics",
    "thedispatch.com": "The Dispatch",
    "thebulwark.com": "The Bulwark",
    "nationalreview.com": "National Review",
    "commentary.org": "Commentary Magazine",
    "washingtontimes.com": "Washington Times",
    "breitbart.com": "Breitbart",
    "oann.com": "One Amereica News Network (OANN)",
    "theepochtimes.com": "Epoch Times",
    "thegatewaypundit.com": "The Gateway Pundit"
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
    "latimes.com": "Left",
    "newyorker.com": "Left",
    "space.com": "Center",
    "theatlantic.com": "Left",
    "wired.com": "Center",
    "news10.com": "Center",
    "westword.com": "Left",
    "wgntv.com": "Center", 
    "abcnews.go.com": "Center",
    "cbsnews.com": "Center",
    "aei.org": "Right",
    "techradar.com": "Center",
    "kdvr.com": "Center",
    "wcvb.com": "Center",
    "news.utexas.edu": "Center",
    "uncnews.unc.edu": "Center",
    "science.org": "Center",
    "marketwatch.com": "Right",
    "fortune.com": "Center",
    "nautil.us": "Left",
    "kabc.com": "Center",
    "engineering.berkeley.edu": "Left",
    "law.berkeley.edu": "Left",
    "journalism.berkeley.edu": "Left",
    "cdss.berkeley.edu": "Left",
    "artshumanities.berkeley.edu": "Left",
    "lib.berkeley.edu": "Left",
    # Add more as needed...
}

def get_domain(url: str) -> str | None:
    try:
        return urlparse(url).netloc.replace("www.", "").lower()
    except Exception:
        return None

def lookup_bias_by_name(source_name: str) -> str | None:
    if not source_name:
        return None
    raw = name_bias_map.get(source_name.strip().lower())
    if raw:
        return canonicalize(raw)
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

    # Collect unmapped for reporting
    missing_items = []
    missing_map = defaultdict(int)

    for article in articles:
        source_name = article.get("source", "")
        url = article.get("url", "")

        bias = lookup_bias_by_name(source_name)
        if not bias:
            bias = lookup_bias_by_domain(url)

        if not bias:
            domain = get_domain(url) or ""
            id_for_log = domain or (source_name.strip().lower() if source_name else "unknown")
            missing_map[id_for_log] += 1
            print(f"⚠️ Missing bias mapping for: {source_name or url or 'UNKNOWN'} — defaulting to '{default_bias}'")
            bias = default_bias

        # Ensure canonical labels always
        article["bias"] = canonicalize(bias, default=default_bias)

    # Write articles with bias
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    # Build & overwrite single unmapped report file every run
    if missing_map:
        # Convert to list of dicts sorted by count desc
        for key, count in sorted(missing_map.items(), key=lambda kv: kv[1], reverse=True):
            missing_items.append({"id": key, "count": count})
        report = {"unmapped": missing_items}
    else:
        report = {"message": "No unmapped domains in latest build"}

    with open(unmapped_output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Tagged {len(articles)} articles with bias and saved to {output_path}")
    if missing_map:
        print(f"📝 Wrote unmapped domain report → {unmapped_output} ({len(missing_items)} items)")
    else:
        print(f"📝 {unmapped_output}: No unmapped domains in latest build")

if __name__ == "__main__":
    args = sys.argv
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