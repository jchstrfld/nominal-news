# generate_market_overview.py
# Nominal News — calm-tech Market & Economy overview.
#
# The reader-facing card is intentionally not a finance dashboard. It answers:
#   1) What is the broad market pulse?
#   2) What is the current economic backdrop (inflation, jobs, Fed policy rate)?
#   3) Why does that backdrop matter now?
#   4) What changed recently?
#   5) What is the next meaningful scheduled signal?
#
# Official quantitative context comes from FRED series sourced from the
# Federal Reserve, BLS, and BEA. Upcoming release dates come from official
# BLS, BEA, and Federal Reserve calendars. News headlines are used only as a
# secondary context layer for emerging forces such as tariffs, fiscal policy,
# energy shocks, or financial stress.

import argparse
import csv
import html
import io
import json
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import openai
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---- Data sources ----
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
BEA_RELEASE_JSON_URL = "https://apps.bea.gov/API/signup/release_dates.json"
FED_MONETARY_POLICY_URL = "https://www.federalreserve.gov/monetarypolicy.htm"

# Three broad U.S. indexes are retained only to calculate one reader-facing
# market pulse. They are not displayed individually in the calm-tech card.
MARKET_SERIES = {
    "nasdaq": "NASDAQCOM",
    "sp500": "SP500",
    "dow": "DJIA",
}

# Reader-facing / contextual economic signals.
ECON_SERIES = {
    "cpi": "CPIAUCNS",               # BLS headline CPI index (not seasonally adjusted); YoY calculated here
    "unemployment": "UNRATE",         # BLS unemployment rate
    "payrolls": "PAYEMS",             # BLS total nonfarm payroll employment, thousands
    "fed_target_lower": "DFEDTARL",   # federal funds target range lower bound
    "fed_target_upper": "DFEDTARU",   # federal funds target range upper bound
    "treasury_10y": "DGS10",          # secondary context only
    "oil_wti": "DCOILWTICO",          # secondary context only
    "real_gdp_growth": "A191RL1Q225SBEA",  # secondary context only
}

REQUEST_HEADERS = {"User-Agent": "NominalNews/1.0 market-economy-overview"}

LOCAL_EMB_MODEL = os.getenv("NN_MARKET_EMB_MODEL", "all-MiniLM-L6-v2")
MARKET_MODEL = os.getenv("NN_MARKET_MODEL", "gpt-4o-mini")
MARKET_MODEL_FALLBACK = os.getenv("NN_MARKET_MODEL_FALLBACK", "gpt-4")

ECON_SEMANTIC_MIN = float(os.getenv("NN_ECON_SEMANTIC_MIN", "0.41"))
ECON_CLASS_MARGIN = float(os.getenv("NN_ECON_CLASS_MARGIN", "0.055"))
MAX_EVIDENCE_PER_THEME = int(os.getenv("NN_ECON_MAX_PER_THEME", "2"))
MAX_EVIDENCE_TOTAL = int(os.getenv("NN_ECON_MAX_EVIDENCE", "8"))

MARKET_FLAT_THRESHOLD = float(os.getenv("NN_MARKET_FLAT_THRESHOLD", "0.15"))
MARKET_OVERALL_FLAT_THRESHOLD = float(os.getenv("NN_MARKET_OVERALL_FLAT_THRESHOLD", "0.20"))
MARKET_MIXED_ACTIVITY_THRESHOLD = float(os.getenv("NN_MARKET_MIXED_ACTIVITY_THRESHOLD", "0.55"))
MARKET_SHARP_THRESHOLD = float(os.getenv("NN_MARKET_SHARP_THRESHOLD", "1.20"))

# Stable semantic categories. These are durable economic concepts, not a
# growing blacklist of current stories, politicians, companies, or outlets.
ECON_EVIDENCE_PROTOTYPES = {
    "monetary_policy": (
        "Federal Reserve interest rate decision monetary policy inflation mandate borrowing conditions",
        "Fed officials policy rates rate hike rate cut central bank decision affecting the U.S. economy",
    ),
    "inflation_prices": (
        "U.S. inflation consumer prices producer prices price pressures cost of living CPI PCE",
        "prices accelerating cooling or remaining elevated across the U.S. economy",
    ),
    "labor_jobs": (
        "U.S. jobs employment unemployment payrolls hiring layoffs wages labor market",
        "employment report job creation job losses unemployment rate or worker demand",
    ),
    "growth_consumers": (
        "U.S. economic growth GDP consumer spending retail activity business activity recession expansion",
        "broad U.S. growth demand spending or production affecting households and businesses",
    ),
    "trade_tariffs": (
        "tariffs trade policy import taxes export restrictions supply chains affecting U.S. prices or growth",
        "major trade dispute tariff change or trade restriction with broad economic effects",
    ),
    "fiscal_debt": (
        "U.S. federal debt deficit government borrowing Treasury financing fiscal policy broad economic consequences",
        "debt ceiling government budget deficit or fiscal policy affecting rates growth or confidence",
    ),
    "energy_supply": (
        "oil gasoline natural gas energy prices supply disruption shipping disruption affecting inflation or consumers",
        "energy shock crude oil move or supply constraint with broad economic implications",
    ),
    "financial_stress": (
        "banking stress credit conditions financial crisis liquidity lending standards systemic financial risk",
        "broad credit market or banking strain that could affect the U.S. economy",
    ),
    "geopolitical_economy": (
        "war sanctions geopolitical conflict materially affecting oil trade shipping supply chains or the global economy",
        "geopolitical event with a clear economic channel such as energy trade or supply disruption",
    ),
    "market_recap": (
        "broad U.S. stock market Wall Street session major indexes market close investors reacted to economic news",
        "S&P 500 Nasdaq Dow broad U.S. equity market daily recap",
    ),
    "noise": (
        "single company earnings individual stock celebrity sports culture politics unrelated to the economy",
        "shopping deal personal finance advice product sale cryptocurrency speculation or unrelated political commentary",
        "foreign local stock market story without a meaningful U.S. or global economic connection",
    ),
}

THEME_LABELS = {
    "monetary_policy": "Fed / rates",
    "inflation_prices": "Inflation",
    "labor_jobs": "Jobs",
    "growth_consumers": "Growth / consumers",
    "trade_tariffs": "Trade / tariffs",
    "fiscal_debt": "Debt / fiscal policy",
    "energy_supply": "Energy",
    "financial_stress": "Financial stress",
    "geopolitical_economy": "Geopolitics / economy",
    "market_recap": "Markets",
}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    return p.parse_args()


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _target_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _fred_csv_url(series_id: str, start: date, end: date) -> str:
    params = {
        "id": series_id,
        "cosd": start.strftime("%Y-%m-%d"),
        "coed": end.strftime("%Y-%m-%d"),
    }
    return requests.Request("GET", FRED_CSV_URL, params=params).prepare().url


def _fetch_fred_series(series_id: str, date_str: str, lookback_days: int) -> List[Tuple[str, float]]:
    target = _target_date(date_str)
    url = _fred_csv_url(series_id, target - timedelta(days=lookback_days), target)
    r = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    r.raise_for_status()

    text = (r.text or "").strip()
    if not text:
        return []

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or len(reader.fieldnames) < 2:
        return []

    date_col = reader.fieldnames[0]
    value_col = series_id if series_id in reader.fieldnames else reader.fieldnames[1]
    rows: List[Tuple[str, float]] = []
    for row in reader:
        ds = (row.get(date_col) or "").strip()
        raw = (row.get(value_col) or "").strip()
        if not ds or not raw or raw == ".":
            continue
        try:
            datetime.strptime(ds, "%Y-%m-%d")
            rows.append((ds, float(raw)))
        except Exception:
            continue
    rows.sort(key=lambda x: x[0])
    return rows


def _latest_row(rows: List[Tuple[str, float]]) -> Optional[Tuple[str, float]]:
    return rows[-1] if rows else None


def _row_at_or_before(rows: List[Tuple[str, float]], wanted: date) -> Optional[Tuple[str, float]]:
    eligible = []
    for ds, value in rows:
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            continue
        if d <= wanted:
            eligible.append((d, ds, value))
    if not eligible:
        return None
    eligible.sort(key=lambda x: x[0])
    return eligible[-1][1], eligible[-1][2]


def _monthly_row(rows: List[Tuple[str, float]], year: int, month: int) -> Optional[Tuple[str, float]]:
    for ds, value in reversed(rows):
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            continue
        if d.year == year and d.month == month:
            return ds, value
    return None


def _shift_month(year: int, month: int, delta: int) -> Tuple[int, int]:
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def _pct_change(prev: float, current: float) -> float:
    if prev == 0:
        return 0.0
    return (current - prev) / prev * 100.0


def _direction(pct: float) -> str:
    if pct > MARKET_FLAT_THRESHOLD:
        return "up"
    if pct < -MARKET_FLAT_THRESHOLD:
        return "down"
    return "flat"


def _closest_trading_pair(rows: List[Tuple[str, float]], date_str: str) -> Optional[Tuple[float, float, str]]:
    if len(rows) < 2:
        return None
    target = _target_date(date_str)
    parsed = []
    for ds, value in rows:
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            continue
        if d <= target:
            parsed.append((d, value))
    parsed.sort(key=lambda x: x[0])
    if len(parsed) < 2:
        return None
    return parsed[-2][1], parsed[-1][1], parsed[-1][0].strftime("%Y-%m-%d")


def _overall_market_state(indexes: Dict[str, Dict[str, Any]]) -> str:
    pcts = [float(x["pct_change"]) for x in indexes.values() if x.get("pct_change") is not None]
    if len(pcts) < 2:
        return "unknown"

    avg_move = float(np.mean(pcts))
    avg_abs = float(np.mean(np.abs(pcts)))
    up_count = sum(1 for x in pcts if x > MARKET_FLAT_THRESHOLD)
    down_count = sum(1 for x in pcts if x < -MARKET_FLAT_THRESHOLD)

    if abs(avg_move) < MARKET_OVERALL_FLAT_THRESHOLD:
        if up_count and down_count and avg_abs >= MARKET_MIXED_ACTIVITY_THRESHOLD:
            return "mixed"
        return "flat"
    if avg_move >= MARKET_SHARP_THRESHOLD and up_count >= 2:
        return "strong_up"
    if avg_move <= -MARKET_SHARP_THRESHOLD and down_count >= 2:
        return "strong_down"
    if avg_move > 0 and up_count >= 2:
        return "up"
    if avg_move < 0 and down_count >= 2:
        return "down"
    return "mixed"


def _market_state_label(state: str) -> str:
    return {
        "strong_up": "Strong Gain",
        "up": "Gain",
        "flat": "Steady",
        "mixed": "Mixed",
        "down": "Decline",
        "strong_down": "Sharp Decline",
        "unknown": "Unavailable",
    }.get(state, "Unavailable")


def _market_stats(indexes: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    pcts = [float(x["pct_change"]) for x in indexes.values() if x.get("pct_change") is not None]
    if not pcts:
        return {"average_pct_change": None, "average_abs_pct_change": None}
    return {
        "average_pct_change": round(float(np.mean(pcts)), 2),
        "average_abs_pct_change": round(float(np.mean(np.abs(pcts))), 2),
    }


def _common_close_date(indexes: Dict[str, Dict[str, Any]]) -> Optional[str]:
    dates = [x.get("close_date") for x in indexes.values() if x.get("close_date")]
    return Counter(dates).most_common(1)[0][0] if dates else None


def _human_date(ds: Optional[str]) -> str:
    if not ds:
        return ""
    try:
        d = datetime.strptime(ds, "%Y-%m-%d")
        return d.strftime("%b %d").replace(" 0", " ")
    except Exception:
        return ds


def _human_date_with_weekday(ds: Optional[str]) -> str:
    if not ds:
        return ""
    try:
        d = datetime.strptime(ds, "%Y-%m-%d")
        return d.strftime("%a, %b %d").replace(" 0", " ")
    except Exception:
        return ds


def _build_market_context(date_str: str) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, Optional[float]], Optional[str]]:
    indexes: Dict[str, Dict[str, Any]] = {}
    for key, series_id in MARKET_SERIES.items():
        try:
            rows = _fetch_fred_series(series_id, date_str, 20)
            pair = _closest_trading_pair(rows, date_str)
            if not pair:
                raise RuntimeError("No trading pair found")
            prev_close, close, close_date = pair
            pct = round(_pct_change(prev_close, close), 2)
            indexes[key] = {
                "series_id": series_id,
                "close_date": close_date,
                "pct_change": pct,
                "direction": _direction(pct),
                "source": "FRED",
            }
        except Exception as exc:
            indexes[key] = {
                "series_id": series_id,
                "close_date": None,
                "pct_change": None,
                "direction": "unknown",
                "source": "FRED",
                "error": str(exc),
            }
    state = _overall_market_state(indexes)
    return indexes, state, _market_stats(indexes), _common_close_date(indexes)


def _build_inflation_signal(date_str: str) -> Dict[str, Any]:
    result = {
        "series_id": ECON_SERIES["cpi"],
        "source": "BLS via FRED",
        "value": None,
        "trend": "Unavailable",
        "trend_direction": "flat",
        "trend_tone": "neutral",
        "trend_label": "Trend unavailable",
        "display_value": "Unavailable",
        "detail": "",
    }
    try:
        rows = _fetch_fred_series(ECON_SERIES["cpi"], date_str, 520)
        latest = _latest_row(rows)
        if not latest:
            return result
        latest_date = datetime.strptime(latest[0], "%Y-%m-%d").date()
        prev_year, prev_month = _shift_month(latest_date.year, latest_date.month, -12)
        base = _monthly_row(rows, prev_year, prev_month)
        if not base:
            return result
        yoy = _pct_change(base[1], latest[1])

        trend = "About the same as three months ago"
        trend_direction = "flat"
        trend_tone = "neutral"
        trend_label = "Inflation broadly stable"
        trend_delta = None
        comparison_value = None
        comparison_text = ""
        old_year, old_month = _shift_month(latest_date.year, latest_date.month, -3)
        old = _monthly_row(rows, old_year, old_month)
        old_base_year, old_base_month = _shift_month(old_year, old_month, -12)
        old_base = _monthly_row(rows, old_base_year, old_base_month)
        if old and old_base:
            old_yoy = _pct_change(old_base[1], old[1])
            comparison_value = round(old_yoy, 1)
            trend_delta = yoy - old_yoy
            if trend_delta <= -0.20:
                trend = "Lower than three months ago"
                trend_direction = "down"
                # Falling inflation generally eases household price pressure,
                # but very low/negative inflation is not automatically positive.
                trend_tone = "positive" if yoy >= 1.0 else "neutral"
                trend_label = "Inflation trending down"
                comparison_text = f"Down from {old_yoy:.1f}% three months ago"
            elif trend_delta >= 0.20:
                trend = "Higher than three months ago"
                trend_direction = "up"
                trend_tone = "negative" if yoy >= 2.0 else "neutral"
                trend_label = "Inflation trending up"
                comparison_text = f"Up from {old_yoy:.1f}% three months ago"
            else:
                comparison_text = f"About the same as {old_yoy:.1f}% three months ago"

        result.update({
            "value": round(yoy, 1),
            "formatted": f"Prices Up {yoy:.1f}% Over Past Year",
            "short_value": f"{yoy:.1f}%",
            "display_value": f"Prices Up {yoy:.1f}% Over Past Year",
            "detail": comparison_text,
            "as_of": latest[0],
            "trend": trend,
            "trend_direction": trend_direction,
            "trend_tone": trend_tone,
            "trend_label": trend_label,
            "trend_delta_3m_pp": round(trend_delta, 2) if trend_delta is not None else None,
            "comparison_value_3m": comparison_value,
        })
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _build_jobs_signal(date_str: str) -> Dict[str, Any]:
    result = {
        "series_id": ECON_SERIES["unemployment"],
        "payroll_series_id": ECON_SERIES["payrolls"],
        "source": "BLS via FRED",
        "value": None,
        "trend": "Unavailable",
        "trend_direction": "flat",
        "trend_tone": "neutral",
        "trend_label": "Trend unavailable",
        "display_value": "Unavailable",
        "detail": "",
    }
    try:
        rows = _fetch_fred_series(ECON_SERIES["unemployment"], date_str, 220)
        latest = _latest_row(rows)
        if not latest:
            return result
        latest_date = datetime.strptime(latest[0], "%Y-%m-%d").date()
        old_year, old_month = _shift_month(latest_date.year, latest_date.month, -3)
        old = _monthly_row(rows, old_year, old_month)
        delta = latest[1] - old[1] if old else None
        trend = "Unemployment about the same as three months ago"
        trend_direction = "flat"
        trend_tone = "neutral"
        trend_label = "Unemployment broadly stable"
        unemployment_comparison = ""
        if old:
            unemployment_comparison = f"About the same as {old[1]:.1f}% three months ago"
        if delta is not None:
            if delta >= 0.20:
                trend = "Unemployment higher than three months ago"
                trend_direction = "up"
                trend_tone = "negative"
                trend_label = "Unemployment trending up"
                unemployment_comparison = f"Up from {old[1]:.1f}% three months ago"
            elif delta <= -0.20:
                trend = "Unemployment lower than three months ago"
                trend_direction = "down"
                trend_tone = "positive"
                trend_label = "Unemployment trending down"
                unemployment_comparison = f"Down from {old[1]:.1f}% three months ago"

        payroll_rows = _fetch_fred_series(ECON_SERIES["payrolls"], date_str, 220)
        payroll_latest = _latest_row(payroll_rows)
        payroll_change = None
        payroll_as_of = None
        detail = ""
        if payroll_latest and len(payroll_rows) >= 2:
            payroll_as_of = payroll_latest[0]
            payroll_change = payroll_latest[1] - payroll_rows[-2][1]
            rounded = int(round(abs(payroll_change)))
            if payroll_change > 0:
                detail = f"{rounded:,}K jobs added last month"
            elif payroll_change < 0:
                detail = f"{rounded:,}K jobs lost last month"
            else:
                detail = "Payroll employment was broadly stable last month"

        result.update({
            "value": round(latest[1], 1),
            "formatted": f"{latest[1]:.1f}% Unemployment",
            "short_value": f"{latest[1]:.1f}% unemployment",
            "display_value": f"{latest[1]:.1f}% Unemployment",
            "detail": unemployment_comparison or detail,
            "as_of": latest[0],
            "trend": trend,
            "trend_direction": trend_direction,
            "trend_tone": trend_tone,
            "trend_label": trend_label,
            "trend_delta_3m_pp": round(delta, 2) if delta is not None else None,
            "payroll_change_k": round(payroll_change, 0) if payroll_change is not None else None,
            "payroll_as_of": payroll_as_of,
        })
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _build_rate_signal(date_str: str) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
    """Build the Fed policy-rate signal from the actual target range.

    The reader-facing value uses the Federal Reserve's stated target range.
    The midpoint is retained internally only for comparing policy changes over
    time, so the UI does not imply that the Fed publishes a single policy rate.
    """
    result = {
        "series_id": f"{ECON_SERIES['fed_target_lower']}+{ECON_SERIES['fed_target_upper']}",
        "source": "Federal Reserve via FRED",
        "value": None,
        "trend": "Unavailable",
        "trend_direction": "flat",
        "trend_tone": "neutral",
        "trend_label": "Trend unavailable",
        "display_value": "Unavailable",
        "detail": "",
    }
    rows: List[Tuple[str, float]] = []
    try:
        lower_rows = _fetch_fred_series(ECON_SERIES["fed_target_lower"], date_str, 80)
        upper_rows = _fetch_fred_series(ECON_SERIES["fed_target_upper"], date_str, 80)
        lower_by_date = dict(lower_rows)
        upper_by_date = dict(upper_rows)
        common_dates = sorted(set(lower_by_date) & set(upper_by_date))
        rows = [
            (ds, (float(lower_by_date[ds]) + float(upper_by_date[ds])) / 2.0)
            for ds in common_dates
        ]
        latest = _latest_row(rows)
        if not latest:
            return result, rows

        latest_date = datetime.strptime(latest[0], "%Y-%m-%d").date()
        old = _row_at_or_before(rows, latest_date - timedelta(days=30))
        delta = latest[1] - old[1] if old else None

        trend = "Unchanged in 30 days"
        trend_direction = "flat"
        trend_tone = "neutral"
        trend_label = "Interest rates unchanged"
        detail = "Fed Rate Unchanged in 30 Days"

        if delta is not None:
            if delta >= 0.10:
                trend = "Higher than 30 days ago"
                trend_direction = "up"
                trend_tone = "negative"
                trend_label = "Interest rates trending up"
                detail = f"Fed Rate Up {abs(delta):.2f} Points in 30 Days"
            elif delta <= -0.10:
                trend = "Lower than 30 days ago"
                trend_direction = "down"
                trend_tone = "positive"
                trend_label = "Interest rates trending down"
                detail = f"Fed Rate Down {abs(delta):.2f} Points in 30 Days"

        latest_lower = lower_by_date.get(latest[0])
        latest_upper = upper_by_date.get(latest[0])
        lower_value = round(float(latest_lower), 2) if latest_lower is not None else None
        upper_value = round(float(latest_upper), 2) if latest_upper is not None else None
        range_display = (
            f"{lower_value:.2f}–{upper_value:.2f}%"
            if lower_value is not None and upper_value is not None
            else f"{latest[1]:.2f}%"
        )

        result.update({
            # Keep the midpoint numeric internally for change calculations and
            # backwards compatibility. Reader-facing fields use the Fed's range.
            "value": round(latest[1], 2),
            "formatted": f"{range_display} Fed target range",
            "short_value": range_display,
            "display_value": range_display,
            "detail": detail,
            "as_of": latest[0],
            "trend": trend,
            "trend_direction": trend_direction,
            "trend_tone": trend_tone,
            "trend_label": trend_label,
            "trend_delta_30d_pp": round(delta, 2) if delta is not None else None,
            "target_range_lower": lower_value,
            "target_range_upper": upper_value,
        })
    except Exception as exc:
        result["error"] = str(exc)
    return result, rows


def _build_secondary_context(date_str: str) -> Dict[str, Dict[str, Any]]:
    context: Dict[str, Dict[str, Any]] = {}

    for key, lookback, compare_days in (
        ("treasury_10y", 80, 30),
        ("oil_wti", 80, 30),
    ):
        series_id = ECON_SERIES[key]
        obj = {"series_id": series_id, "value": None, "source": "FRED"}
        try:
            rows = _fetch_fred_series(series_id, date_str, lookback)
            latest = _latest_row(rows)
            if latest:
                latest_date = datetime.strptime(latest[0], "%Y-%m-%d").date()
                old = _row_at_or_before(rows, latest_date - timedelta(days=compare_days))
                change = _pct_change(old[1], latest[1]) if old and key == "oil_wti" else (latest[1] - old[1] if old else None)
                obj.update({
                    "value": round(latest[1], 2),
                    "as_of": latest[0],
                    "change_30d": round(change, 2) if change is not None else None,
                })
        except Exception as exc:
            obj["error"] = str(exc)
        context[key] = obj

    gdp = {"series_id": ECON_SERIES["real_gdp_growth"], "value": None, "source": "BEA via FRED"}
    try:
        rows = _fetch_fred_series(ECON_SERIES["real_gdp_growth"], date_str, 500)
        latest = _latest_row(rows)
        if latest:
            gdp.update({"value": round(latest[1], 1), "as_of": latest[0]})
    except Exception as exc:
        gdp["error"] = str(exc)
    context["real_gdp_growth"] = gdp

    return context


def _unfold_ics(text: str) -> List[str]:
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines: List[str] = []
    for line in raw_lines:
        if line.startswith((" ", "\t")) and lines:
            lines[-1] += line[1:]
        else:
            lines.append(line)
    return lines


def _parse_ics_date(raw: str) -> Optional[date]:
    raw = (raw or "").strip()
    m = re.search(r"(\d{8})", raw)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except Exception:
        return None


def _fetch_bls_events() -> List[Dict[str, Any]]:
    try:
        r = requests.get(BLS_ICS_URL, headers=REQUEST_HEADERS, timeout=20)
        r.raise_for_status()
    except Exception:
        return []

    events: List[Dict[str, Any]] = []
    current: Dict[str, str] = {}
    in_event = False
    for line in _unfold_ics(r.text or ""):
        if line == "BEGIN:VEVENT":
            current = {}
            in_event = True
            continue
        if line == "END:VEVENT":
            if in_event:
                summary = current.get("SUMMARY", "").strip()
                ds = _parse_ics_date(current.get("DTSTART", ""))
                if summary and ds:
                    events.append({"date": ds, "title": summary, "source": "BLS"})
            current = {}
            in_event = False
            continue
        if not in_event or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.split(";", 1)[0]
        if key in {"SUMMARY", "DTSTART"}:
            current[key] = value
    return events


def _fetch_bea_events() -> List[Dict[str, Any]]:
    try:
        r = requests.get(BEA_RELEASE_JSON_URL, headers=REQUEST_HEADERS, timeout=20)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return []

    events: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return events
    for title, obj in payload.items():
        dates = (obj or {}).get("release_dates") if isinstance(obj, dict) else None
        if not isinstance(dates, list):
            continue
        for raw in dates:
            try:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                events.append({"date": dt.date(), "title": str(title), "source": "BEA"})
            except Exception:
                continue
    return events


def _fetch_fed_events(date_str: str) -> List[Dict[str, Any]]:
    """Parse the Fed's plain-language Upcoming Dates block for FOMC meetings."""
    try:
        r = requests.get(FED_MONETARY_POLICY_URL, headers=REQUEST_HEADERS, timeout=20)
        r.raise_for_status()
    except Exception:
        return []

    text = html.unescape(re.sub(r"<[^>]+>", " ", r.text or ""))
    text = re.sub(r"\s+", " ", text)
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    target = _target_date(date_str)
    events: List[Dict[str, Any]] = []
    # Example page text: "Oct. 27-28 FOMC Meeting"
    for mon, first_day, _last_day in re.findall(r"\b([A-Z][a-z]{2})\.\s+(\d{1,2})(?:\s*-\s*(\d{1,2}))?\s+FOMC Meeting\b", text):
        month = month_map.get(mon)
        if not month:
            continue
        year = target.year
        try:
            event_date = date(year, month, int(first_day))
            if event_date < target - timedelta(days=30):
                event_date = date(year + 1, month, int(first_day))
            events.append({"date": event_date, "title": "Federal Reserve rate decision", "source": "Federal Reserve"})
        except Exception:
            continue
    return events


def _official_calendar_events(date_str: str) -> List[Dict[str, Any]]:
    return _fetch_bls_events() + _fetch_bea_events() + _fetch_fed_events(date_str)


def _normalize_event_title(title: str) -> Optional[str]:
    t = (title or "").casefold()
    if "employment situation" in t:
        return "Jobs Report"
    if "consumer price index" in t:
        return "Inflation Report"
    if "personal income and outlays" in t:
        return "Consumer Spending & Inflation"
    if "gross domestic product" in t and "state" not in t and "county" not in t:
        return "GDP"
    if "federal reserve rate decision" in t:
        return "Fed Rate Decision"
    return None


def _next_signal(date_str: str, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    target = _target_date(date_str)
    candidates: List[Tuple[date, str, str]] = []
    for event in events:
        event_date = event.get("date")
        label = _normalize_event_title(event.get("title", ""))
        if not isinstance(event_date, date) or not label or event_date <= target:
            continue
        if event_date > target + timedelta(days=30):
            continue
        candidates.append((event_date, label, event.get("source", "")))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    first_date = candidates[0][0]
    same_day = [x for x in candidates if x[0] == first_date]
    labels = list(dict.fromkeys(x[1] for x in same_day))[:2]
    sources = list(dict.fromkeys(x[2] for x in same_day if x[2]))
    return {
        "date": first_date.strftime("%Y-%m-%d"),
        "date_label": first_date.strftime("%b %d").replace(" 0", " "),
        "label": " + ".join(labels),
        "source": " / ".join(sources),
    }


def _latest_release_date(events: List[Dict[str, Any]], date_str: str, needle: str) -> Optional[date]:
    target = _target_date(date_str)
    eligible = []
    needle = needle.casefold()
    for event in events:
        event_date = event.get("date")
        title = str(event.get("title", "")).casefold()
        if isinstance(event_date, date) and event_date <= target and needle in title:
            eligible.append(event_date)
    return max(eligible) if eligible else None


def _last_value_change(rows: List[Tuple[str, float]]) -> Optional[Tuple[str, float, float]]:
    if len(rows) < 2:
        return None
    for i in range(len(rows) - 1, 0, -1):
        if abs(rows[i][1] - rows[i - 1][1]) >= 0.005:
            return rows[i][0], rows[i - 1][1], rows[i][1]
    return None


def _what_changed(
    date_str: str,
    inflation: Dict[str, Any],
    jobs: Dict[str, Any],
    fed_rate: Dict[str, Any],
    fed_rows: List[Tuple[str, float]],
    events: List[Dict[str, Any]],
) -> str:
    target = _target_date(date_str)
    changes: List[Tuple[int, str]] = []

    rate_change = _last_value_change(fed_rows)
    if rate_change:
        change_date = datetime.strptime(rate_change[0], "%Y-%m-%d").date()
        delta = rate_change[2] - rate_change[1]
        age = (target - change_date).days
        if 0 <= age <= 10 and abs(delta) >= 0.10:
            direction = "rose" if delta > 0 else "fell"
            changes.append((age, f"The Fed rate {direction} by {abs(delta):.2f} percentage points this week."))

    cpi_release = _latest_release_date(events, date_str, "Consumer Price Index")
    if cpi_release and inflation.get("value") is not None:
        age = (target - cpi_release).days
        delta = inflation.get("trend_delta_3m_pp")
        if 0 <= age <= 10 and delta is not None and abs(float(delta)) >= 0.20:
            direction = "fell" if float(delta) < 0 else "rose"
            old_value = inflation.get("comparison_value_3m")
            if old_value is not None:
                changes.append((age, f"Inflation {direction} from {float(old_value):.1f}% three months ago to {float(inflation['value']):.1f}%."))
            else:
                changes.append((age, f"The latest inflation reading {direction} compared with three months earlier."))

    jobs_release = _latest_release_date(events, date_str, "Employment Situation")
    if jobs_release and jobs.get("value") is not None:
        age = (target - jobs_release).days
        delta = jobs.get("trend_delta_3m_pp")
        if 0 <= age <= 10 and delta is not None and abs(float(delta)) >= 0.20:
            direction = "rose" if float(delta) > 0 else "fell"
            changes.append((age, f"The unemployment rate has {direction} compared with three months earlier."))

    if not changes:
        return "No major economic signal changed enough to alter the broader picture."
    changes.sort(key=lambda x: x[0])
    return " ".join(text for _, text in changes[:2])


def _semantic_evidence(articles: List[dict]) -> Dict[str, List[str]]:
    titles: List[str] = []
    seen = set()
    for article in articles:
        title = (article.get("title") or "").strip()
        key = title.casefold()
        if title and key not in seen:
            titles.append(title)
            seen.add(key)
    if not titles:
        return {}

    model = SentenceTransformer(LOCAL_EMB_MODEL)
    labels = list(ECON_EVIDENCE_PROTOTYPES.keys())
    proto_texts: List[str] = []
    proto_owner: List[str] = []
    for label in labels:
        for text in ECON_EVIDENCE_PROTOTYPES[label]:
            proto_texts.append(text)
            proto_owner.append(label)

    proto_vecs = model.encode(proto_texts, normalize_embeddings=True)
    title_vecs = model.encode(titles, normalize_embeddings=True)
    label_indices = {
        label: [i for i, owner in enumerate(proto_owner) if owner == label]
        for label in labels
    }

    candidates: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for title, vec in zip(titles, title_vecs):
        sims = vec @ proto_vecs.T
        scores = {
            label: float(max(sims[i] for i in label_indices[label]))
            for label in labels
        }
        noise_score = scores["noise"]
        ranked = sorted(
            ((score, label) for label, score in scores.items() if label != "noise"),
            reverse=True,
        )
        if not ranked:
            continue
        best_score, best_label = ranked[0]
        if best_score < ECON_SEMANTIC_MIN:
            continue
        if best_score < noise_score + ECON_CLASS_MARGIN:
            continue
        candidates[best_label].append((best_score, title))

    selected: Dict[str, List[str]] = {}
    total = 0

    # Inflation, jobs, rates, growth, oil and yields already have official data
    # above. Headline evidence is reserved for emerging conditions that official
    # scheduled series do not capture quickly: trade/tariffs, fiscal/debt,
    # financial stress and economically material geopolitical disruptions.
    # Requiring repeated coverage prevents one-off opinion/local stories from
    # becoming the economic narrative. A single very-high-confidence Fed or
    # financial-stress headline may pass because those can change abruptly.
    order = [
        "monetary_policy", "trade_tariffs", "fiscal_debt", "energy_supply",
        "financial_stress", "geopolitical_economy",
    ]
    single_high_conf = {"monetary_policy", "financial_stress"}
    for label in order:
        ranked = sorted(candidates.get(label, []), key=lambda x: x[0], reverse=True)
        if not ranked or total >= MAX_EVIDENCE_TOTAL:
            continue

        enough_support = len(ranked) >= 2
        very_high_conf_single = (
            label in single_high_conf
            and ranked[0][0] >= max(0.52, ECON_SEMANTIC_MIN + 0.08)
        )
        if not enough_support and not very_high_conf_single:
            continue

        room = MAX_EVIDENCE_TOTAL - total
        picks = [title for _, title in ranked[: min(MAX_EVIDENCE_PER_THEME, room)]]
        if picks:
            selected[label] = picks
            total += len(picks)
    return selected


def _evidence_for_prompt(evidence: Dict[str, List[str]]) -> str:
    if not evidence:
        return "- No clearly relevant macroeconomic headlines were identified."
    lines: List[str] = []
    for theme, titles in evidence.items():
        label = THEME_LABELS.get(theme, theme)
        for title in titles:
            lines.append(f"- [{label}] {title}")
    return "\n".join(lines)


def _signal_line(label: str, obj: Dict[str, Any], suffix: str = "") -> str:
    if obj.get("value") is None:
        return f"{label}: unavailable"
    value = obj.get("display_value") or obj.get("formatted") or obj.get("value")
    detail = obj.get("detail")
    trend = obj.get("trend")
    as_of = obj.get("as_of")
    pieces = [f"{label}: {value}"]
    if detail:
        pieces.append(detail)
    if trend and trend != "Unavailable":
        pieces.append(f"comparison={trend}")
    if as_of:
        pieces.append(f"observation={as_of}")
    if suffix:
        pieces.append(suffix)
    return "; ".join(pieces)


def _fallback_what_happened(
    market_state: str,
    fed_rate: Dict[str, Any],
    secondary: Dict[str, Dict[str, Any]],
    evidence: Dict[str, List[str]],
) -> str:
    """Describe the forces shaping the current economic picture.

    This section answers "what happened?" It should describe policy moves,
    market/financial-condition changes, and repeated geopolitical, trade,
    energy, fiscal, or financial-stress themes. It should not translate those
    developments into household-level consequences; that belongs in
    What It Means.
    """
    forces: List[str] = []

    oil = secondary.get("oil_wti", {})
    oil_change = oil.get("change_30d")
    if oil_change is not None and abs(float(oil_change)) >= 10.0:
        direction = "risen" if float(oil_change) > 0 else "fallen"
        forces.append(
            f"WTI crude oil has {direction} {abs(float(oil_change)):.1f}% over the past month."
        )

    rate_delta = fed_rate.get("trend_delta_30d_pp")
    if rate_delta is not None and abs(float(rate_delta)) >= 0.10:
        direction = "raised" if float(rate_delta) > 0 else "lowered"
        lower = fed_rate.get("target_range_lower")
        upper = fed_rate.get("target_range_upper")
        destination = ""
        if lower is not None and upper is not None:
            destination = f" to {float(lower):.2f}–{float(upper):.2f}%"
        forces.append(
            f"The Federal Reserve {direction} its target rate range by {abs(float(rate_delta)):.2f} percentage points{destination}."
        )

    treasury = secondary.get("treasury_10y", {})
    treasury_change = treasury.get("change_30d")
    if treasury_change is not None and abs(float(treasury_change)) >= 0.20:
        direction = "risen" if float(treasury_change) > 0 else "fallen"
        forces.append(
            f"The 10-year Treasury yield has {direction} over the past month."
        )

    theme_phrases = {
        "trade_tariffs": "Trade and tariff developments are reshaping business costs and supply chains.",
        "fiscal_debt": "Federal debt and borrowing concerns are adding pressure to the financial backdrop.",
        "financial_stress": "Credit or banking stress is tightening financial conditions.",
        "geopolitical_economy": "Geopolitical tensions are affecting energy, trade, or supply-chain conditions.",
        "energy_supply": "Energy and supply disruptions are affecting the economic backdrop.",
        "monetary_policy": "Federal Reserve policy is materially changing financial conditions.",
    }
    for theme in ("geopolitical_economy", "trade_tariffs", "energy_supply", "financial_stress", "fiscal_debt", "monetary_policy"):
        if evidence.get(theme):
            phrase = theme_phrases[theme]
            if phrase not in forces:
                forces.append(phrase)
        if len(forces) >= 3:
            break

    if not forces:
        if market_state in {"up", "strong_up", "down", "strong_down", "mixed"}:
            return f"Broad U.S. markets were {_market_state_label(market_state).lower()}, but no broader economic force was clear enough to single out."
        return "No major new economic force stood out beyond the current readings in markets, prices, jobs, and interest rates."

    return " ".join(forces[:2])


def _fallback_what_it_means(
    fed_rate: Dict[str, Any],
    secondary: Dict[str, Dict[str, Any]],
    evidence: Dict[str, List[str]],
) -> str:
    """Translate the economic backdrop into everyday consequences.

    This section answers "what does that mean for people?" It intentionally
    avoids repeating the events, percentages, policy actions, or geopolitical
    details already stated in What Happened.
    """
    oil = secondary.get("oil_wti", {})
    oil_change = oil.get("change_30d")
    rate_delta = fed_rate.get("trend_delta_30d_pp")

    oil_material = oil_change is not None and abs(float(oil_change)) >= 10.0
    rate_material = rate_delta is not None and abs(float(rate_delta)) >= 0.10
    trade = bool(evidence.get("trade_tariffs"))
    financial_stress = bool(evidence.get("financial_stress"))
    fiscal_debt = bool(evidence.get("fiscal_debt"))

    if oil_material and float(oil_change) > 0 and rate_material and float(rate_delta) > 0:
        return "Households may face higher fuel and shipping costs alongside more expensive mortgages, car loans, and other borrowing."

    if oil_material and float(oil_change) > 0:
        return "Households may pay more for fuel, shipping, groceries, and other goods, while businesses face higher transportation and production costs."

    if trade:
        return "Imported goods and business supplies may become more expensive, with some of those higher costs eventually reaching consumers."

    if rate_material:
        if float(rate_delta) > 0:
            return "Mortgages, car loans, credit, and business borrowing may become more expensive."
        return "Mortgages, car loans, credit, and business borrowing may gradually become less expensive."

    if financial_stress:
        return "Loans and credit may become harder or more expensive to obtain for households and businesses."

    if fiscal_debt:
        return "Higher long-term borrowing costs can eventually make mortgages, business loans, and government financing more expensive."

    if oil_material and float(oil_change) < 0:
        return "Lower energy costs can reduce pressure on fuel, shipping, and some everyday prices."

    return "The available evidence does not point to one clear everyday impact beyond the changes already shown in the economic indicators."


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _synthesize_explainer(
    date_str: str,
    market_state: str,
    market_close_date: Optional[str],
    inflation: Dict[str, Any],
    jobs: Dict[str, Any],
    fed_rate: Dict[str, Any],
    secondary: Dict[str, Dict[str, Any]],
    evidence: Dict[str, List[str]],
    events: List[Dict[str, Any]],
) -> Tuple[str, str]:
    fallback_happened = _fallback_what_happened(
        market_state=market_state,
        fed_rate=fed_rate,
        secondary=secondary,
        evidence=evidence,
    )
    fallback_means = _fallback_what_it_means(
        fed_rate=fed_rate,
        secondary=secondary,
        evidence=evidence,
    )
    if not os.getenv("OPENAI_API_KEY"):
        return fallback_happened, fallback_means

    treasury = secondary.get("treasury_10y", {})
    oil = secondary.get("oil_wti", {})
    gdp = secondary.get("real_gdp_growth", {})

    context_lines = [
        f"Market pulse: {_market_state_label(market_state)}; market close={market_close_date or 'unavailable'}",
        _signal_line("Inflation (CPI year-over-year)", inflation),
        _signal_line("Jobs", jobs),
        _signal_line("Federal Reserve benchmark rate", fed_rate),
    ]

    notable_changes: List[str] = []
    target_day = _target_date(date_str)

    # Routine gauge readings belong in the four mini-cards, not automatically in
    # What Happened. CPI/jobs enter the narrative only when their official release
    # is itself a fresh event (within three days of the requested date).
    cpi_release = _latest_release_date(events, date_str, "Consumer Price Index")
    inflation_delta = inflation.get("trend_delta_3m_pp")
    inflation_old = inflation.get("comparison_value_3m")
    if (
        cpi_release is not None
        and 0 <= (target_day - cpi_release).days <= 3
        and inflation.get("value") is not None
        and inflation_old is not None
        and inflation_delta is not None
        and abs(float(inflation_delta)) >= 0.20
    ):
        notable_changes.append(
            f"A new inflation report showed prices rising {float(inflation['value']):.1f}% over the past year, "
            f"compared with {float(inflation_old):.1f}% three months earlier."
        )

    jobs_release = _latest_release_date(events, date_str, "Employment Situation")
    payroll_change = jobs.get("payroll_change_k")
    if (
        jobs_release is not None
        and 0 <= (target_day - jobs_release).days <= 3
        and jobs.get("value") is not None
        and payroll_change is not None
    ):
        payroll_word = "added" if float(payroll_change) >= 0 else "lost"
        notable_changes.append(
            f"A new jobs report {payroll_word} {abs(int(round(float(payroll_change)))):,}K payroll jobs "
            f"with unemployment at {float(jobs['value']):.1f}%."
        )

    # A truly large market move is useful context in its own right. Describe it as
    # a reaction/state, never as the cause of another economic development.
    if market_state in {"strong_up", "strong_down"}:
        market_word = "strong gain" if market_state == "strong_up" else "sharp decline"
        close_label = _human_date(market_close_date) if market_close_date else "the latest close"
        notable_changes.append(
            f"Broad U.S. stocks posted a {market_word} at {close_label}."
        )

    rate_delta = fed_rate.get("trend_delta_30d_pp")
    if fed_rate.get("value") is not None and rate_delta is not None and abs(float(rate_delta)) >= 0.10:
        direction = "rose" if float(rate_delta) > 0 else "fell"
        lower = fed_rate.get("target_range_lower")
        upper = fed_rate.get("target_range_upper")
        if lower is not None and upper is not None:
            destination = f" to a {float(lower):.2f}–{float(upper):.2f}% target range"
        else:
            destination = ""
        notable_changes.append(
            f"The Fed target range {direction} by {abs(float(rate_delta)):.2f} percentage points "
            f"over 30 days{destination}."
        )
    if treasury.get("value") is not None:
        context_lines.append(
            f"10-year Treasury yield: {treasury['value']:.2f}%; 30-day change={treasury.get('change_30d')} percentage points; observation={treasury.get('as_of')}"
        )
        treasury_change = treasury.get("change_30d")
        if treasury_change is not None and abs(float(treasury_change)) >= 0.20:
            direction = "rose" if float(treasury_change) > 0 else "fell"
            notable_changes.append(
                f"The 10-year Treasury yield {direction} by {abs(float(treasury_change)):.2f} percentage points "
                f"over 30 days to {float(treasury['value']):.2f}%."
            )
    if oil.get("value") is not None:
        context_lines.append(
            f"WTI crude oil: ${oil['value']:.2f}; 30-day change={oil.get('change_30d')}%; observation={oil.get('as_of')}"
        )
        oil_change = oil.get("change_30d")
        if oil_change is not None and abs(float(oil_change)) >= 10.0:
            direction = "rose" if float(oil_change) > 0 else "fell"
            notable_changes.append(
                f"WTI crude oil {direction} {abs(float(oil_change)):.1f}% over 30 days "
                f"to ${float(oil['value']):.2f}."
            )
    if gdp.get("value") is not None:
        context_lines.append(
            f"Latest real GDP growth: {gdp['value']:.1f}% annualized; observation={gdp.get('as_of')}"
        )

    if market_state in {"strong_up", "strong_down"}:
        reaction = _market_state_label(market_state).lower()
        reaction_date = _human_date(market_close_date) if market_close_date else "the latest close"
        market_reaction_requirement = (
            f"REQUIRED MARKET CONTEXT: Broad U.S. stocks posted a {reaction} at {reaction_date}. "
            "Mention this reaction in what_happened, while keeping it clearly separate from the causes or forces behind the move."
        )
    else:
        market_reaction_requirement = (
            "MARKET CONTEXT: The market move was not unusually large, so mention it only if it materially improves the explanation."
        )

    prompt = f"""You are writing the calm-tech Market & Economy section of Nominal News for {date_str}.

The four signal cards already tell the reader where broad U.S. markets, inflation, unemployment, and interest rates stand. DO NOT use the prose below to summarize those cards again. Your job is to explain what is happening around them: the current forces, policy changes, shocks, and tensions that help a reader understand the economic picture.

Think of the card as gauges and this prose as the explanation behind the gauges. A general North American reader should understand it without finance knowledge.

OFFICIAL QUANTITATIVE CONTEXT (grounding and financial-condition evidence):
{chr(10).join('- ' + line for line in context_lines)}

NOTABLE RECENT CHANGES (use these first when they are meaningful):
{chr(10).join('- ' + line for line in notable_changes) if notable_changes else '- No official change crossed the significance thresholds.'}

{market_reaction_requirement}

REPEATED EMERGING NEWS THEMES (secondary context only):
{_evidence_for_prompt(evidence)}

Return valid JSON with exactly two keys:
- "what_happened": 1-2 short sentences describing the 2-3 most important EVENTS, POLICY MOVES, RELEASES, or ECONOMIC FORCES shaping the current U.S. economic backdrop. It answers only: "What happened in the economy/world that is important to understand today?" Good material includes geopolitical or supply disruptions, oil/energy moves, tariffs/trade, Fed decisions, Treasury-yield moves, financial stress, genuinely fresh major data releases, and unusually large market reactions.
- "what_it_means": exactly 1 short sentence by default, translating those developments into practical everyday IMPACTS for households or businesses. It answers only: "How could this show up in ordinary life?" Prefer concrete effects such as fuel, groceries, shipping, mortgages, car loans, credit, hiring, wages, or business costs. If two impacts matter, combine them naturally in the same sentence using "while", "and", or a semicolon. Use 2 sentences only in the exceptional case that combining two separate critical impacts into one sentence would materially reduce accuracy or clarity.

Rules:
- The four signal cards are the STATUS layer. These two paragraphs are the CONTEXT + IMPACT layers. Do not duplicate the status layer.
- Routine inflation and unemployment readings are already visible in the cards. Do NOT mention them in "what_happened" merely because they differ from three months ago. Mention them only when a fresh CPI/jobs release is itself one of the day's major economic events.
- If a fresh data release is important, frame it as an EVENT (for example, "a new inflation report showed...") rather than mechanically restating a card.
- A Strong Gain or Sharp Decline MUST be mentioned in "what_happened" as a MARKET REACTION because it is itself meaningful context. Never imply that the market move caused the underlying economic event; use contrast language such as "while", "even as", or "despite" when appropriate. Ordinary Gain/Decline/Steady/Mixed states remain optional context.
- "what_happened" should usually begin with the most consequential current event/force, not with the four gauge readings.
- If repeated emerging news themes are supplied, "what_happened" should normally include at least one when it has broad economic relevance. Fast-moving themes are the main way to capture what monthly/quarterly official data miss.
- "what_happened" stays at the macro/context level. Do not explain household budgets, mortgages, groceries, or personal spending there.
- "what_it_means" stays at the everyday-impact level. It should add NEW information rather than summarize "what_happened" again.
- In "what_it_means", do NOT repeat percentages, dates, named geopolitical events, Fed decision details, oil-price moves, tariff announcements, or Treasury-yield changes already stated above. A short bridge such as "higher energy costs" or "tighter borrowing conditions" is fine when needed for clarity.
- Start "what_it_means" from the consequence, not from a recap of the cause. Example style: "Households may face higher fuel and shipping costs, while mortgages and other borrowing become more expensive."
- Treat one sentence as a hard default. Do not start a second sentence merely to cover another ordinary consequence; combine related consequences cleanly in the first sentence.
- If one practical impact clearly dominates, focus on it instead of forcing every force into the sentence.
- Prefer concrete household/business effects over generic endings such as "consumer spending" or "economic growth" unless those broader outcomes are genuinely the clearest supported implication.
- Official quantitative context is factual grounding. Use headlines to identify current themes; do not treat a headline as proof of a causal claim it does not support.
- CAUSALITY DISCIPLINE: never infer causation from timing or co-occurrence. State that X "caused", "drove", "contributed to", "reflected", or was "because of" Y only when the supplied evidence explicitly supports that relationship. Otherwise describe the developments as concurrent using language such as "while", "alongside", "amid", or "at the same time".
- In particular, do not attribute moves in Treasury yields, stock prices, oil prices, inflation, or employment to a geopolitical/policy event unless the supplied evidence explicitly establishes that link.
- Do not use interpretive phrases such as "reflecting market reactions to these developments", "signaling investor concern", "in response to", or "suggesting markets reacted to" unless the supplied evidence explicitly makes that attribution. A factual move in a market indicator is not, by itself, evidence of why it moved.
- When two developments are independently supported but their relationship is not, state them independently. Example: "Oil prices rose sharply over the past month, while the 10-year Treasury yield also increased." Stop there; do not append an unsupported explanation for the yield or market move.
- Do not invent statistics, events, policy changes, causal relationships, or forecasts.
- Do not imply the stock market and the economy are the same thing.
- Do not give investment advice or predict what markets will do next.
- Avoid jargon and vague evaluative language. Prefer direct descriptions over labels such as "cooling", "heating up", "strengthening", or "weakening".
- Do not strengthen the evidence: say "geopolitical tensions" when the evidence supports tensions, and say "war" only when supplied evidence clearly establishes a war.
- When multiple themes are present, connect them only through supported economic channels; do not invent a direct causal chain.
- Prefer the 2-3 forces that matter most over cataloguing everything.
- Calm, neutral, concise. No drama.
- JSON only. No markdown fences.
"""

    messages = [
        {"role": "system", "content": "You are a careful economic explainer. Separate observed facts, context, and causal claims. Never infer causation or market interpretation from co-occurrence. A price, yield, rate, or index move does not reveal why it moved unless the supplied evidence explicitly establishes that relationship."},
        {"role": "user", "content": prompt},
    ]

    def _call(model_name: str) -> str:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
        return resp["choices"][0]["message"]["content"]

    text = None
    for model_name in (MARKET_MODEL, MARKET_MODEL_FALLBACK):
        try:
            text = _call(model_name)
            break
        except Exception:
            continue
    text = _strip_code_fences(text) if text else ""
    if not text:
        return fallback_happened, fallback_means
    try:
        obj = json.loads(text)
        what_happened = (obj.get("what_happened") or obj.get("what_matters") or "").strip() or fallback_happened
        what_it_means = (obj.get("what_it_means") or "").strip() or fallback_means
        return what_happened, what_it_means
    except Exception:
        return fallback_happened, fallback_means


def main():
    args = _parse_args()
    date_str = args.date or datetime.today().strftime("%Y-%m-%d")

    in_path = BASE_DIR / f"articles_with_bias_{date_str}.json"
    out_path = BASE_DIR / f"market_overview_{date_str}.json"
    if not in_path.exists():
        print(f"❌ Missing {in_path.name}. Run bias_labeler.py first.")
        raise SystemExit(1)

    articles = _read_json(in_path)
    if not isinstance(articles, list):
        print("❌ Unexpected input shape: expected a list of articles.")
        raise SystemExit(1)

    indexes, market_state, market_stats, close_date = _build_market_context(date_str)
    inflation = _build_inflation_signal(date_str)
    jobs = _build_jobs_signal(date_str)
    fed_rate, fed_rows = _build_rate_signal(date_str)
    secondary = _build_secondary_context(date_str)
    evidence = _semantic_evidence(articles)

    calendar_events = _official_calendar_events(date_str)
    next_signal = _next_signal(date_str, calendar_events)
    what_changed = _what_changed(
        date_str=date_str,
        inflation=inflation,
        jobs=jobs,
        fed_rate=fed_rate,
        fed_rows=fed_rows,
        events=calendar_events,
    )

    what_happened, what_it_means = _synthesize_explainer(
        date_str=date_str,
        market_state=market_state,
        market_close_date=close_date,
        inflation=inflation,
        jobs=jobs,
        fed_rate=fed_rate,
        secondary=secondary,
        evidence=evidence,
        events=calendar_events,
    )

    market_trend_direction = "flat"
    market_trend_tone = "neutral"
    market_trend_label = "Markets steady"
    if market_state in {"up", "strong_up"}:
        market_trend_direction = "up"
        market_trend_tone = "positive"
        market_trend_label = "Markets trending up"
    elif market_state in {"down", "strong_down"}:
        market_trend_direction = "down"
        market_trend_tone = "negative"
        market_trend_label = "Markets trending down"
    elif market_state == "mixed":
        market_trend_direction = "mixed"
        market_trend_label = "Markets mixed"

    signals = {
        "markets": {
            "label": "Markets",
            "descriptor": "How investors view overall U.S. business",
            "value": _market_state_label(market_state),
            "short_value": _market_state_label(market_state),
            "display_value": _market_state_label(market_state),
            "detail": f"Broad U.S. Stocks · {_human_date(close_date)}" if close_date else "",
            "state": market_state,
            "trend_direction": market_trend_direction,
            "trend_tone": market_trend_tone,
            "trend_label": market_trend_label,
            "as_of": close_date,
            "source": "FRED",
        },
        "inflation": {
            "label": "Inflation",
            "descriptor": "How quickly prices are rising",
            **inflation,
        },
        "jobs": {
            "label": "Jobs",
            "descriptor": "How many people are out of work and looking",
            **jobs,
        },
        "rates": {
            "label": "Interest Rates",
            "descriptor": "How expensive it is to borrow money",
            **fed_rate,
        },
    }

    evidence_titles = [title for titles in evidence.values() for title in titles]
    out = {
        "date": date_str,
        "label": "Market & Economy Overview",
        "market_state": market_state,
        "market_state_label": _market_state_label(market_state),
        "market_close_date": close_date,
        "market_close_label": _human_date_with_weekday(close_date),
        "market_average_pct_change": market_stats["average_pct_change"],
        "market_average_abs_pct_change": market_stats["average_abs_pct_change"],
        "indexes": indexes,
        "signals": signals,
        "what_happened": what_happened,
        "what_matters": what_happened,
        "what_it_means": what_it_means,
        "what_changed": what_changed,
        "next_signal": next_signal,
        "secondary_context": secondary,
        "macro_evidence_by_theme": evidence,
        "evidence_titles": evidence_titles,
        # Backward-compatible aliases for any older template/code paths.
        "markets_because": what_happened,
        "investments_doing": what_it_means,
        "data_source": "Official U.S. data via FRED + BLS/BEA/Federal Reserve calendars",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    _write_json(out_path, out)

    signal_text = []
    for key in ("markets", "inflation", "jobs", "rates"):
        obj = signals[key]
        signal_text.append(f"{obj['label']}={obj.get('short_value') or 'n/a'}")

    print(f"✅ Wrote market overview → {out_path.name}")
    print("📊 Economy signals: " + " | ".join(signal_text))
    print(
        f"📈 Market pulse: {_market_state_label(market_state)} | "
        f"avg={market_stats['average_pct_change'] if market_stats['average_pct_change'] is not None else 'n/a'}% | "
        f"close={close_date or 'n/a'}"
    )
    print(
        f"🧾 Macro evidence: {sum(len(v) for v in evidence.values())} titles across "
        f"{len(evidence)} themes"
    )
    if next_signal:
        print(f"🗓️ Next signal: {next_signal['label']} · {next_signal['date']}")
    else:
        print("🗓️ Next signal: unavailable")


if __name__ == "__main__":
    main()
