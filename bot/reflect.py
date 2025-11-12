# bot/reflect.py
import json, argparse, datetime as dt
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_PATH = DATA_DIR / "history.json"
LIKES_PATH   = DATA_DIR / "user_likes.json"
INSIGHTS_JSON = DATA_DIR / "insights.json"
DIGEST_MD     = DATA_DIR / "weekly_digest.md"

def _load_json(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def _safe_iter(x):
    return x if isinstance(x, list) else []

def summarize(turns, likes):
    """turns = chat history objects; likes = per-turn extracted dicts."""
    # Basic counters
    sentiment_ctr = Counter()
    reason_ctr    = Counter()
    action_ctr    = Counter()

    # Buckets by product/sub-category if present
    by_product = defaultdict(lambda: Counter())
    by_subcat  = defaultdict(lambda: Counter())

    for row in _safe_iter(likes):
        sent = row.get("sentiment") or row.get("last_sentiment") or "neutral"
        reason = row.get("last_reason", "unspecified")
        action = (row.get("suggested_action") or {}).get("message", "none")

        sentiment_ctr[sent] += 1
        reason_ctr[reason]  += 1
        action_ctr[action]  += 1

        prod = row.get("product_name") or row.get("product") or "unknown_product"
        subc = row.get("sub_category") or "unknown_subcategory"
        by_product[prod][reason] += 1
        by_subcat[subc][reason]  += 1

    # Simple trend lines (by day)
    by_day_sent = defaultdict(lambda: Counter())
    for t in _safe_iter(turns):
        try:
            # ChatMessageHistory dicts usually carry 'additional_kwargs' or 'kwargs'; fall back to now()
            ts = t.get("kwargs", {}).get("timestamp") or t.get("additional_kwargs", {}).get("timestamp")
            when = dt.datetime.fromisoformat(ts) if isinstance(ts, str) else dt.datetime.now()
        except Exception:
            when = dt.datetime.now()
        day = when.date().isoformat()
        role = t.get("type") or t.get("role", "")
        if role in ("human", "user"):
            # If you store per-turn sentiment in likes, you could align here.
            by_day_sent[day]["turns"] += 1

    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "top_reasons": reason_ctr.most_common(10),
        "sentiment_distribution": sentiment_ctr,
        "top_actions": action_ctr.most_common(10),
        "by_product_top_reasons": {k: v.most_common(5) for k, v in by_product.items()},
        "by_subcategory_top_reasons": {k: v.most_common(5) for k, v in by_subcat.items()},
        "activity_by_day": {k: dict(v) for k, v in by_day_sent.items()}
    }

def write_digest(insights):
    lines = []
    lines.append(f"# Weekly Customer Reflection — {insights['generated_at']}\n")
    tr = insights["top_reasons"]
    sd = insights["sentiment_distribution"]
    ta = insights["top_actions"]

    def pct(n, d): 
        return f"{(100*n/d):.1f}%" if d else "0%"

    total_sent = sum(sd.values())
    lines.append("## Top Reasons for Returns / Friction")
    for reason, n in tr[:5]:
        lines.append(f"- **{reason}** — {n} reports ({pct(n, sum(dict(tr).values()))})")

    lines.append("\n## Sentiment Mix")
    for s, n in sd.items():
        lines.append(f"- {s}: {n} ({pct(n, total_sent)})")

    lines.append("\n## Commonly Suggested Actions")
    for act, n in ta[:5]:
        lines.append(f"- {act} — {n}")

    lines.append("\n## Signals by Sub-Category (top reasons)")
    for sub, reasons in insights["by_subcategory_top_reasons"].items():
        if not reasons: 
            continue
        bullets = ", ".join([f"{r} ({c})" for r, c in reasons[:3]])
        lines.append(f"- **{sub}**: {bullets}")

    DIGEST_MD.write_text("\n".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Aggregate Thia reflections for the team.")
    parser.add_argument("--period", default="weekly", help="label only; aggregation is over all data")
    args = parser.parse_args()

    turns = _load_json(HISTORY_PATH)
    likes = _load_json(LIKES_PATH)

    insights = summarize(turns, likes)

    # Save JSON (for dashboards) and Markdown (for humans)
    INSIGHTS_JSON.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    write_digest(insights)

    print(f"Wrote: {INSIGHTS_JSON}")
    print(f"Wrote: {DIGEST_MD}")

if __name__ == "__main__":
    main()

###What it does: reads data/history.json + data/user_likes.json you already write every turn 
# - aggregates reasons, sentiment, actions, and per-product/sub-category signals 
# - writes data/insights.json (for dashboards) and a readable data/weekly_digest.md for the team 
# - does not change customer replies

crontab -e
# add: runs on a schedule, Monday at 7am 
0 7 * * MON /usr/bin/python3 /path/to/your/repo/bot/reflect.py --period weekly >> /path/to/your/repo/data/reflect.log 2>&1


