"""
Convert question + token pairs into statements for NLI. Input is JSON, output is CSV.
"""

import json
import re
from datetime import datetime
from hashlib import md5
import pandas as pd
import sys


import json
import re
from datetime import datetime
from hashlib import md5


# -------------------------
# Basic helpers
# -------------------------

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_resolution_date(entry) -> str:
    """Convert end_date_iso into YYYY-MM-DD."""
    dt = datetime.fromisoformat(entry["end_date_iso"].replace("Z", "+00:00"))
    return dt.date().isoformat()

# -------------------------
# Up/down helpers
# -------------------------
def detect_updown_market(question, token_names):
    q = question.lower()
    tokens = {t.lower() for t in token_names}
    if "up or down" in q:
        return True
    if tokens == {"up", "down"}:
        return True
    return False


def parse_updown_context(question):
    """
    Extract asset name and time window from Up/Down-style questions.
    Example:
        "Bitcoin Up or Down - November 1, 5:15AM-5:30AM ET"
    """
    # Asset is everything before "Up or Down"
    asset = question.split("Up or Down")[0].strip()

    # Time window is everything after the dash
    if "-" in question:
        time_window = question.split("-", 1)[1].strip()
    else:
        time_window = ""

    return asset, time_window


def build_updown_statement(question, token_name, resolution_date):
    asset, time_window = parse_updown_context(question)
    token = token_name.lower()

    if token == "up":
        stmt = (
            f"The price of {asset} will end the {time_window} period "
            f"higher than it began."
        )
    else:  # Down
        stmt = (
            f"The price of {asset} will end the {time_window} period "
            f"lower than it began."
        )

    return stmt.rstrip(".") + f" before {resolution_date}."


# -------------------------
# Question-based handling
# -------------------------

def question_to_base_assertion(question: str) -> str:
    """
    Turn a yes/no question like:
        "Will Zohran Mamdani win by 15–20%?"
    into:
        "Zohran Mamdani will win by 15–20%."
    """
    q = question.strip().rstrip("?")

    # common yes/no starters
    m = re.match(r"^(Will|Does|Do|Is|Are|Was|Were)\s+(.*)$", q, flags=re.IGNORECASE)
    if m:
        core = m.group(2).strip()
    else:
        # fallback: just treat whole question as a statement
        core = q

    # Ensure "will" future tense if not obvious
    if not re.match(r"(?i)^will\b", core):
        core = "will " + core

    # Convert "will X" → "X will"
    m = re.match(r"(?i)will\s+(.*)$", core)
    if m:
        core = m.group(1).strip()
        core = core[0].upper() + core[1:]
        if not core.lower().startswith("will "):
            core = "will " + core
        # Now "will X..." → "X will..."
        m = re.match(r"(?i)will\s+(.*)$", core)
        if m:
            core = m.group(1).strip()

    if not core.endswith("."):
        core += "."

    return core


def build_yes_no_statement(question: str, token_name: str, resolution_date: str) -> str:
    """
    For yes/no tokens, use the question-based positive assertion.
    """
    base = question_to_base_assertion(question).rstrip(".")
    token = token_name.lower().strip()

    if token == "yes":
        return f"{base} before {resolution_date}."
    elif token == "no":
        return f"It is NOT the case that {base} before {resolution_date}."
    else:
        # Unusual case: question is yes/no but token is something else.
        # Treat token as categorical & fall back to slug logic in caller.
        return ""  # signals: not handled here


def build_wh_winner_statement(question: str, token_name: str, resolution_date: str) -> str:
    """
    Handle 'Who/Which ... will ...' winner-style questions.

    E.g. question: "Who will win the 2025 NBA Finals?"
         token: "Lakers"
         → "Lakers will win the 2025 NBA Finals before 2025-06-20."
    """
    q = question.strip()
    q_no_qmark = q.rstrip("?")

    # Who / Which / What (winner-style)
    m = re.match(r"(?i)who\s+(will.*)$", q_no_qmark)
    if m:
        tail = m.group(1).strip()
        stmt = f"{token_name} {tail}"
    else:
        m = re.match(r"(?i)which\s+(.*)$", q_no_qmark)
        if m:
            tail = m.group(1).strip()
            # e.g. "party will win the House in 2026"
            stmt = f"{token_name} {tail}"
        else:
            # not a recognized winner question
            return ""

    if not stmt.endswith("."):
        stmt += "."
    stmt = stmt[0].upper() + stmt[1:]

    return f"{stmt.rstrip('.')} before {resolution_date}."


# -------------------------
# Slug-based fallback (robust-ish)
# -------------------------

ABBR_EVENT_MAP = {
    "cwbb": "women's college basketball game",
    "cbb": "college basketball game",
    "nba": "NBA game",
    "ncaab": "college basketball game",
}

ABBR_TEAM_MAP = {
    "merri": "Merrimack College",
    "ri": "Rhode Island",
    # extend as needed
}


def parse_slug_to_event(slug: str, description: str) -> str:
    """
    Try to convert a slug like 'cwbb-merri-ri-2025-11-07' into
        "the women's college basketball game between Merrimack College and Rhode Island on 2025-11-07"
    If we can't do something smart, fall back to the first sentence of the description.
    """
    parts = slug.split("-")
    year = month = day = None
    base_parts = parts

    # Detect YYYY-MM-DD at end
    if len(parts) >= 3 and parts[-3].isdigit() and len(parts[-3]) == 4 \
       and parts[-2].isdigit() and parts[-1].isdigit():
        year, month, day = parts[-3], parts[-2], parts[-1]
        base_parts = parts[:-3]

    event_type = None
    participants = []

    for p in base_parts:
        pl = p.lower()
        if pl in ABBR_EVENT_MAP and event_type is None:
            event_type = ABBR_EVENT_MAP[pl]
        elif pl in ABBR_TEAM_MAP:
            participants.append(ABBR_TEAM_MAP[pl])
        else:
            # heuristic: treat as a name-ish token
            participants.append(p.capitalize())

    if event_type is None:
        event_type = "event"

    if len(participants) >= 2:
        between_clause = f"{participants[0]} and {participants[1]}"
        event_desc = f"the {event_type} between {between_clause}"
    elif len(participants) == 1:
        event_desc = f"the {event_type} involving {participants[0]}"
    else:
        event_desc = f"the {event_type}"

    if year and month and day:
        date_str = f"{year}-{month}-{day}"
        event_desc += f" on {date_str}"

    # If this is still too vague, fall back to description first sentence
    if event_desc == "the event" and description:
        first_sentence = description.split(".")[0].strip()
        if first_sentence:
            event_desc = first_sentence

    return event_desc


def build_slug_fallback_statement(slug: str,
                                  description: str,
                                  token_name: str,
                                  resolution_date: str) -> str:
    """
    Fallback when we *cannot* form a good question-based statement.
    Uses slug (and description) to generate a contextual sentence.
    """
    event_desc = parse_slug_to_event(slug, description)
    return (f"{token_name} will be the correct outcome for {event_desc} "
            f"before {resolution_date}.")


# -------------------------
# Master token → statement logic
# -------------------------

def build_statement_for_token(question: str,
                              slug: str,
                              description: str,
                              token_name: str,
                              resolution_date: str,
                              all_token_names) -> str:
    """
    Core decision logic:
      1. If market is yes/no and question is yes/no → use question_to_base_assertion.
      2. Else if question is a 'Who/Which ... will ...' style → use winner pattern.
      3. Else → slug-based fallback (robust parser).
    """
    if detect_updown_market(question, all_token_names):
        return build_updown_statement(question, token_name, resolution_date)

    token_set = {t.lower().strip() for t in all_token_names}
    q_lower = question.strip().lower()

    # CASE 1: Yes/No binary market with yes/no-style question
    if token_set == {"yes", "no"} and re.match(r"^(will|does|do|is|are|was|were)\s", q_lower):
        stmt = build_yes_no_statement(question, token_name, resolution_date)
        if stmt:  # handled successfully
            return stmt

    # CASE 2: Winner-style WH question ("Who will ...", "Which ... will ...")
    if q_lower.startswith("who ") or q_lower.startswith("which "):
        stmt = build_wh_winner_statement(question, token_name, resolution_date)
        if stmt:
            return stmt

    # If we reach here, we'd be tempted to say
    # "the market will resolve in favor of X".
    # Instead, we FALL BACK to slug-based context.
    return build_slug_fallback_statement(slug, description, token_name, resolution_date)


# -------------------------
# Main routine
# -------------------------

def prepare_nli_dataset(json_path: str):
    """
    Load Polymarket-style JSON and produce NLI-ready samples.

    Each sample:
      {
        "id":        market_id,
        "token":     token_name,
        "statement": canonical_nli_statement,
        "price":     token_price,
      }

    Deduplicates markets by (question + description).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen = set()
    samples = {
        "id": [],
        "token": [],
        "statement": [],
        "price": [],
    }

    for market_id, entry in data.items():
        question = entry.get("question", "").strip()
        description = entry.get("description", "")
        slug = entry.get("market_slug", "")

        # Dedup by question + description
        key_norm = normalize_text(question) + "||" + normalize_text(description)
        h = md5(key_norm.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        resolution_date = extract_resolution_date(entry)
        token_names = [t["outcome"] for t in entry["tokens"]]

        for t in entry["tokens"]:
            token_name = t["outcome"]
            price = t["price"]

            statement = build_statement_for_token(
                question=question,
                slug=slug,
                description=description,
                token_name=token_name,
                resolution_date=resolution_date,
                all_token_names=token_names,
            )

            samples["id"].append(market_id)
            samples["token"].append(token_name)
            samples["statement"].append(statement)
            samples["price"].append(price)

    return pd.DataFrame(samples)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python nli_preprocess.py input.json output.csv")
        sys.exit(1)

    df = prepare_nli_dataset(sys.argv[1])
    df.to_csv(sys.argv[2])