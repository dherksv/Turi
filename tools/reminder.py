import re
from datetime import datetime, timedelta
import pytz
import os
from dotenv import load_dotenv

load_dotenv()
TIMEZONE = os.getenv("USER_TIMEZONE", "Asia/Kolkata")


def parse_datetime(text: str) -> str | None:
    """
    Parse natural language time expressions into ISO datetime string.
    Returns None if no time found.
    """
    tz  = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    lower = text.lower()

    # ── relative: "in X minutes/hours/days" ──────────────────
    match = re.search(
        r'in\s+(\d+)\s+(minute|hour|day|week)s?', lower
    )
    if match:
        amount = int(match.group(1))
        unit   = match.group(2)
        delta  = {
            "minute": timedelta(minutes=amount),
            "hour":   timedelta(hours=amount),
            "day":    timedelta(days=amount),
            "week":   timedelta(weeks=amount),
        }[unit]
        return (now + delta).isoformat()

    # ── absolute time "at HH:MM am/pm" ───────────────────────
    time_match = re.search(
        r'at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', lower
    )

    # ── day reference ─────────────────────────────────────────
    day_offset = 0
    if "tomorrow"        in lower: day_offset = 1
    elif "day after"     in lower: day_offset = 2
    elif "next week"     in lower: day_offset = 7

    # named weekdays
    weekdays = {
        "monday":0,"tuesday":1,"wednesday":2,"thursday":3,
        "friday":4,"saturday":5,"sunday":6
    }
    for day_name, day_num in weekdays.items():
        if day_name in lower:
            days_ahead = (day_num - now.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7  # next occurrence
            day_offset = days_ahead
            break

    # build target date
    target_date = (now + timedelta(days=day_offset)).date()

    if time_match:
        hour   = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        period = time_match.group(3)

        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0

        target = tz.localize(
            datetime(target_date.year, target_date.month,
                     target_date.day, hour, minute)
        )
        return target.isoformat()

    # day reference but no time — default to 9am
    if day_offset > 0:
        target = tz.localize(
            datetime(target_date.year, target_date.month,
                     target_date.day, 9, 0)
        )
        return target.isoformat()

    # "tonight" → today 8pm
    if "tonight" in lower:
        target = tz.localize(
            datetime(now.year, now.month, now.day, 20, 0)
        )
        return target.isoformat()

    # "this morning/afternoon/evening"
    time_of_day = {
        "this morning":   8,
        "this afternoon": 14,
        "this evening":   18,
    }
    for phrase, hour in time_of_day.items():
        if phrase in lower:
            target = tz.localize(
                datetime(now.year, now.month, now.day, hour, 0)
            )
            return target.isoformat()

    return None  # could not parse


def extract_task(text: str) -> str:
    """
    Strip the scheduling command words to get the actual task.
    "remind me to call dentist tomorrow" → "call dentist"
    """
    cleaned = re.sub(
        r'^(remind\s+me\s+(to\s+)?|set\s+(a\s+)?reminder\s+(to\s+|for\s+)?'
        r'|schedule\s+(a\s+)?reminder\s+(to\s+|for\s+)?'
        r'|add\s+(a\s+)?reminder\s+(to\s+|for\s+)?)',
        '', text, flags=re.IGNORECASE
    ).strip()

    # also strip trailing time expression
    cleaned = re.sub(
        r'\s+(tomorrow|tonight|today|next\s+\w+|this\s+\w+'
        r'|on\s+\w+day|at\s+\d+.*|in\s+\d+.*)$',
        '', cleaned, flags=re.IGNORECASE
    ).strip()

    return cleaned if cleaned else text


def format_remind_at(iso_str: str) -> str:
    """Human-readable version of ISO datetime for confirmation."""
    tz  = pytz.timezone(TIMEZONE)
    dt  = datetime.fromisoformat(iso_str).astimezone(tz)
    return dt.strftime("%A, %d %B %Y at %I:%M %p")


def build_reminder_intent(intent: dict) -> dict:
    """
    Parse the full reminder intent from user text.
    Returns task, remind_at, and whether we have enough info.
    """
    text      = intent["text"]
    task      = extract_task(text)
    remind_at = parse_datetime(text)

    return {
        "task":       task,
        "remind_at":  remind_at,
        "has_time":   remind_at is not None,
        "raw_text":   text
    }