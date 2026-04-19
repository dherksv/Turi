import re


# ── keyword banks ─────────────────────────────────────────────

QUESTION_STARTERS = (
    "what", "who", "when", "where", "why", "how",
    "is", "are", "was", "were", "will", "would",
    "can", "could", "should", "do", "does", "did",
    "which", "whose", "whom"
)

COMMAND_VERBS = {
    # scheduling
    "remind":    "set_reminder",
    "schedule":  "set_reminder",
    "set":       "set_reminder",
    "add":       "set_reminder",
    # search
    "search":    "web_search",
    "find":      "web_search",
    "look up":   "web_search",
    "google":    "web_search",
    "browse":    "web_search",
    # music
    "play":      "play_music",
    "pause":     "play_music",
    "stop":      "play_music",
    "skip":      "play_music",
    # messaging
    "send":      "send_message",
    "message":   "send_message",
    "text":      "send_message",
    "call":      "make_call",
    # calendar
    "book":      "create_event",
    "create":    "create_event",
    "cancel":    "create_event",
    # system
    "open":      "open_app",
    "launch":    "open_app",
    "show":      "show_info",
    "list":      "show_info",
    "read":      "show_info",
}

CHITCHAT_PATTERNS = [
    r"^(hi|hey|hello|howdy|sup|yo)[\s!.]*$",
    r"^(bye|goodbye|see you|cya|later)[\s!.]*$",
    r"^(thanks|thank you|thx|ty)[\s!.]*$",
    r"^(ok|okay|sure|alright|got it|noted)[\s!.]*$",
    r"^how are you",
    r"^what('s| is) up",
    r"^(good (morning|afternoon|evening|night))[\s!.]*$",
    r"^(yes|no|nope|yep|yeah)[\s!.]*$",
]

TIME_PATTERNS = [
    r"\b(today|tomorrow|tonight|this (morning|afternoon|evening))\b",
    r"\b(at \d{1,2}(:\d{2})?\s*(am|pm)?)\b",
    r"\b(in \d+ (minutes?|hours?|days?))\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b",
]


# ── classifier ────────────────────────────────────────────────

def classify(normalized: dict) -> dict:
    """
    Takes a normalized payload, returns it enriched with:
      - intent_type : "command" | "question" | "chat"
      - tool        : which tool to call (or None)
      - confidence  : 0.0 – 1.0
      - entities    : extracted values (time refs, etc.)
      - raw_text    : original cleaned text
    """
    text  = normalized["text"]
    lower = text.lower().strip()
    words = lower.split()
    first = words[0] if words else ""

    entities = _extract_entities(lower)

    # ── chitchat — check first, short-circuits everything ────
    for pattern in CHITCHAT_PATTERNS:
        if re.match(pattern, lower):
            return _result("chat", None, 0.95, entities, normalized)

    # ── very short input — treat as chat ────────────────────
    if normalized["word_count"] <= 2 and not lower.endswith("?"):
        return _result("chat", None, 0.75, entities, normalized)

    # ── explicit question mark ───────────────────────────────
    if text.strip().endswith("?"):
        return _result("question", None, 0.90, entities, normalized)

    # ── starts with question word ────────────────────────────
    if first in QUESTION_STARTERS:
        return _result("question", None, 0.85, entities, normalized)

    # ── command verb match ───────────────────────────────────
    # check first word
    if first in COMMAND_VERBS:
        tool = COMMAND_VERBS[first]
        conf = 0.90
        # boost confidence if time entity found for scheduling tools
        if tool == "set_reminder" and entities.get("time_refs"):
            conf = 0.95
        return _result("command", tool, conf, entities, normalized)

    # check two-word phrases like "look up"
    if normalized["word_count"] >= 2:
        two_words = " ".join(words[:2])
        if two_words in COMMAND_VERBS:
            return _result("command", COMMAND_VERBS[two_words],
                           0.88, entities, normalized)

    # ── implicit command — has time reference + action-like ──
    if entities.get("time_refs") and normalized["word_count"] >= 3:
        return _result("command", "set_reminder", 0.65, entities, normalized)

    # ── default: treat as question if long enough ────────────
    if normalized["word_count"] >= 4:
        return _result("question", None, 0.70, entities, normalized)

    # ── fallback: chat ───────────────────────────────────────
    return _result("chat", None, 0.60, entities, normalized)


def _extract_entities(text: str) -> dict:
    entities = {}

    # time references
    time_refs = []
    for pattern in TIME_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            time_refs.extend(
                [m if isinstance(m, str) else m[0] for m in matches]
            )
    if time_refs:
        entities["time_refs"] = list(set(time_refs))

    # detect if message mentions a contact name (simple heuristic)
    # "send X a message" or "message X"
    contact_match = re.search(
        r'\b(send|message|text|call)\s+([A-Z][a-z]+)', text, re.IGNORECASE
    )
    if contact_match:
        entities["contact"] = contact_match.group(2)

    return entities


def _result(
    intent_type: str,
    tool:        str | None,
    confidence:  float,
    entities:    dict,
    normalized:  dict
) -> dict:
    return {
        "intent_type": intent_type,
        "tool":        tool,
        "confidence":  confidence,
        "entities":    entities,
        "text":        normalized["text"],
        "word_count":  normalized["word_count"],
        "timestamp":   normalized["timestamp"],
    }