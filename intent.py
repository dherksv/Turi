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
    "remind":     "set_reminder",
    "schedule":   "set_reminder",
    "set":        "set_reminder",
    "add":        "set_reminder",

    # search
    "search":     "web_search",
    "find":       "web_search",
    "look":       "web_search",
    "google":     "web_search",
    "browse":     "web_search",
    "lookup":     "web_search",

    # shopping
    "buy":        "amazon_search",
    "shop":       "amazon_search",
    "order":      "amazon_search",
    "purchase":   "amazon_search",
    "get":        "amazon_search",
    "need":       "amazon_search",
    "want":       "amazon_search",
    "suggest":    "amazon_search",
    "recommend":  "amazon_search",

    # youtube / media
    "play":       "youtube_search",
    "watch":      "youtube_search",
    "stream":     "youtube_search",
    "listen":     "youtube_search",
    "show":       "youtube_search",
    "put":        "youtube_search",
    "queue":      "youtube_search",

    # files
    "open":       "open_file",
    "launch":     "open_file",
    "read":       "file_search",
    "load":       "open_file",

    # messaging
    "send":       "send_message",
    "message":    "send_message",
    "text":       "send_message",
    "call":       "make_call",

    # calendar
    "book":       "create_event",
    "create":     "create_event",
    "cancel":     "create_event",

    # system
    "list":       "show_info",
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
    text  = normalized["text"]
    lower = text.lower().strip()
    words = lower.split()
    first = words[0] if words else ""

    entities = _extract_entities(lower)

    # ── chitchat ──────────────────────────────────────────────
    for pattern in CHITCHAT_PATTERNS:
        if re.match(pattern, lower):
            return _result("chat", None, 0.95, entities, normalized)

    # ── very short ────────────────────────────────────────────
    if normalized["word_count"] <= 2 and not lower.endswith("?"):
        return _result("chat", None, 0.75, entities, normalized)

    # ── scan ALL words for command verbs BEFORE question check ─
    # This catches: "can you open...", "could you play...",
    # "I need to find...", "please search..."
    for i, word in enumerate(words):
        if word in COMMAND_VERBS:
            tool = COMMAND_VERBS[word]
            # make sure it's not a false positive
            # (e.g. "what is the best..." — "best" not a verb)
            conf = 0.90
            if i > 0:
                conf = 0.85  # slightly lower if not first word
            if tool == "set_reminder" and entities.get("time_refs"):
                conf = 0.95
            return _result("command", tool, conf, entities, normalized)

    # ── two-word phrases ──────────────────────────────────────
    two_word_map = {
        "look up":    "web_search",
        "search for": "web_search",
        "find me":    "amazon_search",
        "buy me":     "amazon_search",
        "get me":     "amazon_search",
        "play me":    "youtube_search",
        "show me":    "youtube_search",
        "open file":  "open_file",
        "open the":   "open_file",
        "read file":  "file_search",
        "put on":     "youtube_search",
        "i need":     "amazon_search",
        "i want":     "amazon_search",
        "i am looking": "amazon_search",
        "looking for": "amazon_search",
    }
    # check all consecutive word pairs, not just first two
    for i in range(len(words) - 1):
        pair = f"{words[i]} {words[i+1]}"
        if pair in two_word_map:
            return _result(
                "command", two_word_map[pair],
                0.88, entities, normalized
            )

    # three-word phrases
    three_word_map = {
        "i am looking": "amazon_search",
        "can you open": "open_file",
        "can you play": "youtube_search",
        "can you find": "web_search",
        "can you search": "web_search",
        "can you show": "youtube_search",
        "can you buy":  "amazon_search",
        "can you list": "file_search",
        "could you open": "open_file",
        "could you play": "youtube_search",
        "could you find": "web_search",
        "please open":   "open_file",
        "please find":   "web_search",
        "please play":   "youtube_search",
        "please search": "web_search",
    }
    for i in range(len(words) - 2):
        triple = f"{words[i]} {words[i+1]} {words[i+2]}"
        if triple in three_word_map:
            return _result(
                "command", three_word_map[triple],
                0.87, entities, normalized
            )

    # ── explicit question mark ────────────────────────────────
    if text.strip().endswith("?"):
        return _result("question", None, 0.90, entities, normalized)

    # ── starts with question word ─────────────────────────────
    if first in QUESTION_STARTERS:
        return _result("question", None, 0.85, entities, normalized)

    # ── implicit command with time reference ──────────────────
    if entities.get("time_refs") and normalized["word_count"] >= 3:
        return _result("command", "set_reminder", 0.65,
                       entities, normalized)

    # ── default ───────────────────────────────────────────────
    if normalized["word_count"] >= 4:
        return _result("question", None, 0.70, entities, normalized)

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