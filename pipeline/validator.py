import json
from dataclasses import dataclass
from agents import validator_agent
from agents.base import BaseAgent


@dataclass
class ValidationResult:
    passed:     bool
    verdict:    str      # "pass" | "fix" | "reject"
    reason:     str
    confidence: float
    fixed:      dict | None = None


# tools that always need extra scrutiny
HIGH_RISK_TOOLS = {
    "send_message", "make_call", "amazon_search",
    "open_file", "file_search"
}

# patterns that are always rejected — no model needed
INSTANT_REJECT = [
    "ignore previous",
    "disregard instructions",
    "system prompt",
    "jailbreak",
    "rm -rf",
    "<script",
    "drop table",
    "delete from",
]


async def validate_intent(
    user_text: str,
    intent:    dict,
    use_llm:   bool = True
) -> ValidationResult:
    """
    Three-layer validation:
    1. Instant reject — dangerous patterns
    2. Rule check — schema and confidence
    3. LLM crosscheck — Phi-4 Mini semantic check
    """

    # layer 1 — instant reject
    combined = (user_text + str(intent)).lower()
    for pattern in INSTANT_REJECT:
        if pattern in combined:
            return ValidationResult(
                passed     = False,
                verdict    = "reject",
                reason     = f"dangerous pattern: '{pattern}'",
                confidence = 0.0
            )

    # layer 2 — rule checks
    rule_result = _rule_check(intent)
    if not rule_result.passed:
        return rule_result

    # layer 3 — LLM crosscheck (skip for simple chat)
    if not use_llm or intent.get("intent_type") == "chat":
        return ValidationResult(
            passed     = True,
            verdict    = "pass",
            reason     = "rules passed, chat skips LLM check",
            confidence = intent.get("confidence", 0.8)
        )

    # only call validator LLM if agent is online
    online = await validator_agent.is_online()
    if not online:
        print("[VALIDATOR] offline — skipping LLM check")
        return ValidationResult(
            passed     = True,
            verdict    = "pass",
            reason     = "validator offline — rule-only pass",
            confidence = intent.get("confidence", 0.7)
        )

    return await _llm_check(user_text, intent)


def _rule_check(intent: dict) -> ValidationResult:
    confidence = intent.get("confidence", 0)
    tool       = intent.get("tool")
    args       = intent.get("args", {})

    # confidence floor
    if confidence < 0.55:
        return ValidationResult(
            False, "reject",
            f"confidence {confidence:.2f} below floor",
            confidence
        )

    # tool exists check
    from tools.registry import TOOL_MAP
    if tool and tool not in TOOL_MAP:
        return ValidationResult(
            False, "reject",
            f"unknown tool: {tool}",
            confidence
        )

    # args must be dict
    if not isinstance(args, dict):
        return ValidationResult(
            False, "fix",
            "args must be a dict",
            confidence,
            fixed={**intent, "args": {}}
        )

    return ValidationResult(True, "pass", "rules ok", confidence)


async def _llm_check(
    user_text: str,
    intent:    dict
) -> ValidationResult:
    prompt = f"""Check this AI intent against the user message.

User said: "{user_text}"

Intent extracted:
{json.dumps(intent, indent=2)}

Respond with ONLY this JSON:
{{
  "valid": true or false,
  "verdict": "pass" or "fix" or "reject",
  "reason": "one sentence",
  "confidence": 0.0 to 1.0,
  "corrected_intent": null
}}

Rules:
- pass: intent matches user message well
- fix: intent is close but needs correction
- reject: intent completely wrong or unsafe
- Never pass if tool could harm user with these args"""

    try:
        raw = await validator_agent.chat(
            [{"role": "user", "content": prompt}]
        )
        # extract JSON from response
        raw = raw.strip()
        # find JSON block
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("no JSON in response")

        result = json.loads(raw[start:end])

        return ValidationResult(
            passed     = result.get("valid", False),
            verdict    = result.get("verdict", "reject"),
            reason     = result.get("reason", "unknown"),
            confidence = result.get("confidence", 0.0),
            fixed      = result.get("corrected_intent")
        )

    except Exception as e:
        print(f"[VALIDATOR] LLM check error: {e}")
        # fail safe — if validator errors, still pass with warning
        return ValidationResult(
            passed     = True,
            verdict    = "pass",
            reason     = f"validator error (fail-open): {e}",
            confidence = 0.6
        )