import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from agents import monitor_agent

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# failure classification
TRANSIENT_KEYWORDS  = ["timeout", "connection", "network", "retry"]
LOGIC_KEYWORDS      = ["invalid", "schema", "missing", "parse", "json"]
FATAL_KEYWORDS      = ["auth", "403", "401", "permission", "not found"]


@dataclass
class TaskCheckpoint:
    task_id:    str
    session_id: str
    intent:     dict
    steps:      list = field(default_factory=list)
    status:     str  = "running"   # running|done|failed|recovering
    created_at: str  = ""
    updated_at: str  = ""

    def save(self):
        path = CHECKPOINT_DIR / f"{self.task_id}.json"
        self.updated_at = datetime.utcnow().isoformat()
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, task_id: str):
        path = CHECKPOINT_DIR / f"{task_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(**data)


@dataclass
class FailureAnalysis:
    error_type:     str    # "transient" | "logic" | "fatal" | "unknown"
    should_retry:   bool
    max_retries:    int
    recovery_plan:  str
    notify_user:    bool


async def analyze_failure(
    error:   str,
    context: dict
) -> FailureAnalysis:
    """
    Llama 3.2 1B analyzes what went wrong and how to recover.
    Falls back to rule-based if monitor is offline.
    """
    # fast rule-based first
    rule_result = _rule_classify(error)

    online = await monitor_agent.is_online()
    if not online:
        print("[MONITOR] offline — using rule-based analysis")
        return rule_result

    return await _llm_analyze(error, context, rule_result)


def _rule_classify(error: str) -> FailureAnalysis:
    lower = error.lower()

    if any(k in lower for k in TRANSIENT_KEYWORDS):
        return FailureAnalysis(
            error_type    = "transient",
            should_retry  = True,
            max_retries   = 3,
            recovery_plan = "exponential backoff retry",
            notify_user   = False
        )
    if any(k in lower for k in LOGIC_KEYWORDS):
        return FailureAnalysis(
            error_type    = "logic",
            should_retry  = True,
            max_retries   = 1,
            recovery_plan = "replan with error context",
            notify_user   = False
        )
    if any(k in lower for k in FATAL_KEYWORDS):
        return FailureAnalysis(
            error_type    = "fatal",
            should_retry  = False,
            max_retries   = 0,
            recovery_plan = "escalate to user",
            notify_user   = True
        )
    return FailureAnalysis(
        error_type    = "unknown",
        should_retry  = True,
        max_retries   = 1,
        recovery_plan = "single retry then escalate",
        notify_user   = False
    )


async def _llm_analyze(
    error:       str,
    context:     dict,
    rule_result: FailureAnalysis
) -> FailureAnalysis:
    prompt = f"""Analyze this AI system failure.

Error: {error}
Context: {json.dumps(context)[:300]}
Rule-based classification: {rule_result.error_type}

Respond ONLY with this JSON:
{{
  "error_type": "transient" or "logic" or "fatal" or "unknown",
  "should_retry": true or false,
  "max_retries": 0 to 3,
  "recovery_plan": "brief description",
  "notify_user": true or false
}}

transient = network/timeout issues, retry helps
logic = bad output/parsing, replan needed
fatal = auth/permissions/missing resource, user must intervene"""

    try:
        raw    = await monitor_agent.chat(
            [{"role": "user", "content": prompt}]
        )
        start  = raw.find('{')
        end    = raw.rfind('}') + 1
        result = json.loads(raw[start:end])

        return FailureAnalysis(
            error_type    = result.get("error_type",    "unknown"),
            should_retry  = result.get("should_retry",  True),
            max_retries   = result.get("max_retries",   1),
            recovery_plan = result.get("recovery_plan", "retry"),
            notify_user   = result.get("notify_user",   False)
        )
    except Exception as e:
        print(f"[MONITOR] LLM analysis error: {e}")
        return rule_result


async def execute_with_recovery(
    task_id:    str,
    session_id: str,
    intent:     dict,
    execute_fn,
    max_attempts: int = 3
) -> dict:
    """
    Execute a task with automatic failure recovery.
    Saves checkpoints so partial work isn't lost.
    """
    checkpoint = TaskCheckpoint(
        task_id    = task_id,
        session_id = session_id,
        intent     = intent,
        created_at = datetime.utcnow().isoformat()
    )
    checkpoint.save()

    last_error   = None
    attempt      = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"[MONITOR] attempt {attempt}/{max_attempts} "
              f"for task {task_id[:8]}")

        try:
            result = await execute_fn()
            checkpoint.status = "done"
            checkpoint.steps.append({
                "attempt": attempt,
                "status":  "success",
                "time":    datetime.utcnow().isoformat()
            })
            checkpoint.save()
            return {
                "status":   "success",
                "result":   result,
                "attempts": attempt
            }

        except Exception as e:
            last_error = str(e)
            print(f"[MONITOR] attempt {attempt} failed: {e}")

            checkpoint.steps.append({
                "attempt": attempt,
                "status":  "failed",
                "error":   last_error,
                "time":    datetime.utcnow().isoformat()
            })
            checkpoint.save()

            # analyze the failure
            analysis = await analyze_failure(
                last_error,
                {"intent": intent, "attempt": attempt}
            )

            print(f"[MONITOR] classified: {analysis.error_type} "
                  f"retry={analysis.should_retry}")

            if not analysis.should_retry:
                checkpoint.status = "failed"
                checkpoint.save()
                return {
                    "status":       "failed",
                    "error":        last_error,
                    "error_type":   analysis.error_type,
                    "notify_user":  True,
                    "attempts":     attempt
                }

            if attempt < max_attempts:
                # exponential backoff
                import asyncio
                delay = 2 ** (attempt - 1)
                print(f"[MONITOR] waiting {delay}s before retry")
                await asyncio.sleep(delay)

    checkpoint.status = "failed"
    checkpoint.save()
    return {
        "status":      "exhausted",
        "error":       last_error,
        "attempts":    attempt,
        "notify_user": True
    }