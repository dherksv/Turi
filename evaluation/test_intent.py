"""
Intent Classification Evaluation
Tests the rule-based classifier against labeled inputs.
Reports accuracy, precision, recall, F1 per class.
"""

import sys
import json
sys.path.insert(0, '..')

from normalizer import normalize
from intent     import classify

# ── labeled test dataset ──────────────────────────────────────
TEST_CASES = [
    # (input_text, expected_intent_type, expected_tool)
    # commands
    ("play some lofi music",              "command", "youtube_search"),
    ("put on a funny cat video",          "command", "youtube_search"),
    ("search for latest AI news",         "command", "web_search"),
    ("find me a good laptop under 50000", "command", "amazon_search"),
    ("buy wireless headset under 5000",   "command", "amazon_search"),
    ("remind me to call dentist at 9am",  "command", "set_reminder"),
    ("set reminder for meeting tomorrow", "command", "set_reminder"),
    ("open file explorer",                "command", "open_file"),
    ("open camera",                       "command", "open_app"),
    ("launch calculator",                 "command", "open_app"),
    ("can you open Dos_report.pdf",       "command", "open_file"),
    ("search for weather in Kerala",      "command", "web_search"),
    ("find bluetooth speaker under 2000", "command", "amazon_search"),
    ("play billie jean video",            "command", "youtube_search"),
    ("open notepad",                      "command", "open_app"),
    ("remind me to drink water in 1 hour","command", "set_reminder"),

    # questions
    ("what time is it",                   "question", None),
    ("what is machine learning",          "question", None),
    ("who is the current cm of kerala",   "question", None),
    ("how does a transformer work",       "question", None),
    ("what is the weather today",         "question", None),
    ("can you explain neural networks",   "question", None),
    ("what do you think about python",    "question", None),
    ("how many planets are there",        "question", None),

    # chat
    ("hello",                             "chat", None),
    ("hi there",                          "chat", None),
    ("thanks",                            "chat", None),
    ("okay got it",                       "chat", None),
    ("yes please",                        "chat", None),
    ("bye",                               "chat", None),
    ("good morning",                      "chat", None),
    ("ok",                                "chat", None),
]


def evaluate():
    results = {
        "command":  {"tp": 0, "fp": 0, "fn": 0},
        "question": {"tp": 0, "fp": 0, "fn": 0},
        "chat":     {"tp": 0, "fp": 0, "fn": 0},
    }

    tool_correct   = 0
    tool_total     = 0
    correct_intent = 0
    total          = len(TEST_CASES)
    errors         = []

    print(f"\n{'INPUT':<45} {'EXPECTED':<12} {'GOT':<12} {'TOOL':<20} {'OK'}")
    print("─" * 100)

    for text, exp_type, exp_tool in TEST_CASES:
        normalized = normalize(text)
        intent     = classify(normalized)

        got_type = intent["intent_type"]
        got_tool = intent["tool"]
        conf     = intent["confidence"]

        type_ok  = got_type == exp_type
        tool_ok  = (exp_tool is None) or (got_tool == exp_tool)
        overall  = type_ok and tool_ok

        if type_ok:
            correct_intent += 1
            results[exp_type]["tp"] += 1
        else:
            results[exp_type]["fn"] += 1
            results[got_type]["fp"] += 1
            errors.append({
                "input":    text,
                "expected": f"{exp_type}/{exp_tool}",
                "got":      f"{got_type}/{got_tool}",
                "conf":     conf
            })

        if exp_tool:
            tool_total += 1
            if got_tool == exp_tool:
                tool_correct += 1

        status = "✓" if overall else "✗"
        print(
            f"{text:<45} "
            f"{exp_type:<12} "
            f"{got_type:<12} "
            f"{str(got_tool):<20} "
            f"{status}"
        )

    # ── metrics ───────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("CLASSIFICATION REPORT")
    print("═" * 60)

    all_f1 = []
    for cls, counts in results.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)
        support   = tp + fn
        all_f1.append(f1)

        print(f"\n  {cls.upper()}")
        print(f"    Precision : {precision:.3f}")
        print(f"    Recall    : {recall:.3f}")
        print(f"    F1-score  : {f1:.3f}")
        print(f"    Support   : {support}")

    overall_acc  = correct_intent / total
    tool_acc     = tool_correct / tool_total if tool_total else 0
    macro_f1     = sum(all_f1) / len(all_f1)

    print(f"\n{'─'*60}")
    print(f"  Overall intent accuracy : {overall_acc:.3f}  ({correct_intent}/{total})")
    print(f"  Tool routing accuracy   : {tool_acc:.3f}  ({tool_correct}/{tool_total})")
    print(f"  Macro F1-score          : {macro_f1:.3f}")

    if errors:
        print(f"\n  Misclassified ({len(errors)}):")
        for e in errors:
            print(f"    ✗ \"{e['input']}\"")
            print(f"        expected {e['expected']} → got {e['got']}")

    # save results
    report = {
        "total":           total,
        "correct_intent":  correct_intent,
        "intent_accuracy": round(overall_acc, 4),
        "tool_accuracy":   round(tool_acc, 4),
        "macro_f1":        round(macro_f1, 4),
        "per_class":       {
            cls: {
                "precision": round(
                    r["tp"] / (r["tp"] + r["fp"])
                    if (r["tp"] + r["fp"]) > 0 else 0, 4
                ),
                "recall": round(
                    r["tp"] / (r["tp"] + r["fn"])
                    if (r["tp"] + r["fn"]) > 0 else 0, 4
                ),
            }
            for cls, r in results.items()
        },
        "errors": errors
    }

    with open("results/intent_classification_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Results saved to results/intent_classification_results.json")
    return report


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    evaluate()