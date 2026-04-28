"""
Wake Word Evaluation
Tests detection accuracy using pre-recorded audio samples.
Measures FAR (False Accept Rate) and FRR (False Reject Rate).
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, '..')

from wake_word.detector import audio_to_mel
import onnxruntime as ort


# ✅ Set correct model path
WAKE_MODEL_PATH = Path(
    r"C:\Users\Hp\Documents\Arch\assistant\wake_word\models\hey_turi.onnx"
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_audio_file(
    session: ort.InferenceSession,
    wav_path: str,
    threshold: float = 0.5
) -> dict:
    import wave

    with wave.open(wav_path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        framerate = wf.getframerate()

    mel = audio_to_mel(audio, sr=framerate)
    logit = session.run(None, {"mel_spectrogram": mel})[0]

    # ✅ Ensure Python float
    prob = float(sigmoid(float(logit.flat[0])))

    return {
        "file": os.path.basename(wav_path),
        "prob": round(prob, 4),                # Python float
        "detected": bool(prob >= threshold)   # Python bool ✅ FIX
    }


def run_evaluation(
    positive_dir: str = "audio/positive",
    negative_dir: str = "audio/negative",
    threshold: float = 0.5
):
    print(f"\nWake Word Evaluation")
    print(f"Model    : {WAKE_MODEL_PATH}")
    print(f"Threshold: {threshold}")
    print(f"{'═'*50}")

    # ✅ Debug (optional but useful)
    print("Resolved model path:", WAKE_MODEL_PATH.resolve())
    print("Exists:", WAKE_MODEL_PATH.exists())

    if not WAKE_MODEL_PATH.exists():
        print("ERROR: Model not found")
        return

    session = ort.InferenceSession(
        str(WAKE_MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    tp = fp = tn = fn = 0
    results = {"positive": [], "negative": []}

    # -----------------------
    # Positive samples
    # -----------------------
    if os.path.exists(positive_dir):
        print(f"\nPositive samples (should detect):")

        for fname in sorted(os.listdir(positive_dir)):
            if not fname.lower().endswith(".wav"):  # ✅ FIX
                continue

            path = os.path.join(positive_dir, fname)
            result = evaluate_audio_file(session, path, threshold)
            results["positive"].append(result)

            if result["detected"]:
                tp += 1
                status = "✓ DETECTED"
            else:
                fn += 1
                status = "✗ MISSED"

            print(f"  {status}  prob={result['prob']:.3f}  {fname}")
    else:
        print(f"\nNo positive samples found at {positive_dir}")

    # -----------------------
    # Negative samples
    # -----------------------
    if os.path.exists(negative_dir):
        print(f"\nNegative samples (should NOT detect):")

        for fname in sorted(os.listdir(negative_dir)):
            if not fname.lower().endswith(".wav"):  # ✅ FIX
                continue

            path = os.path.join(negative_dir, fname)
            result = evaluate_audio_file(session, path, threshold)
            results["negative"].append(result)

            if result["detected"]:
                fp += 1
                status = "✗ FALSE TRIGGER"
            else:
                tn += 1
                status = "✓ CORRECT REJECT"

            print(f"  {status}  prob={result['prob']:.3f}  {fname}")
    else:
        print(f"\nNo negative samples at {negative_dir}")

    # -----------------------
    # Metrics
    # -----------------------
    total_pos = tp + fn
    total_neg = tn + fp

    far = fp / total_neg if total_neg > 0 else 0.0
    frr = fn / total_pos if total_pos > 0 else 0.0
    acc = (tp + tn) / (total_pos + total_neg) if (total_pos + total_neg) > 0 else 0.0

    print(f"\n{'═'*50}")
    print(f"WAKE WORD RESULTS")
    print(f"{'─'*50}")
    print(f"  True Positives  (TP) : {tp}")
    print(f"  False Positives (FP) : {fp}")
    print(f"  True Negatives  (TN) : {tn}")
    print(f"  False Negatives (FN) : {fn}")
    print(f"  FAR (False Accept)   : {far:.3f}  ({far*100:.1f}%)")
    print(f"  FRR (False Reject)   : {frr:.3f}  ({frr*100:.1f}%)")
    print(f"  Accuracy             : {acc:.3f}  ({acc*100:.1f}%)")

    # ✅ FULL JSON SAFE REPORT
    report = {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "far": float(round(far, 4)),
        "frr": float(round(frr, 4)),
        "accuracy": float(round(acc, 4)),
        "samples": results
    }

    os.makedirs("results", exist_ok=True)

    with open("results/wake_word_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to results/wake_word_results.json")

    return report


if __name__ == "__main__":
    run_evaluation()