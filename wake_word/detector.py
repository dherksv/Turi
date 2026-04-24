"""
Wake word detector using custom hey_turi.onnx model.
Architecture: CNN + GRU trained on mel spectrograms.
Input:  mel_spectrogram (1, 1, 40, T_frames)
Output: logit (single float) → sigmoid → probability
"""

import threading
import asyncio
import numpy as np
from pathlib import Path

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[WAKE] pyaudio not installed")

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("[WAKE] onnxruntime not installed — run: pip install onnxruntime")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[WAKE] librosa not installed — run: pip install librosa")

from debug_logger import log_event

# ── must match your training config exactly ───────────────────
SAMPLE_RATE  = 16000
N_MELS       = 40
HOP_LENGTH   = 160       # 10ms hop at 16kHz
WIN_LENGTH   = 400       # 25ms window
N_FFT        = 512
CLIP_SECONDS = 1.5       # clip length used during training
CLIP_SAMPLES = int(CLIP_SECONDS * SAMPLE_RATE)   # 24000

# detection config
CHUNK_DURATION_MS = 500   # run inference every 500ms
CHUNK_SAMPLES     = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 8000
BUFFER_SECONDS    = 2.0   # rolling audio buffer
BUFFER_SAMPLES    = int(SAMPLE_RATE * BUFFER_SECONDS)  # 32000

THRESHOLD     = 0.5
COOLDOWN_SEC  = 1.5   # seconds to wait after detection before listening again

WAKE_MODEL_PATH = Path("wake_word/models/hey_turi.onnx")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def audio_to_mel(
    audio: np.ndarray,
    sr:    int = SAMPLE_RATE
) -> np.ndarray:
    """
    Convert raw audio (int16 or float32) to mel spectrogram.
    Matches your training preprocessing exactly.
    Returns shape: (1, 1, N_MELS, T_frames) — ready for ONNX input.
    """
    # ensure float32 normalized to -1..1
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # trim or pad to CLIP_SAMPLES
    if len(audio) > CLIP_SAMPLES:
        audio = audio[-CLIP_SAMPLES:]   # take most recent samples
    elif len(audio) < CLIP_SAMPLES:
        pad   = CLIP_SAMPLES - len(audio)
        audio = np.pad(audio, (pad, 0))  # pad at start

    # compute mel spectrogram using librosa
    if LIBROSA_AVAILABLE:
        mel = librosa.feature.melspectrogram(
            y          = audio,
            sr         = sr,
            n_fft      = N_FFT,
            hop_length = HOP_LENGTH,
            win_length = WIN_LENGTH,
            n_mels     = N_MELS,
            fmin       = 0.0,
            fmax       = sr / 2
        )
        # log mel
        mel = librosa.power_to_db(mel, ref=np.max)

        # normalize to 0..1
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    else:
        # fallback: manual mel without librosa
        mel = _manual_mel(audio, sr)

    # shape: (N_MELS, T_frames) → (1, 1, N_MELS, T_frames)
    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


def _manual_mel(
    audio: np.ndarray,
    sr:    int
) -> np.ndarray:
    """Minimal mel spectrogram without librosa."""
    # STFT
    frames   = []
    for start in range(0, len(audio) - WIN_LENGTH, HOP_LENGTH):
        frame    = audio[start:start + WIN_LENGTH]
        frame    = frame * np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(frame, n=N_FFT)) ** 2
        frames.append(spectrum)

    if not frames:
        return np.zeros((N_MELS, 1), dtype=np.float32)

    stft = np.array(frames).T  # (n_fft//2+1, T)

    # mel filterbank
    mel_fb = _mel_filterbank(N_MELS, N_FFT, sr)
    mel    = mel_fb @ stft          # (N_MELS, T)
    mel    = np.log(mel + 1e-8)

    # normalize
    mel    = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    return mel.astype(np.float32)


def _mel_filterbank(
    n_mels: int,
    n_fft:  int,
    sr:     int
) -> np.ndarray:
    """Build mel filterbank matrix."""
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    low_mel  = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = mel_to_hz(mel_pts)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fbank    = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_m = bin_pts[m - 1]
        f_m   = bin_pts[m]
        f_m_p = bin_pts[m + 1]
        for k in range(f_m_m, f_m):
            if f_m != f_m_m:
                fbank[m-1, k] = (k - f_m_m) / (f_m - f_m_m)
        for k in range(f_m, f_m_p):
            if f_m_p != f_m:
                fbank[m-1, k] = (f_m_p - k) / (f_m_p - f_m)
    return fbank


class WakeWordDetector:
    """
    Continuously listens on microphone.
    Maintains a rolling audio buffer.
    Every CHUNK_DURATION_MS, runs mel+ONNX inference.
    Calls on_detected(score) when hey_turi probability > threshold.
    """

    def __init__(
        self,
        on_detected = None,
        threshold:    float = THRESHOLD
    ):
        self.on_detected = on_detected
        self.threshold   = threshold
        self._running    = False
        self._thread     = None
        self._session    = None
        self._pa         = None
        # rolling buffer: holds last BUFFER_SECONDS of audio
        self._buffer     = np.zeros(
            BUFFER_SAMPLES, dtype=np.float32
        )
        self._last_detection = 0.0   # timestamp for cooldown

    def _load_model(self):
        if not WAKE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {WAKE_MODEL_PATH}\n"
                f"Place hey_turi.onnx in wake_word/models/"
            )

        print(f"[WAKE] loading: {WAKE_MODEL_PATH}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._session = ort.InferenceSession(
            str(WAKE_MODEL_PATH),
            sess_options = opts,
            providers    = ["CPUExecutionProvider"]
        )

        # verify input/output names match training
        inp = self._session.get_inputs()[0]
        out = self._session.get_outputs()[0]
        print(f"[WAKE] model input:  {inp.name} {inp.shape}")
        print(f"[WAKE] model output: {out.name} {out.shape}")
        print(f"[WAKE] model loaded — threshold={self.threshold}")

    def _infer(self, audio_buffer: np.ndarray) -> float:
        """Run one inference pass. Returns probability 0..1."""
        mel   = audio_to_mel(audio_buffer)
        logit = self._session.run(
            None,
            {"mel_spectrogram": mel}
        )[0]
        # logit shape: (1,) or scalar
        raw   = float(logit.flat[0])
        prob  = sigmoid(raw)
        return prob

    def start(self):
        if not PYAUDIO_AVAILABLE:
            print("[WAKE] pyaudio missing — skipping")
            return
        if not ORT_AVAILABLE:
            print("[WAKE] onnxruntime missing — skipping")
            return

        try:
            self._load_model()
        except Exception as e:
            print(f"[WAKE] failed to load model: {e}")
            return

        self._running = True
        self._thread  = threading.Thread(
            target = self._listen_loop,
            daemon = True,
            name   = "turi-wake-word"
        )
        self._thread.start()
        print("[WAKE] detector started")
        print(f"[WAKE] listening for 'Hey Turi' "
              f"(threshold={self.threshold})")

    def stop(self):
        self._running = False
        print("[WAKE] detector stopped")

    def _listen_loop(self):
        import time

        try:
            self._pa  = pyaudio.PyAudio()
            stream    = self._pa.open(
                rate              = SAMPLE_RATE,
                channels          = 1,
                format            = pyaudio.paInt16,
                input             = True,
                frames_per_buffer = CHUNK_SAMPLES
            )
            print("[WAKE] microphone open")

            while self._running:
                try:
                    # read one chunk
                    raw   = stream.read(
                        CHUNK_SAMPLES,
                        exception_on_overflow=False
                    )
                    chunk = (
                        np.frombuffer(raw, dtype=np.int16)
                        .astype(np.float32) / 32768.0
                    )

                    # update rolling buffer
                    self._buffer = np.roll(
                        self._buffer, -len(chunk)
                    )
                    self._buffer[-len(chunk):] = chunk

                    # cooldown check
                    now = time.time()
                    if now - self._last_detection < COOLDOWN_SEC:
                        continue

                    # run inference on rolling buffer
                    prob = self._infer(self._buffer)

                    if prob >= self.threshold:
                        self._last_detection = now
                        print(
                            f"[WAKE] ✓ Hey Turi detected! "
                            f"prob={prob:.3f}"
                        )
                        log_event(
                            "wake_detected", "wake_word",
                            {"prob": prob,
                             "threshold": self.threshold}
                        )
                        if self.on_detected:
                            self.on_detected(prob)

                except OSError as e:
                    print(f"[WAKE] audio read error: {e}")
                    break
                except Exception as e:
                    print(f"[WAKE] inference error: {e}")
                    continue

            stream.stop_stream()
            stream.close()
            self._pa.terminate()

        except Exception as e:
            print(f"[WAKE] listener failed: {e}")
            self._running = False


# ── global singleton ──────────────────────────────────────────

_detector:  WakeWordDetector | None = None
_callbacks: list                    = []
_loop:      asyncio.AbstractEventLoop | None = None


def register_callback(fn):
    _callbacks.append(fn)


def _on_detected(prob: float):
    for fn in _callbacks:
        try:
            if _loop and _loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    _call(fn, prob), _loop
                )
            else:
                fn(prob)
        except Exception as e:
            print(f"[WAKE] callback error: {e}")


async def _call(fn, prob):
    try:
        if asyncio.iscoroutinefunction(fn):
            await fn(prob)
        else:
            fn(prob)
    except Exception as e:
        print(f"[WAKE] async callback error: {e}")


def start_detector(
    threshold:   float = THRESHOLD,
    event_loop:  asyncio.AbstractEventLoop = None
):
    global _detector, _loop
    _loop     = event_loop
    _detector = WakeWordDetector(
        on_detected = _on_detected,
        threshold   = threshold
    )
    _detector.start()


def stop_detector():
    global _detector
    if _detector:
        _detector.stop()
        _detector = None