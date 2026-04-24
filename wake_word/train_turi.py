"""
Train a custom wake word model for "Hey Turi".
Generates synthetic audio samples using TTS,
trains a small neural network, exports to ONNX.
Run this once. Takes about 10-20 minutes on CPU.
"""

from openwakeword.train import train

train(
    # what you want to detect
    target_phrase       = "hey turi",

    # negative examples — things that sound similar
    # but should NOT trigger
    negative_phrases    = [
        "hey siri",
        "hey tory",
        "hey torii",
        "ok google",
        "alexa",
    ],

    # output directory for the trained model
    output_dir          = "wake_word/models",
    model_name          = "hey_turi",

    # training config
    n_samples           = 4000,   # synthetic samples to generate
    epochs              = 100,
    batch_size          = 32,
)

print("Training complete. Model saved to wake_word/models/hey_turi.onnx")