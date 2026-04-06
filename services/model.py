import numpy as np
import tensorflow as tf

# single value output — below 0.5 = normal, above 0.5 = abnormal
THRESHOLD = 0.5

_interpreter = None


def load_model(model_path: str = "model/arrythmia.tflite"):
    """
    Load the TFLite model from disk.
    Called once at server startup.
    """
    global _interpreter

    try:
        _interpreter = tf.lite.Interpreter(model_path=model_path)
        _interpreter.allocate_tensors()

        input_details = _interpreter.get_input_details()
        output_details = _interpreter.get_output_details()

        print(f"✓ TFLite model loaded from {model_path}")
        print(f"✓ Input shape: {input_details[0]['shape']}")
        print(f"✓ Output shape: {output_details[0]['shape']}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise e


def get_interpreter():
    if _interpreter is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _interpreter


def predict_spectrogram(spectrogram: np.ndarray) -> float:
    """
    Run inference on a single 128x128 mel spectrogram.
    Returns a float between 0 and 1.
    0 = normal, 1 = abnormal.
    """
    interpreter = get_interpreter()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # reshape to (1, 128, 128, 1) — batch size of 1
    x = spectrogram.reshape(1, 128, 128, 1).astype(np.float32)

    # set input and run
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

    # get output — single float value
    output = interpreter.get_tensor(output_details[0]['index'])
    return float(output[0][0])


def predict(spectrograms: list) -> dict:
    """
    Run inference on all spectrograms and average the results.
    """
    if len(spectrograms) == 0:
        raise ValueError("No valid spectrograms to predict on.")

    all_scores = []

    for spec in spectrograms:
        score = predict_spectrogram(spec)
        all_scores.append(score)

    # average across all chunks
    avg_score = float(np.mean(all_scores))

    # classify
    result = "abnormal" if avg_score >= THRESHOLD else "normal"
    confidence = avg_score if result == "abnormal" else 1 - avg_score

    return {
        "result": result,
        "confidence": round(confidence, 4),
        "chunks_analyzed": len(spectrograms),
        "raw_score": round(avg_score, 4)
    }