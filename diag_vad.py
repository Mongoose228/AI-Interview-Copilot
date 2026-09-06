"""Test Silero VAD v4 ONNX model with correct state format (h, c)."""
import numpy as np
import onnxruntime as ort

model_path = "context/silero_vad_v4.onnx"
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# v4 state: separate h and c, shape (2, 1, 64) each
h = np.zeros((2, 1, 64), dtype=np.float32)
c = np.zeros((2, 1, 64), dtype=np.float32)
sr = np.array([16000], dtype=np.int64)

# Test with 300Hz tone (simulated speech)
t = np.arange(512) / 16000.0
tone = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32).reshape(1, 512)

print("=== TEST: 300Hz tone ===")
for i in range(20):
    out, h, c = sess.run(None, {'input': tone, 'h': h, 'c': c, 'sr': sr})
    if i % 5 == 0 or i == 19:
        print(f"  Chunk {i}: Prob: {out[0][0]:.6f}")

# Test with silence
print("\n=== TEST: Silence ===")
h = np.zeros((2, 1, 64), dtype=np.float32)
c = np.zeros((2, 1, 64), dtype=np.float32)
silence = np.zeros((1, 512), dtype=np.float32)
for i in range(5):
    out, h, c = sess.run(None, {'input': silence, 'h': h, 'c': c, 'sr': sr})
    print(f"  Chunk {i}: Prob: {out[0][0]:.6f}")

# Test with noise
print("\n=== TEST: Random noise ===")
h = np.zeros((2, 1, 64), dtype=np.float32)
c = np.zeros((2, 1, 64), dtype=np.float32)
rng = np.random.default_rng(42)
for i in range(20):
    noise = (rng.random(512).astype(np.float32) * 0.4 - 0.2).reshape(1, 512)
    out, h, c = sess.run(None, {'input': noise, 'h': h, 'c': c, 'sr': sr})
    if i % 5 == 0 or i == 19:
        print(f"  Chunk {i}: Prob: {out[0][0]:.6f}")
