import soundcard as sc

mics = sc.all_microphones(include_loopback=True)
mic = mics[0]
for m in mics:
    if m.isloopback:
        mic = m
        break

print("Using loopback:", mic.name)

try:
    with mic.recorder(samplerate=16000, channels=2) as recorder:
        data = recorder.record(numframes=1024)
        print("Success! Got data shape:", data.shape)
except Exception as e:
    print("Error:", e)
