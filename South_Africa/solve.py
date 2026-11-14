"""Sigmoid South Africa - "Ngiyakuzwa?"  End-to-end pipeline driver.
Heavy stages run on the GPU box (over ssh, ~/sigmoid_southafrica, venv with
torch/transformers/demucs/noisereduce/soundfile/librosa). Run from this folder:  python solve.py
"""
import subprocess, os
H = "user@gpu-host"; R = "~/sigmoid_southafrica"
def ssh(cmd): subprocess.run(["ssh", "-o", "BatchMode=yes", H, f"cd {R} && {cmd}"], check=True)
def scp(a, b): subprocess.run(["scp", "-q", a, b], check=True)

scp("data/voice_note.wav", f"{H}:{R}/")
for f in os.listdir("scripts"):
    scp(f"scripts/{f}", f"{H}:{R}/")
# 1. classical: notch the two vuvuzela harmonic combs (217.4 / 233.1 Hz), bandpass, noisereduce
ssh("venv/bin/python denoise.py voice_note.wav out")
# 2. neural vocal separation (Demucs htdemucs_ft) on the notched audio
ssh("venv/bin/python separate.py out/stage1_notched.wav out/demucs_notched.wav htdemucs_ft")
# 3. ASR: MMS-1b-all zul on pause-segmented utterances -> search for 'ikhodi'
ssh("venv/bin/python mms_vad.py out/demucs_notched.wav out/demucs_notched.wav out/mmsvad_demucs.txt")
ssh("grep -n 'kh[ou][dtj]i\\|ikodi\\|ikuti' out/mmsvad_demucs.txt || true")
# 4. zoom: frame-level CTC, candidate scoring, and SeamlessM4T second opinion on the key region
ssh("venv/bin/python zoom_ctc.py out/demucs_notched.wav 437.4 442.0 out/cleaned.wav out/stage1_notched.wav")
ssh("venv/bin/python score_candidates_ctx.py 437.4 442.95 0 out/demucs_notched.wav out/stage1_notched.wav out/cleaned.wav")
ssh("venv/bin/python seamless_asr.py 436.5 443.2 0 out/demucs_notched.wav")
os.makedirs("out", exist_ok=True)
for f in ["demucs_notched.wav", "cleaned.wav", "stage1_notched.wav", "mmsvad_demucs.txt"]:
    scp(f"{H}:{R}/out/{f}", "out/")
subprocess.run(["python3", "scripts/make_outputs.py"], check=True)
print("Word after ikhodi: see transcript.txt and key_clip.wav")
