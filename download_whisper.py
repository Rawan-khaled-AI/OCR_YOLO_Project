from faster_whisper import WhisperModel

# ده هيجبره ينزّل الموديل مرة واحدة
WhisperModel("medium", device="cuda", compute_type="float16")
print("Downloaded and ready ✅")