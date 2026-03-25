import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests
import io
import numpy as np
import tempfile
import textwrap
from openai import OpenAI
from faster_whisper import WhisperModel

# ----------------------------
# API Keys (يفضل تحطيهم في .env بدل الكود)
# ----------------------------
OCR_API_KEY = "K88928882188957"
OPENROUTER_API_KEY = (
    "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"
)

# ----------------------------
# Paths
# ----------------------------
WEIGHTS_PATH = r"E:\project_ocr\medicine.v3i.yolov8\runs\detect\train\weights\best.pt"


# ----------------------------
# Load YOLOv8 Model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO(WEIGHTS_PATH)


model = load_model()


# ----------------------------
# Load Whisper (faster-whisper) (cached)
# ----------------------------
@st.cache_resource
def load_whisper():
    # GPU (CUDA)
    return WhisperModel("medium", device="cuda", compute_type="float16")


whisper_model = load_whisper()


# ----------------------------
# OCR Function
# ----------------------------
def run_ocr_on_image(pil_image: Image.Image) -> str:
    url = "https://api.ocr.space/parse/image"

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    buffered.seek(0)

    files = {"file": ("image.png", buffered, "image/png")}
    data = {"apikey": OCR_API_KEY, "language": "eng"}

    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        result = response.json()
        text = result["ParsedResults"][0]["ParsedText"]
        return (text or "").strip()
    except Exception:
        return "OCR failed."


# ----------------------------
# Whisper Transcription (Voice -> Text)
# ----------------------------
def transcribe_audio_faster_whisper(audio_file) -> str:
    """
    audio_file: streamlit UploadedFile from st.audio_input
    """
    if audio_file is None:
        return ""

    audio_bytes = audio_file.getvalue()

    # streamlit غالباً بيرجع wav/webm حسب المتصفح
    suffix = ".wav"
    if getattr(audio_file, "name", None) and "." in audio_file.name:
        suffix = "." + audio_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = whisper_model.transcribe(
            tmp_path,
            language="en",  # خليها "en" لو عايزة إنجليزي
            beam_size=5,
            vad_filter=True,  # يقلل السكات/الفراغ
        )
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(text_parts).strip()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ----------------------------
# OpenRouter ChatBot Function
# ----------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def ask_about_drug_openrouter(active_ingredient: str, question: str) -> str:
    prompt = f"""
You are a professional medical assistant.
The active ingredient is: {active_ingredient}.
Answer medically and clearly in Arabic (Egyptian).
Provide GENERAL information only (not a diagnosis or prescription).
Include:
- Uses
- Side effects
- Contraindications
- Warnings
- Common interactions (if relevant)
- What conditions it treats

User question: {question}
"""
    completion = client.chat.completions.create(
        model="openai/gpt-5.1-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return completion.choices[0].message.content


# ----------------------------
# UI Helpers
# ----------------------------
def render_chat_bubble(role: str, text: str):
    with st.chat_message(role):
        st.markdown(text)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Medicine Detection & ChatBot", layout="wide")

st.title(
    "💊 Active Ingredient Detection System (YOLOv8 + OCR) + 💬 ChatBot (Text/Voice)"
)

with st.sidebar:
    st.header("⚙️ Settings")
    yolo_conf = st.slider("YOLO Confidence", 0.05, 0.95, 0.25, 0.05)
    ocr_lang = st.selectbox(
        "OCR Language", ["eng", "ara", "eng+ara"], index=0
    )  # متركناه هنا لو هتطوري OCR
    show_crops = st.checkbox("Show Crops", value=True)

    st.divider()
    st.caption("🎤 Voice uses faster-whisper (local GPU)")

uploaded_file = st.file_uploader("Upload Medicine Image", type=["jpg", "jpeg", "png"])

# Session state init
st.session_state.setdefault("results", None)
st.session_state.setdefault("image", None)
st.session_state.setdefault("active_ingredients", [])
st.session_state.setdefault(
    "chat_history", []
)  # list of dicts: {"role":..., "content":..., "ingredient":...}
st.session_state.setdefault("last_transcript", "")

# ----------------------------
# Upload + Detect
# ----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🔍 Run Detection"):
        results = model(image, conf=yolo_conf)
        result = results[0]
        plotted = result.plot()

        with col2:
            st.subheader("Detection Result")
            st.image(plotted, use_container_width=True)

        st.session_state["results"] = result
        st.session_state["image"] = image

# ----------------------------
# Crop + OCR
# ----------------------------
if st.session_state["results"] is not None and st.session_state["image"] is not None:
    if st.button("✂ Crop & Read Text"):
        result = st.session_state["results"]
        image = st.session_state["image"]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            st.warning("No detections found.")
            st.session_state["active_ingredients"] = []
        else:
            st.success(f"Detected {len(boxes)} region(s)")
            active_ingredients = []

            w, h = image.size

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # clamp
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image.crop((x1, y1, x2, y2))

                if show_crops:
                    st.image(
                        crop,
                        caption=f"Cropped #{i+1} | Confidence: {conf:.2f}",
                        use_container_width=True,
                    )

                with st.spinner("Reading text (OCR)..."):
                    extracted_text = run_ocr_on_image(crop)

                st.text_area(f"OCR Result #{i+1}", extracted_text, height=120)

                if extracted_text.strip():
                    active_ingredients.append(extracted_text.strip())

            # Remove duplicates while keeping order
            seen = set()
            active_ingredients_unique = []
            for t in active_ingredients:
                key = t.lower()
                if key not in seen:
                    seen.add(key)
                    active_ingredients_unique.append(t)

            st.session_state["active_ingredients"] = active_ingredients_unique

# ----------------------------
# ChatBot Section (Nice UI + Text/Voice)
# ----------------------------
if st.session_state["active_ingredients"]:
    st.divider()
    st.subheader("💬 ChatBot About Active Ingredients")

    ingredient = st.selectbox(
        "Select Active Ingredient (from OCR)",
        st.session_state["active_ingredients"],
    )

    # Show chat history (filtered by selected ingredient is optional; هنا بنعرض كله)
    for msg in st.session_state["chat_history"]:
        render_chat_bubble(msg["role"], msg["content"])

    st.markdown("#### ✍️/🎤 Ask (Text or Voice)")

    mode = st.segmented_control(
        "Input mode",
        options=["Text", "Voice"],
        default="Text",
    )

    user_question = ""

    if mode == "Text":
        user_question = st.chat_input("اكتب سؤالك هنا…")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            audio = st.audio_input("سجّل سؤالك (واضح وقصير)")
            if audio is not None:
                st.audio(audio)
        with c2:
            st.write("")
            st.write("")
            transcribe_btn = st.button("🎙️ تحويل", use_container_width=True)

        if transcribe_btn:
            if audio is None:
                st.warning("سجّل الصوت الأول.")
            else:
                with st.spinner("بيحوّل الصوت لتكست..."):
                    transcript = transcribe_audio_faster_whisper(audio)
                st.session_state["last_transcript"] = transcript
                if transcript:
                    st.success("اتحوّل ✅")
                else:
                    st.warning("محصلش نص واضح. جرّبي تاني بصوت أوضح.")

        # عرض الترانسكريبت + زر إرسال
        if st.session_state["last_transcript"]:
            st.text_area(
                "📄 النص بعد التحويل:", st.session_state["last_transcript"], height=110
            )
            if st.button("📨 ابعت السؤال", type="primary"):
                user_question = st.session_state["last_transcript"]

    # If we have a question, answer
    if user_question:
        # Add user msg
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_question, "ingredient": ingredient}
        )
        render_chat_bubble("user", user_question)

        with st.spinner("Bot is thinking..."):
            answer = ask_about_drug_openrouter(ingredient, user_question)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer, "ingredient": ingredient}
        )
        render_chat_bubble("assistant", answer)

    st.warning("⚠️ This AI does not replace professional medical advice.")
else:
    st.info("ارفع صورة → Run Detection → Crop & Read Text علشان يظهر الشات.")
