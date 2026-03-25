import streamlit as st
import requests
import re
from PIL import Image, ImageEnhance, ImageFilter
import io

st.set_page_config(page_title="OCR (English) - OCR.space", layout="wide")
st.title("OCR (English) باستخدام OCR.space API")

# ⚠️ يفضل تخزين المفتاح في Streamlit secrets بدل ما يكون مكتوب هنا
api_key = ""

# ✅ إنجليزي فقط
language = "eng"

# خيارات إضافية
col1, col2 = st.columns(2)
with col1:
    use_preprocess = st.checkbox("تحسين الصورة قبل OCR (مُوصى به)", value=True)
with col2:
    is_table = st.checkbox("الصورة تحتوي جدول/نموذج", value=False)

uploaded_file = st.file_uploader(
    "اختر صورة (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    left, right = st.columns(2)

    # عرض الصورة الأصلية
    with left:
        st.subheader("Original")
        st.image(uploaded_file, caption="Uploaded image", width="stretch")

    # تجهيز الصورة للإرسال
    if use_preprocess:
        img = Image.open(uploaded_file)

        # 1) Grayscale
        img = img.convert("L")

        # 2) Upscale x2 (يساعد جدًا للنص الصغير)
        w, h = img.size
        img = img.resize((w * 2, h * 2))

        # 3) Contrast
        img = ImageEnhance.Contrast(img).enhance(2.0)

        # 4) Sharpen
        img = img.filter(ImageFilter.SHARPEN)

        # Save to memory as PNG
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        with right:
            st.subheader("Preprocessed")
            st.image(img, caption="After preprocessing", width="stretch")

        files = {"filename": ("preprocessed.png", buffer, "image/png")}
    else:
        files = {"filename": (uploaded_file.name, uploaded_file, uploaded_file.type)}

    url = "https://api.ocr.space/parse/image"
    payload = {
        "apikey": api_key,
        "language": language,  # ✅ eng only
        "isOverlayRequired": False,
        "OCREngine": 2,
        "scale": True,
        "detectOrientation": True,
        "isTable": is_table,
    }

    st.divider()

    if st.button("Extract Text", type="primary"):
        with st.spinner("Extracting text..."):
            try:
                response = requests.post(url, data=payload, files=files, timeout=60)
                result = response.json()
            except Exception as e:
                st.error(f"API connection error: {e}")
                st.stop()

        if result.get("IsErroredOnProcessing"):
            st.error("OCR Error: " + str(result.get("ErrorMessage")))
            st.json(result)
        else:
            # ✅ جمع كل النتائج (مش أول عنصر بس)
            parsed_text = "\n\n".join(
                [r.get("ParsedText", "") for r in result.get("ParsedResults", [])]
            ).strip()

            if not parsed_text:
                st.warning(
                    "No text extracted. Try enabling preprocessing or using a clearer image."
                )
                st.stop()

            # فلتر بسيط لقياس وضوح النص (اختياري)
            text_only = re.sub(r"[^A-Za-z0-9\s]", "", parsed_text)
            ratio = len(text_only) / max(len(parsed_text), 1)

            if len(parsed_text) < 5 or ratio < 0.5:
                st.warning(
                    "The extracted text looks unclear 🤔 (try a clearer image / enable preprocessing)."
                )
            else:
                st.success("Done!")

            st.text_area("Extracted text", parsed_text, height=350)

            st.download_button(
                label="Download TXT",
                data=parsed_text,
                file_name="extracted_text.txt",
                mime="text/plain",
            )
