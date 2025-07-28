import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import pyperclip

st.set_page_config(page_title="🌐 Language Translator", layout="centered", initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center; color: cyan;'>🌍 Multilingual Translator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by Hugging Face Transformers | Translate across languages instantly</p>", unsafe_allow_html=True)

# Supported languages
lang_pairs = {
    "English to French": ("en", "fr"),
    "English to German": ("en", "de"),
    "English to Hindi": ("en", "hi"),
    "English to Spanish": ("en", "es"),
    "English to Italian": ("en", "it"),
    "English to Russian": ("en", "ru"),
    "French to English": ("fr", "en"),
    "German to English": ("de", "en"),
    "Hindi to English": ("hi", "en"),
    "Spanish to English": ("es", "en"),
    "Italian to English": ("it", "en"),
    "Russian to English": ("ru", "en"),
}

# Select translation pair
option = st.selectbox("Select Language Pair 🌐", list(lang_pairs.keys()))
src_lang, tgt_lang = lang_pairs[option]

# Input text
text = st.text_area("📝 Enter text to translate", "", height=150)

# Show character and word count
if text:
    st.info(f"🧾 Characters: {len(text)} | Words: {len(text.split())}")

# Translate button
if st.button("🚀 Translate"):
    if text.strip() == "":
        st.warning("⚠️ Please enter text before translating.")
    else:
        with st.spinner("🔄 Translating..."):
            try:
                model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

                tokens = tokenizer([text], return_tensors="pt", padding=True)
                translated = model.generate(**tokens)
                output = tokenizer.decode(translated[0], skip_special_tokens=True)

                st.success("✅ Translation Complete!")
                st.text_area("📌 Translated Text", output, height=150)

                if st.button("📋 Copy to Clipboard"):
                    pyperclip.copy(output)
                    st.success("✅ Text copied to clipboard!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
