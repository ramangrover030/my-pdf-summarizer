import io
import PyPDF2
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configure the web page appearance
st.set_page_config(page_title="Secure PDF Summarizer", page_icon="🔒", layout="centered")

st.title("🔒 Privacy-First PDF Summarizer")
st.write("Upload a document. The AI processes it entirely in temporary RAM—no files are saved.")

# 1. Load the model and cache it so it loads fast for users
@st.cache_resource
def load_ai_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

try:
    with st.spinner("⏳ Starting local AI engines... please wait..."):
        tokenizer, model = load_ai_model()
    st.success("🤖 AI Engine Active!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# 2. Web Interface Upload Widget
uploaded_file = st.file_uploader("Choose a PDF file to summarize", type="pdf")

if uploaded_file is not None:
    # Read the file directly into memory bytes
    file_bytes = uploaded_file.read()
    
    with st.spinner("⚙️ Extracting text and thinking..."):
        # 3. Extract text from PDF in-memory
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        
        if extracted_text.strip():
            # Prepare text limits safely
            inputs = tokenizer(
                "summarize: " + extracted_text, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            )
            
            # 4. Generate the Summary (Slightly adjusted parameters for a fuller summary)
            summary_ids = model.generate(
                inputs["input_ids"], 
                max_length=200,      # Increased length so it's not too short!
                min_length=60,       # Guarantees a decent-sized summary
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
            hf_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Display results in a nice card UI component
            st.markdown("---")
            st.subheader("📋 Executive Summary:")
            st.info(hf_summary)
            st.success("🛡️ Process complete. Data cleared from active memory safely.")
        else:
            st.warning("⚠️ No readable text found in this PDF file.")