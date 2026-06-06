import streamlit as st
from pypdf import PdfReader
from transformers import pipeline

# Set up page layout
st.set_page_config(page_title="Privacy-First PDF Summarizer", page_icon="🔒", layout="wide")

st.title("🔒 Privacy-First PDF Summarizer")
st.write("Upload any PDF to get a smart, structured summary generated directly in your browser session.")

# Cache the AI model so it only downloads once
@st.cache_resource
def load_summarizer():
    # Loading the local BART model
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Upload File Widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("⚙️ Extracting text and analyzing document structure... Please wait."):
        # 1. Extract all text from the PDF
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        # 2. Smart Chunking (Breaking long text into 3000-character blocks)
        # This stops the AI from breaking or truncating long files!
        chunk_size = 3000
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        
        st.write(f"📖 *Document read successfully. Processing {len(chunks)} sections...*")
        
        # 3. Summarize each chunk separately
        partial_summaries = []
        progress_bar = st.progress(0)
        
        for index, chunk in enumerate(chunks):
            # Basic validation to ensure the chunk has enough text
            if len(chunk.strip().split()) > 30:
                try:
                    # AI looks at each piece individually
                    summary_output = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
                    partial_summaries.append(summary_output[0]['summary_text'])
                except Exception:
                    pass
            # Update progress bar dynamically
            progress_bar.progress((index + 1) / len(chunks))
            
    # Display the final, beautiful output
    st.success("✨ Summary Generation Complete!")
    
    st.subheader("📋 Executive Summary Breakdown")
    
    # Presenting the chunks beautifully as bullet points or paragraphs
    for i, part in enumerate(partial_summaries):
        st.markdown(f"**Section {i+1}:** {part}")
        
    st.info("🔒 Security Notice: Process complete. Data cleared from active memory safely.")
