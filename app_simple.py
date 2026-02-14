"""
Simple RAG Application - Streamlined Version
A straightforward document Q&A application.
"""

import streamlit as st
import os
from pathlib import Path

# Try to import optional dependencies
try:
    import PyPDF2
    from docx import Document
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page config
st.set_page_config(
    page_title="Simple RAG App",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Simple Document Q&A")

if not DEPENDENCIES_AVAILABLE:
    st.error(f"Missing dependencies: {IMPORT_ERROR}")
    st.info("Please run: pip install -r requirements.txt")
    st.stop()

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []
if 'model' not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'docx', 'csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Files"):
        for file in uploaded_files:
            try:
                # Extract text based on file type
                text = ""
                if file.name.endswith('.txt'):
                    text = file.read().decode('utf-8')
                elif file.name.endswith('.pdf'):
                    pdf = PyPDF2.PdfReader(file)
                    text = "\n".join([page.extract_text() for page in pdf.pages])
                elif file.name.endswith('.docx'):
                    doc = Document(file)
                    text = "\n".join([para.text for para in doc.paragraphs])
                elif file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                    text = df.to_string()
                
                if text:
                    # Create embeddings
                    embedding = st.session_state.model.encode([text])[0]
                    st.session_state.documents.append({
                        'name': file.name,
                        'text': text,
                        'embedding': embedding
                    })
                    
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        if st.session_state.documents:
            st.success(f"Processed {len(st.session_state.documents)} documents!")
    
    if st.session_state.documents:
        st.write(f"**{len(st.session_state.documents)} documents loaded**")
        if st.button("Clear All"):
            st.session_state.documents = []
            st.rerun()

# Main area - Q&A
st.header("Ask Questions")

if not st.session_state.documents:
    st.info("👈 Upload some documents to get started!")
else:
    query = st.text_input("Enter your question:")
    
    if query:
        # Search for relevant documents
        query_embedding = st.session_state.model.encode([query])[0]
        
        similarities = []
        for doc in st.session_state.documents:
            sim = cosine_similarity([query_embedding], [doc['embedding']])[0][0]
            similarities.append((doc, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Show results
        st.subheader("Relevant Information:")
        
        for doc, score in similarities[:3]:
            if score > 0.3:
                with st.expander(f"📄 {doc['name']} (Relevance: {score:.2%})"):
                    # Show first 500 characters
                    preview = doc['text'][:500]
                    if len(doc['text']) > 500:
                        preview += "..."
                    st.write(preview)
        
        # Check for OpenAI API
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
            if api_key and api_key != "your-openai-api-key-here":
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                
                # Get context from top document
                context = similarities[0][0]['text'][:2000] if similarities else ""
                
                with st.spinner("Generating answer..."):
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Answer based on the provided context."},
                            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                        ],
                        max_tokens=500
                    )
                    
                    st.subheader("AI Answer:")
                    st.write(response.choices[0].message.content)
        except Exception as e:
            st.info("💡 Add OpenAI API key to `.streamlit/secrets.toml` for AI-powered answers")

st.markdown("---")
st.caption("Simple RAG Application")
