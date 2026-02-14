import streamlit as st
import requests
import os
from openai import OpenAI
import time
from typing import List, Dict, Tuple
import base64
import hashlib
import io
import tempfile
from pathlib import Path
import zipfile

# File processing imports
import PyPDF2
import docx
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json


# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="CodeSage RAG", page_icon="🤖", layout="wide")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["groq_api_key"]
)

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for embeddings"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# Code file extensions to process
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', 
    '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.sql', '.sh', 
    '.yaml', '.yml', '.json', '.md', '.txt', '.dockerfile', '.gitignore',
    '.html', '.css', '.scss', '.sass', '.vue', '.svelte', '.r', '.m', '.scala'
}

# Document Processing Functions
class DocumentProcessor:
    def __init__(self):
        self.embedding_model = load_embedding_model()
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    def extract_text_from_txt(self, file) -> str:
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return ""
    
    def extract_text_from_code_file(self, file_content: bytes, filename: str) -> str:
        """Extract text from code files"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    content = file_content.decode(encoding)
                    # Add filename as header for better context
                    return f"# File: {filename}\n{content}"
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with error handling
            content = file_content.decode('utf-8', errors='replace')
            return f"# File: {filename}\n{content}"
            
        except Exception as e:
            st.error(f"Error reading code file {filename}: {e}")
            return f"# File: {filename}\n# Error reading file: {e}"
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            if chunk.strip():
                chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        if not self.embedding_model:
            return np.array([])
        
        try:
            embeddings = self.embedding_model.encode(chunks)
            return embeddings
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")
            return np.array([])

# Project Folder Processor
class ProjectFolderProcessor:
    def __init__(self, doc_processor: DocumentProcessor):
        self.doc_processor = doc_processor
    
    def extract_files_from_zip(self, zip_file) -> List[Dict]:
        """Extract and process files from uploaded zip"""
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                for file_path in file_list:
                    # Skip directories and hidden files
                    if file_path.endswith('/') or '/.git/' in file_path or '/__pycache__/' in file_path:
                        continue
                    
                    # Skip unwanted directories
                    skip_dirs = ['node_modules', 'dist', 'build', '.git', '__pycache__', 
                                '.venv', 'venv', '.env', 'logs', 'tmp', 'temp']
                    
                    if any(skip_dir in file_path for skip_dir in skip_dirs):
                        continue
                    
                    # Check file extension
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext in CODE_EXTENSIONS:
                        try:
                            file_content = zip_ref.read(file_path)
                            
                            # Skip very large files (> 1MB)
                            if len(file_content) > 1024 * 1024:
                                continue
                            
                            # Extract text content
                            text_content = self.doc_processor.extract_text_from_code_file(
                                file_content, file_path
                            )
                            
                            if text_content.strip():
                                extracted_files.append({
                                    'path': file_path,
                                    'content': text_content,
                                    'size': len(file_content),
                                    'extension': file_ext
                                })
                                
                        except Exception as e:
                            st.warning(f"Could not process file {file_path}: {e}")
                            continue
            
            return extracted_files
            
        except Exception as e:
            st.error(f"Error extracting zip file: {e}")
            return []
    
    def get_project_structure(self, files: List[Dict]) -> str:
        """Generate a project structure overview"""
        structure = {}
        
        for file_info in files:
            path_parts = file_info['path'].split('/')
            current_level = structure
            
            for part in path_parts[:-1]:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
            
            # Add file
            filename = path_parts[-1]
            current_level[filename] = f"({file_info['extension']})"
        
        def build_tree(structure, prefix=""):
            lines = []
            items = sorted(structure.items())
            
            for i, (name, content) in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                lines.append(f"{prefix}{current_prefix}{name}")
                
                if isinstance(content, dict) and content:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    lines.extend(build_tree(content, next_prefix))
            
            return lines
        
        tree_lines = build_tree(structure)
        return "\n".join(tree_lines)

# RAG Pipeline Class (enhanced)
class RAGPipeline:
    def __init__(self, doc_processor: DocumentProcessor):
        self.doc_processor = doc_processor
        self.project_processor = ProjectFolderProcessor(doc_processor)
        self.chunks = []
        self.embeddings = None
        self.metadata = []
    
    def add_document(self, content: str, filename: str, doc_type: str, file_path: str = None):
        """Add a document to the RAG pipeline"""
        chunks = self.doc_processor.chunk_text(content)
        
        if not chunks:
            return
        
        # Create embeddings for new chunks
        new_embeddings = self.doc_processor.create_embeddings(chunks)
        
        if new_embeddings.size == 0:
            return
        
        # Add chunks and metadata
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.metadata.append({
                'filename': filename,
                'file_path': file_path or filename,
                'doc_type': doc_type,
                'chunk_id': len(self.chunks) - 1
            })
        
        # Update embeddings
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def add_project_folder(self, zip_file, project_name: str) -> Dict:
        """Add an entire project folder to the pipeline"""
        files = self.project_processor.extract_files_from_zip(zip_file)
        
        if not files:
            return {'success': False, 'message': 'No valid code files found in the uploaded folder'}
        
        processed_files = 0
        total_lines = 0
        file_types = set()
        
        for file_info in files:
            self.add_document(
                content=file_info['content'],
                filename=Path(file_info['path']).name,
                doc_type='CODE',
                file_path=file_info['path']
            )
            
            processed_files += 1
            total_lines += file_info['content'].count('\n')
            file_types.add(file_info['extension'])
        
        # Generate project structure
        project_structure = self.project_processor.get_project_structure(files)
        
        return {
            'success': True,
            'project_name': project_name,
            'processed_files': processed_files,
            'total_lines': total_lines,
            'file_types': sorted(list(file_types)),
            'project_structure': project_structure,
            'files': files
        }
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Retrieve most relevant chunks for a query"""
        if not self.chunks or self.embeddings is None:
            return []
        
        if not self.doc_processor.embedding_model:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.doc_processor.embedding_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append((
                        self.chunks[idx], 
                        self.metadata[idx], 
                        similarities[idx]
                    ))
            
            return results
        except Exception as e:
            st.error(f"Error in retrieval: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        doc_types = {}
        for meta in self.metadata:
            doc_type = meta['doc_type']
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'unique_documents': len(set(meta['filename'] for meta in self.metadata)),
            'document_types': list(set(meta['doc_type'] for meta in self.metadata)),
            'doc_type_counts': doc_types,
            'code_files': len([m for m in self.metadata if m['doc_type'] == 'CODE']),
            'document_files': len([m for m in self.metadata if m['doc_type'] != 'CODE'])
        }
    
    def get_project_files_summary(self) -> str:
        """Get a summary of uploaded project files"""
        code_files = [m for m in self.metadata if m['doc_type'] == 'CODE']
        
        if not code_files:
            return ""
        
        # Group by file extension
        ext_counts = {}
        file_paths = set()
        
        for meta in code_files:
            file_path = meta.get('file_path', meta['filename'])
            file_paths.add(file_path)
            
            ext = Path(file_path).suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
        summary = "**Uploaded Project Files:**\n"
        summary += f"- **Total files:** {len(file_paths)}\n"
        summary += f"- **File types:** {', '.join(sorted(ext_counts.keys()))}\n"
        
        # Show file type distribution
        for ext, count in sorted(ext_counts.items()):
            summary += f"  - {ext}: {count} files\n"
        
        return summary

# Initialize RAG Pipeline
@st.cache_resource
def initialize_rag_pipeline():
    """Initialize the RAG pipeline"""
    doc_processor = DocumentProcessor()
    return RAGPipeline(doc_processor)

# GitHub functions (existing)
@st.cache_data(ttl=3600)
def get_github_files(repo_url: str, max_files: int = 50) -> List[Dict]:
    """Fetch files from a GitHub repository using GitHub API"""
    try:
        parts = repo_url.replace("https://github.com/", "").split("/")
        if len(parts) < 2:
            return []
        
        owner, repo = parts[0], parts[1]
        
        def get_repo_contents(path="", current_count=0):
            if current_count >= max_files:
                return [], current_count
                
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            response = requests.get(api_url)
            
            if response.status_code != 200:
                return [], current_count
            
            contents = response.json()
            files = []
            
            for item in contents:
                if current_count >= max_files:
                    break
                    
                if item["type"] == "file":
                    if any(item["name"].endswith(ext) for ext in [
                        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", 
                        ".c", ".h", ".cs", ".php", ".rb", ".go", ".rs", ".swift", 
                        ".kt", ".scala", ".sql", ".sh", ".yaml", ".yml", 
                        ".json", ".md", ".txt", ".dockerfile", ".gitignore"
                    ]):
                        files.append(item)
                        current_count += 1
                        
                elif item["type"] == "dir" and not item["name"].startswith(".") and item["name"] not in ["node_modules", "__pycache__", "dist", "build"]:
                    sub_files, current_count = get_repo_contents(item["path"], current_count)
                    files.extend(sub_files)
            
            return files, current_count
        
        files, _ = get_repo_contents()
        return files
        
    except Exception as e:
        st.error(f"Error fetching repository: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_file_content(download_url: str, max_size: int = 50000) -> str:
    """Get the actual content of a file from GitHub"""
    try:
        response = requests.get(download_url)
        if response.status_code == 200:
            content = response.text
            if len(content) > max_size:
                content = content[:max_size] + "\n\n[... file truncated due to size ...]"
            return content
    except:
        pass
    return ""

def count_tokens_roughly(text: str) -> int:
    """Rough token count estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

def enhanced_analyze_with_rag(repo_url: str, question: str, rag_pipeline: RAGPipeline) -> str:
    """Enhanced analysis using both GitHub repo and RAG pipeline"""
    try:
        context_parts = []
        
        # 1. Get relevant chunks from RAG pipeline
        rag_results = rag_pipeline.retrieve_relevant_chunks(question, top_k=5)
        
        if rag_results:
            context_parts.append("=== RELEVANT CONTEXT FROM UPLOADED CONTENT ===")
            for i, (chunk, metadata, score) in enumerate(rag_results):
                if metadata['doc_type'] == 'CODE':
                    context_parts.append(f"\n--- CODE FILE: {metadata['file_path']} (Relevance: {score:.3f}) ---")
                else:
                    context_parts.append(f"\n--- DOCUMENT: {metadata['filename']} (Relevance: {score:.3f}) ---")
                context_parts.append(chunk)
                context_parts.append("--- END CHUNK ---")
        
        # 2. Get GitHub repository context (existing logic)
        if repo_url:
            with st.spinner("🔍 Fetching repository files..."):
                files = get_github_files(repo_url, max_files=20)
            
            if files:
                st.success(f"✅ Found {len(files)} code files in GitHub repo")
                
                # Smart file selection
                priority_files = []
                question_lower = question.lower()
                question_keywords = ["auth", "database", "api", "config", "main", "index", "app", "server", "model", "route", "component"]
                
                for file in files:
                    file_name = file["name"].lower()
                    file_path = file["path"].lower()
                    
                    if any(keyword in question_lower and keyword in file_path for keyword in question_keywords):
                        priority_files.append(file)
                    elif any(key in file_name for key in ["main", "index", "app", "server", "readme", "__init__"]):
                        priority_files.append(file)
                
                # Add repository context
                max_repo_tokens = 2000 if rag_results else 3000
                current_tokens = count_tokens_roughly("\n".join(context_parts))
                
                if current_tokens < max_repo_tokens:
                    context_parts.append("\n=== GITHUB REPOSITORY CONTEXT ===")
                    context_parts.append(f"REPOSITORY: {repo_url}")
                    
                    selected_files = []
                    for file in priority_files[:5]:
                        content = get_file_content(file["download_url"], max_size=5000)
                        if content:
                            file_tokens = count_tokens_roughly(content)
                            if current_tokens + file_tokens < max_repo_tokens:
                                selected_files.append((file, content))
                                current_tokens += file_tokens
                            else:
                                break
                    
                    for file, content in selected_files:
                        context_parts.append(f"\n--- GITHUB FILE: {file['path']} ---")
                        context_parts.append(content)
                        context_parts.append("--- END GITHUB FILE ---")
        
        # 3. Create comprehensive context
        full_context = "\n".join(context_parts)
        
        if not context_parts:
            return "❌ **No context available.** Please upload documents/project folder or provide a GitHub repository URL."
        
        # 4. Generate response
        system_prompt = """You are a Senior Software Engineer and Technical Analyst. 
        
You have access to:
1. Document content from uploaded files (marked as DOCUMENT)
2. Code files from uploaded project folders (marked as CODE FILE)
3. Code repository content from GitHub (marked as GITHUB FILE/REPOSITORY CONTEXT)

Analyze the provided context and answer the user's question comprehensively. 
- If the question relates to uploaded documents, prioritize that information
- If the question relates to code (either uploaded or from GitHub), focus on the relevant code context
- Combine insights from multiple sources when relevant
- Be specific about what you see in the provided context
- When referencing code, mention the specific file name and path"""
        
        user_prompt = f"""CONTEXT:
{full_context}

USER QUESTION: {question}

Provide a detailed analysis based on the available context. Be specific about which files you're referencing and clearly distinguish between uploaded project files, documents, and GitHub repository files."""
        
        with st.spinner("🤖 Generating comprehensive analysis..."):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.1
                )
                
                result = response.choices[0].message.content
                
                # Add analysis summary
                rag_stats = rag_pipeline.get_stats()
                summary_parts = [f"\n---\n**Enhanced Analysis Summary:**"]
                
                if rag_results:
                    code_chunks = len([r for r in rag_results if r[1]['doc_type'] == 'CODE'])
                    doc_chunks = len([r for r in rag_results if r[1]['doc_type'] != 'CODE'])
                    
                    if code_chunks > 0:
                        unique_code_files = len(set(r[1]['file_path'] for r in rag_results if r[1]['doc_type'] == 'CODE'))
                        summary_parts.append(f"- Uploaded code files analyzed: {unique_code_files}")
                        summary_parts.append(f"- Code chunks used: {code_chunks}")
                    
                    if doc_chunks > 0:
                        unique_docs = len(set(r[1]['filename'] for r in rag_results if r[1]['doc_type'] != 'CODE'))
                        summary_parts.append(f"- Documents analyzed: {unique_docs}")
                        summary_parts.append(f"- Document chunks used: {doc_chunks}")
                
                if repo_url:
                    summary_parts.append(f"- GitHub repository: {repo_url}")
                    summary_parts.append(f"- GitHub files analyzed: {len(selected_files) if 'selected_files' in locals() else 0}")
                
                summary_parts.append(f"- Total content in pipeline: {rag_stats['total_chunks']} chunks")
                if rag_stats['code_files'] > 0:
                    summary_parts.append(f"- Total uploaded code files: {rag_stats['code_files']}")
                if rag_stats['document_files'] > 0:
                    summary_parts.append(f"- Total documents: {rag_stats['document_files']}")
                
                summary = "\n".join(summary_parts)
                return result + summary
                
            except Exception as api_error:
                error_str = str(api_error)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    return """💔😔 **Rate limit reached!** Please wait and try again."""
                else:
                    return f"❌ **API Error:** {error_str}"
        
    except Exception as e:
        return f"❌ **Error in enhanced analysis:** {str(e)}"

def response_generator(text: str):
    """Stream response for better UX"""
    for line in text.split("\n"):
        yield line + "\n"
        time.sleep(0.02)

# Initialize RAG Pipeline
rag_pipeline = initialize_rag_pipeline()

# Main UI
st.title("🤖 CodeSage RAG - Enhanced Code & Document Analyzer")
st.markdown("🚀 **Upload documents + project folders + Analyze GitHub repos** - Get comprehensive AI-powered insights!")

# Sidebar
with st.sidebar:
    st.title("🤖 CodeSage RAG")
    st.markdown("---")
    
    # Document Upload Section
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload documents to enhance analysis with relevant context"
    )
    
    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing uploaded documents..."):
            doc_processor = rag_pipeline.doc_processor
            
            for uploaded_file in uploaded_files:
                # Check if file already processed (simple cache)
                if not any(meta['filename'] == uploaded_file.name and meta['doc_type'] != 'CODE' 
                          for meta in rag_pipeline.metadata):
                    if uploaded_file.type == "application/pdf":
                        content = doc_processor.extract_text_from_pdf(uploaded_file)
                        doc_type = "PDF"
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        content = doc_processor.extract_text_from_docx(uploaded_file)
                        doc_type = "DOCX"
                    elif uploaded_file.type == "text/plain":
                        content = doc_processor.extract_text_from_txt(uploaded_file)
                        doc_type = "TXT"
                    else:
                        continue
                    
                    if content.strip():
                        rag_pipeline.add_document(content, uploaded_file.name, doc_type)
    
    st.markdown("---")
    
    # NEW: Project Folder Upload Section
    st.subheader("📁 Upload Project Folder")
    st.info("💡 **Tip:** Zip your project folder and upload it here")
    
    project_zip = st.file_uploader(
        "Upload project as ZIP file",
        type=['zip'],
        help="Upload a ZIP file containing your project source code"
    )
    
    if project_zip:
        project_name = st.text_input(
            "Project Name (optional)", 
            value=project_zip.name.replace('.zip', ''),
            help="Give your project a name for easier reference"
        )
        
        if st.button("🔄 Process Project Folder", type="primary"):
            with st.spinner("🔍 Processing project folder..."):
                result = rag_pipeline.add_project_folder(project_zip, project_name)
                
                if result['success']:
                    st.success(f"✅ **Project processed successfully!**")
                    st.info(f"""
                    **Project:** {result['project_name']}
                    - **Files processed:** {result['processed_files']}
                    - **Total lines:** {result['total_lines']:,}
                    - **File types:** {', '.join(result['file_types'])}
                    """)
                    
                    # Show project structure in expander
                    with st.expander("📂 Project Structure", expanded=False):
                        st.code(result['project_structure'], language='text')
                else:
                    st.error(f"❌ {result['message']}")
    
    st.markdown("---")
    
    # Repository Input Section
    st.subheader("📂 GitHub Repository")
    selected_repo = st.text_input(
        "Enter GitHub repository URL:",
        placeholder="https://github.com/owner/repo",
        help="Optional: Add a GitHub repo for additional code analysis"
    )
    
    if selected_repo and not selected_repo.startswith("https://github.com/"):
        st.error("❌ Please enter a valid GitHub URL")
        selected_repo = ""
    
    st.markdown("---")
    
    # Show current pipeline stats
    stats = rag_pipeline.get_stats()
    if stats['total_chunks'] > 0:
        st.subheader("📊 Current Content")
        st.success(f"""
        **Total Content Loaded:**
        - **Documents:** {stats['document_files']} files
        - **Code files:** {stats['code_files']} files  
        - **Text chunks:** {stats['total_chunks']}
        - **Unique files:** {stats['unique_documents']}
        """)
        
        # Show project files summary
        project_summary = rag_pipeline.get_project_files_summary()
        if project_summary:
            with st.expander("📁 Project Files Details", expanded=False):
                st.markdown(project_summary)
    
    st.markdown("---")
    
    # How it works
    st.subheader("💡 How Enhanced RAG Works")
    st.info("""
    **1. Document Processing:**
    - 📄 Upload PDF/DOCX/TXT files
    - 📁 Upload entire project folders (ZIP)
    - 🔗 Connect GitHub repositories
    
    **2. Smart Analysis:**
    - 🔤 Extract and chunk all content  
    - 🧮 Create semantic embeddings
    - 🗃️ Store in unified vector database
    
    **3. Intelligent Retrieval:**
    - 🔍 Query finds most relevant content
    - 🎯 Combines docs + uploaded code + GitHub
    - 🤖 AI generates comprehensive answers
    """)
    
    st.markdown("---")
    
    # Enhanced example questions
    st.subheader("💭 Enhanced Question Examples")
    st.markdown("""
    **Project Analysis:**
    - Analyze the architecture of my uploaded project
    - What security vulnerabilities exist in my code?
    - Generate documentation for the main components
    - How can I improve the code quality?
    
    **Cross-Reference Analysis:**
    - How does my code compare to the GitHub repo?
    - Do the uploaded docs match the implementation?
    - What features are documented but not implemented?
    
    **Code Understanding:**
    - Explain the main workflow in my project
    - What external dependencies are used?
    - How is error handling implemented?
    - What are the main API endpoints?
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_repo" not in st.session_state:
    st.session_state.current_repo = None

# Clear messages if repo changed
if selected_repo != st.session_state.current_repo:
    st.session_state.current_repo = selected_repo
    # Don't clear messages when repo changes - keep document context

# Main chat interface
stats = rag_pipeline.get_stats()
has_documents = stats['document_files'] > 0
has_code_files = stats['code_files'] > 0
has_repo = bool(selected_repo)

if not has_documents and not has_code_files and not has_repo:
    st.info("👉 **Please upload content in the sidebar to start!**")
    
    # Enhanced feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📄 **Documents**
        - Upload PDF, DOCX, TXT files
        - Extract and analyze text content
        - Perfect for specifications, requirements, documentation
        
        **Use cases:**
        - Technical documentation analysis
        - Requirements extraction
        - Meeting notes processing
        """)
    
    with col2:
        st.markdown("""
        ### 📁 **Project Folders** 
        - Upload entire codebases as ZIP
        - Supports 25+ programming languages
        - Automatic file structure analysis
        
        **Use cases:**
        - Complete project analysis
        - Code review and auditing
        - Architecture understanding
        - Bug detection and optimization
        """)
    
    with col3:
        st.markdown("""
        ### 🔗 **GitHub Repos**
        - Connect any public GitHub repository
        - Real-time code fetching
        - Smart file prioritization
        
        **Use cases:**
        - Compare implementations
        - Learn from open source
        - Analyze external dependencies
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### ✨ **Enhanced RAG Capabilities:**
    
    - **🧠 Semantic Search**: AI understands context and meaning, not just keywords
    - **📊 Multi-Source Analysis**: Combines documents, uploaded code, and GitHub repos
    - **🎯 Smart Chunking**: Optimal text segmentation for better retrieval  
    - **⚡ Real-time Processing**: No pre-setup required, instant analysis
    - **🔍 Cross-Reference**: Compare documentation with actual implementation
    - **📈 Project Insights**: Architecture analysis, code quality assessment
    
    ### 🎯 **Perfect for:**
    - **Developers**: Understand large codebases quickly
    - **Code Reviewers**: Comprehensive analysis with documentation context
    - **Technical Writers**: Verify documentation accuracy against code
    - **Students**: Learn from real-world projects with guided analysis
    - **Teams**: Onboard new members with AI-powered code explanation
    """)
    
    st.stop()

# Display current context info
context_info = []
if has_documents:
    context_info.append(f"📄 {stats['document_files']} documents")
if has_code_files:
    context_info.append(f"📁 {stats['code_files']} code files")
if has_repo:
    context_info.append(f"🔗 {selected_repo.split('/')[-1]} repo")

if context_info:
    st.success(f"**Active Context:** {' + '.join(context_info)} ({stats['total_chunks']} chunks)")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
context_name = []
if has_documents:
    context_name.append("documents")
if has_code_files:
    context_name.append("uploaded code")
if has_repo:
    context_name.append(selected_repo.split('/')[-1])

prompt_text = f"Ask about {' + '.join(context_name) if context_name else 'your content'}..."

if prompt := st.chat_input(prompt_text):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate enhanced response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Enhanced analysis with RAG
        response = enhanced_analyze_with_rag(selected_repo, prompt, rag_pipeline)
        
        # Stream the response
        full_response = ""
        for chunk in response_generator(response):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.8em;'>
    🤖 CodeSage RAG v2.0 - Enhanced with Project Folder Upload, Document Processing & Semantic Search<br>
    💡 Upload documents, project folders (ZIP), or connect GitHub repos for comprehensive AI-powered analysis
    </div>
    """, 
    unsafe_allow_html=True
)