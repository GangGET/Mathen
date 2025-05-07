import os
import pdfplumber
import re
import time
import pickle
import faiss
import numpy as np
from io import BytesIO
from fpdf import FPDF
from flask import Flask, render_template, request, jsonify, send_file, session
from flask import Flask, send_from_directory, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import ollama
import logging
from datetime import timedelta
import tempfile
import shutil
import PyPDF2
from paddleocr import PaddleOCR
import cv2
from PIL import Image

# Initialize Flask app with session configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-dev-key')
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem instead of cookie storage
Session(app)

# Initialize FAISS index and metadata
index_dir = "vector_store"
index_file = os.path.join(index_dir, "document_index.faiss")
metadata_file = os.path.join(index_dir, "document_metadata.pkl")

# Ensure vector store directory exists
if not os.path.exists(index_dir):
    os.makedirs(index_dir)

# Initialize or load FAISS index
if os.path.exists(index_file):
    faiss_index = faiss.read_index(index_file)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
else:
    # Initialize empty FAISS index with correct dimensions
    embedding_dim = 384  # MiniLM-L6-v2 dimension
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    metadata = []

# Initialize embedding model and OCR
embedder = None
ocr = None




def get_ocr():
    global ocr
    if ocr is None:
        #ocr_engine = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, enable_mkldnn=False)
        ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, enable_mkldnn=False)
    return ocr
def get_embedder():
    global embedder
    if embedder is None:
        import torch
        # Configure PyTorch for secure model loading
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Load model with secure weights_only setting
        with torch.no_grad():
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return embedder

EMBEDDING_DIM = 384  # Dimension size of the all-MiniLM-L6-v2 model

# Create necessary directories
upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = upload_folder

# Improved function to extract text from PDF with better error handling
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber, PyPDF2, and PaddleOCR for comprehensive coverage"""
    text = ""
    ocr_text = ""
    
    # Try pdfplumber first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                except Exception as e:
                    logger.warning(f"pdfplumber page error: {str(e)}")
    except Exception as e:
        logger.warning(f"pdfplumber error: {str(e)}")
        
    # If pdfplumber failed, try PyPDF2
    if not text.strip():
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"PyPDF2 page error: {str(e)}")
        except Exception as e:
            logger.warning(f"PyPDF2 error: {str(e)}")
    
    # If both text extraction methods failed or text is sparse, try OCR
    if not text.strip() or len(text.split()) < 50:
        try:
            # Get OCR engine with explicit CPU settings
            ocr_engine = get_ocr()
        
            # Convert PDF to images
            images = convert_pdf_to_images(pdf_path)
        
            # Process each image with PaddleOCR with better error handling
            for i, img in enumerate(images):
                try:
                    # Ensure proper image format - convert to RGB numpy array with explicit type
                    img_array = np.array(img, dtype=np.uint8)
                
                    # Handle grayscale conversion safely
                    if len(img_array.shape) == 2:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    elif img_array.shape[2] == 4:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                    # Add more debug info
                    logger.info(f"Processing image {i+1}, shape: {img_array.shape}, dtype: {img_array.dtype}")
                
                    # Run OCR with timeout to avoid hanging
                    result = ocr_engine.ocr(img_array, cls=True)
                
                    if result:
                        page_text = ""
                        for line in result:
                            for word_info in line:
                                if isinstance(word_info, (list, tuple)) and len(word_info) >= 2:
                                    text_info = word_info[1]
                                    if isinstance(text_info, (list, tuple)) and len(text_info) > 0:
                                        page_text += text_info[0] + " "
                        if page_text.strip():
                            ocr_text += page_text.strip() + "\n\n"
                            logger.info(f"Successfully extracted text from page {i + 1}")
                except Exception as e:
                    logger.warning(f"OCR processing error on page {i+1}: {str(e)}")
                    logger.warning(f"Image details: shape={getattr(img, 'size', 'unknown')}, mode={getattr(img, 'mode', 'unknown')}")
                    continue
            
            # If OCR found text, use it
            if ocr_text.strip():
                if text.strip():
                    # Combine both texts if regular extraction also found something
                    text = text + "\n\n" + ocr_text
                else:
                    text = ocr_text
        except Exception as e:
            logger.warning(f"OCR error: {str(e)}")
        
        
            
            # Process each image with PaddleOCR
            for i, img in enumerate(images):
                try:
                    # Convert PIL Image to numpy array in correct format
                    img_array = np.array(img)
                    
                    # Ensure the image is in RGB format
                    if len(img_array.shape) == 2:  # Convert grayscale to RGB
                        img_array = np.stack([img_array] * 3, axis=-1)
                    
                    result = ocr_engine.ocr(img_array, cls=True)
                    if result:
                        page_text = ""
                        for line in result:
                            for word_info in line:
                                if isinstance(word_info, (list, tuple)) and len(word_info) >= 2:
                                    text_info = word_info[1]
                                    if isinstance(text_info, (list, tuple)) and len(text_info) > 0:
                                        page_text += text_info[0] + " "
                        if page_text.strip():
                            ocr_text += page_text.strip() + "\n\n"
                            logger.info(f"Successfully extracted text from page {i + 1}")
                except Exception as e:
                    logger.warning(f"OCR processing error on page {i + 1}: {str(e)}")
                    continue
            
            # If OCR found text, use it
            if ocr_text.strip():
                if text.strip():
                    # Combine both texts if regular extraction also found something
                    text = text + "\n\n" + ocr_text
                else:
                    text = ocr_text
        except Exception as e:
            logger.warning(f"OCR error: {str(e)}")
    
    # Clean up the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = text.strip()
    
    return text

def convert_pdf_to_images(pdf_path):
    """Convert PDF pages to PIL Images for OCR processing using pdf2image"""
    images = []
    try:
        # Try using pdf2image (poppler-based)
        from pdf2image import convert_from_path
        import os
        
        # Setup poppler path for Windows
        poppler_path = None
        
        # Verify PDF exists and is readable
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if os.path.getsize(pdf_path) == 0:
            raise ValueError("PDF file is empty")
        
        # Convert PDF to list of PIL images
        logger.info(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(
            pdf_path,
            dpi=200,  # Lower DPI for faster processing
            fmt='ppm',
            thread_count=4,
            grayscale=False,
            size=(2000, None),  # Max width of 2000px, height auto
    
            use_pdftocairo=True,  # Try pdftocairo first
            timeout=30  # Add timeout
        )
        
        if not images:
            raise ValueError("No images were extracted from the PDF")
            
        logger.info(f"Successfully converted {len(images)} pages to images")
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        try:
            # Fallback to PyMuPDF
            import fitz
            doc = fitz.open(pdf_path)
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    zoom = 4  # higher zoom for better quality
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Resize if needed
                    if img.width > 2000:
                        ratio = 2000 / img.width
                        new_size = (2000, int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    images.append(img)
                    logger.info(f"Successfully converted page {page_num + 1} using PyMuPDF")
                except Exception as e:
                    logger.warning(f"Error converting page {page_num}: {str(e)}")
                    continue
            
            doc.close()
            
        except ImportError:
            logger.error("Both pdf2image and PyMuPDF failed. Please install one of these packages.")
        except Exception as e:
            logger.error(f"Error in PyMuPDF fallback: {str(e)}")
    
    return images

# Improved text cleaning function
def clean_text(text):
    # Store paragraph breaks
    paragraphs = text.split('\n\n')
    
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Remove Hindi text
        hindi_pattern = re.compile(r'[\u0900-\u097F]+')  # Hindi Unicode range
        paragraph = re.sub(hindi_pattern, '', paragraph)
        
        # Remove excessive whitespace within paragraph
        paragraph = re.sub(r'\s+', ' ', paragraph)
        
        # Remove page numbers (common formats)
        paragraph = re.sub(r'^\s*\d+\s*$', '', paragraph)
        
        # Remove headers/footers that might appear on every page
        paragraph = re.sub(r'^\d+(/\d+)*$', '', paragraph)
        
        # Only add non-empty paragraphs
        if paragraph.strip():
            cleaned_paragraphs.append(paragraph.strip())
    
    # Rejoin with double newlines to preserve paragraph structure
    return '\n\n'.join(cleaned_paragraphs)

# Completely improved text chunking function
def improved_semantic_text_chunking(text, max_chunk_size=2000, overlap=150):
    """
    Split text into semantically meaningful chunks with improved robustness
    for handling large documents and various text formats.
    """
    # First, ensure we have a string
    if not isinstance(text, str):
        text = str(text)
    
    # Store original length for logging
    original_length = len(text)
    
    # Adjust chunk size for small documents
    if original_length < 3000:
        max_chunk_size = min(max_chunk_size, original_length)
        overlap = min(overlap, max_chunk_size // 4)
    
    # Force paragraph breaks if there aren't enough
    paragraph_count = text.count('\n\n')
    if paragraph_count < max(10, original_length // 5000):
        # Add paragraph breaks at sentence endings
        text = re.sub(r'([.!?])\s+', r'\1\n\n', text)
        new_count = text.count('\n\n')
        logger.info("Added paragraph breaks. Now have {} breaks.".format(new_count))
    
    # Split by paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    logger.info(f"Processing {len(paragraphs)} paragraphs for chunking")
    
    # Add a safeguard to prevent infinite loops
    max_iterations = len(text) // 50  # reasonable upper bound
    iteration_count = 0
    
    for paragraph in paragraphs:
        iteration_count += 1
        if iteration_count > max_iterations:
            # Emergency break if something goes wrong
            logger.warning(f"Max iterations ({max_iterations}) reached. Breaking loop.")
            break
            
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # Force chunk creation if current chunk is getting too large
        if len(current_chunk) >= max_chunk_size * 0.9:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        
        # If paragraph is too long, split it
        if len(paragraph) > max_chunk_size:
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                # Skip empty sentences
                if not sentence.strip():
                    continue
                    
                if len(current_chunk) + len(sentence) <= max_chunk_size:
                    current_chunk += sentence + " "
                else:
                    # Store the current chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap if possible
                    if len(current_chunk) > overlap:
                        words = current_chunk.split()
                        overlap_text = " ".join(words[-30:]) + " " if len(words) > 30 else current_chunk
                        current_chunk = overlap_text + sentence + " "
                    else:
                        current_chunk = sentence + " "
        else:
            # If adding this paragraph exceeds the max size, store current chunk and start new
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
            else:
                current_chunk += paragraph + "\n\n"
    
    # Add the last chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Safety check: If we still have no chunks but have text, force chunking by size
    if len(chunks) <= 1 and len(text) > max_chunk_size:
        logger.warning(f"Initial chunking failed, forcing chunking by size")
        chunks = []
        for i in range(0, len(text), max_chunk_size - overlap):
            chunk = text[i:i + max_chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
    
    logger.info("Created {} chunks from {} characters of text".format(len(chunks), original_length))
    
    # Final sanity check - never return just one chunk for a large document
    if len(chunks) <= 1 and len(text) > max_chunk_size * 2:
        # Force split into roughly equal chunks
        target_chunks = max(10, len(text) // max_chunk_size)
        chunk_size = len(text) // target_chunks
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        logger.info(f"Forced splitting into {len(chunks)} chunks")
    
    # Log chunk sizes for debugging
    if chunks:
        chunk_sizes = [len(chunk) for chunk in chunks]
        logger.info("Chunk sizes - Min: {}, Max: {}, Avg: {:.2f}".format(
            min(chunk_sizes), 
            max(chunk_sizes), 
            sum(chunk_sizes)/len(chunk_sizes)
        ))
    
    return chunks

# Improved summarization with better prompts and temperature settings
def summarize_text_with_headings(text, model="llama3.2:3b", max_retries=3, timeout=60):
    """
    Summarize text with heading structure using LLM
    """
    if not text.strip():
        return "No text to summarize."
    
    # For very short texts, don't chunk
    if len(text) < 3000:
        try: 
            logger.info("Text is short, generating direct summary without chunking")
            system_prompt = """
            You are an expert at summarizing documents. Create a clear, concise summary that:
            1. Ensure to include key dates such as 'bid opening date' and 'bid end date'
            2. Uses clear headings to organize information
            3. Highlights important facts and conclusions
            4. Captures the main points and key details.

            Keep the summary focused and avoid redundancy.
            Format your response in markdown with appropriate headings.
            """
             
            response = ollama.chat(model, messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": "Please summarize this text concisely:\n\n{}".format(text)
                }
            ], options={"temperature": 0.7, "top_k": 50})
            
            summary = response["message"]["content"] if "message" in response and "content" in response["message"] else "[Error: Unexpected response format]"
            return summary
            
        except Exception as e:
            logger.error("Error generating direct summary: {}".format(str(e)))
            return "Error generating summary: {}".format(str(e))
    
    # For longer texts, use chunking with progress tracking
    chunks = improved_semantic_text_chunking(text, max_chunk_size=4000)
    summary = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            logger.warning("Empty chunk at index {}".format(i))
            continue
            
        logger.info("Processing chunk {}/{} for summarization (length: {})".format(
            i+1, total_chunks, len(chunk)))
        
        retries = 0
        while retries < max_retries:
            try:
                system_prompt = """
                You are a helpful and knowledgeable assistant specializing in analyzing PDF documents. Your task is to provide detailed, accurate, and natural responses to questions about the document content.

Rules:
1. Base your answers ONLY on the provided context - never make up information
2. If the context doesn't contain enough information, honestly say so
3. Be conversational and engaging while maintaining professionalism
4. Format responses using markdown for better readability
5. If the user's query is a greeting or casual conversation, respond naturally while gently reminding them you're here to help with the PDF content
6. For complex queries, break down your response into organized sections
7. Always cite specific details from the context when available"""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context from PDF:\n{chunk}\n\nUser Query: Summarize this section concisely"}
                ]

                response = ollama.chat(model, messages=messages, options={
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.9
                })

                summary_chunk = response["message"]["content"]
                
                logger.info("Generated summary for chunk {}/{} ({} chars)".format(
                    i+1, total_chunks, len(summary_chunk)))
                
                summary.append("### Section {}\n{}\n\n".format(i+1, summary_chunk))
                break
            except Exception as e:
                retries += 1
                logger.error("Error on attempt {}/{} for chunk {}: {}".format(
                    retries, max_retries, i+1, str(e)))
                if retries >= max_retries:
                    summary.append("### Section {}\n[Error: Failed to summarize this section after {} attempts]\n\n".format(
                        i+1, max_retries))
                time.sleep(1)  # Brief pause before retry
    
    # Add overall summary if there are multiple chunks
    if len(chunks) > 1:
        try:
            logger.info("Generating overall summary...")
            section_summaries = "\n".join(summary)
            
            overall_prompt = f"""
            Based on the following section summaries, provide a brief overall summary that captures the main themes and key points:
            
            {section_summaries}
            
            Create a concise "Executive Summary" of the entire document (3-5 bullets).
            """
            
            response = ollama.chat(
                model=model, 
                messages=[
                    {"role": "system", "content": "You create concise executive summaries that capture the essence of complex documents."},
                    {"role": "user", "content": overall_prompt}
                ]
            )
            
            exec_summary = response["message"]["content"] if "message" in response and "content" in response["message"] else "[Error generating executive summary]"
            complete_summary = "# Executive Summary\n\n{}\n\n# Detailed Section Summaries\n\n".format(exec_summary) + "".join(summary)
            return complete_summary
        except Exception as e:
            logger.error("Error generating overall summary: {}".format(str(e)))
            return "# Detailed Section Summaries\n\n" + "".join(summary)
    else:
        return "".join(summary)

# Enhanced RAG-based query answering
def summarize_with_rag(user_query, model="llama3.2:3b", max_retries=3):
    try:
        # Get relevant chunks for the query
        relevant_chunks = get_relevant_chunks(user_query, k=5)
        if not relevant_chunks:
            return "I don't have enough context from the PDF to answer that question. Could you try rephrasing or asking something else?"

        # Combine chunks with query for context
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        system_prompt = """You are a helpful and knowledgeable assistant specializing in analyzing PDF documents. Your task is to provide detailed, accurate, and natural responses to questions about the document content.

Rules:
1. Base your answers ONLY on the provided context - never make up information
2. If the context doesn't contain enough information, honestly say so
3. Be conversational and engaging while maintaining professionalism
4. Format responses using markdown for better readability
5. If the user's query is a greeting or casual conversation, respond naturally while gently reminding them you're here to help with the PDF content
6. For complex queries, break down your response into organized sections
7. Always cite specific details from the context when available"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from PDF:\n{context}\n\nUser Query: {user_query}"}
        ]

        # Handle casual greetings
        greetings = ["hi", "hello", "hey", "hii"]
        if user_query.lower().strip() in greetings:
            return "Hello! I'm here to help you understand your PDF document. Feel free to ask me any specific questions about its content!"

        response = ollama.chat(model, messages=messages, options={
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9
        })

        answer = response["message"]["content"]
        
        # Log successful interaction
        logger.info(f"Successfully processed chat query. Query length: {len(user_query)}, Response length: {len(answer)}")
        
        return answer

    except Exception as e:
        logger.error(f"Error in summarize_with_rag: {str(e)}")
        return "I apologize, but I encountered an error processing your request. Please try again or rephrase your question."

# Enhanced function to retrieve relevant chunks using Faiss
def retrieve_relevant_chunks(query, top_k=5):
    """
    Retrieve the most relevant chunks from Faiss index with improved robustness
    """
    global faiss_index, metadata
    
    if faiss_index.ntotal == 0:
        logger.warning("Vector database is empty - no documents to search")
        return []
    
    try:
        # Generate query embedding
        query_embedding = get_embedder().encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        distances, indices = faiss_index.search(query_embedding, min(top_k, faiss_index.ntotal))
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(metadata):  # Ensure valid index
                # Convert distance to similarity score
                similarity_score = float(distances[0][i])
                
                # Skip results with very low similarity
                if similarity_score < 0.1:  # Filter out irrelevant results
                    continue
                
                results.append({
                    "text": metadata[idx]["text"],
                    "score": similarity_score,
                    "source": metadata[idx]["pdf_filename"],
                    "chunk_id": metadata[idx]["chunk_id"]
                })
        
        logger.info(f"Retrieved {len(results)} relevant chunks for query: {query[:50]}...")
        return results
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        return []

# Function to add documents to Faiss index - improved for chunking robustness
def add_to_faiss_index(pdf_filename, text_chunks):
    global faiss_index, metadata
    
    if not text_chunks:
        logger.warning("No text chunks to add to the index")
        return 0
        
    logger.info("Adding {} chunks to FAISS index from {}".format(len(text_chunks), pdf_filename))
    
    # Generate embeddings
    try:
        # Process in batches if there are many chunks
        BATCH_SIZE = 1
        total_added = 0
        original_metadata = metadata.copy()  # Backup current metadata
        original_index_ntotal = faiss_index.ntotal
        
        # For small number of chunks, process all at once
        if len(text_chunks) <= BATCH_SIZE:
            logger.info("Small document, processing all chunks at once")
            embeddings = get_embedder().encode(text_chunks)
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to the index
            current_size = faiss_index.ntotal
            faiss_index.add(embeddings)
            
            # Add metadata
            for i, chunk in enumerate(text_chunks):
                metadata.append({
                    "pdf_filename": pdf_filename,
                    "text": chunk,
                    "chunk_id": current_size + i
                })
            total_added = len(text_chunks)
            logger.info("Added {} chunks in single batch".format(total_added))
        else:
            # Process in batches for larger documents
            for i in range(0, len(text_chunks), BATCH_SIZE):
                batch = text_chunks[i:i+BATCH_SIZE]
                logger.info("Processing batch {}/{} ({} chunks)".format(
                    i//BATCH_SIZE + 1,
                    (len(text_chunks)-1)//BATCH_SIZE + 1,
                    len(batch)
                ))
                
                embeddings = get_embedder().encode(batch)
                faiss.normalize_L2(embeddings)
                
                # Get the current index size to assign new IDs
                current_size = faiss_index.ntotal
                
                # Add embeddings to the index
                faiss_index.add(embeddings)
                
                # Add metadata
                for j, chunk in enumerate(batch):
                    metadata.append({
                        "pdf_filename": pdf_filename,
                        "text": chunk,
                        "chunk_id": current_size + j
                    })
                
                total_added += len(batch)
                logger.info("Added {} chunks, running total: {}".format(len(batch), total_added))
        
        # Save updated index and metadata
        try:
            logger.info("Saving FAISS index to {}".format(index_file))
            faiss.write_index(faiss_index, index_file)
            logger.info("FAISS index saved successfully")

            logger.info("Saving metadata to {}".format(metadata_file))
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info("Metadata saved successfully")
            return total_added
        except Exception as e:
            logger.error("Error saving files: {}".format(str(e)))
            # Rollback changes on save failure
            metadata = original_metadata
            faiss_index = faiss.read_index(index_file)
            return 0
            
    except Exception as e:
        logger.error("Error generating embeddings: {}".format(str(e)))
        return 0

# Improved function to create a downloadable PDF
def text_to_pdf(text, title="Generated Summary"):
    """
    Create a well-formatted PDF from text with better handling of special characters
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(5)
    
    # Add timestamp
    pdf.set_font("Arial", 'I', size=10)
    pdf.cell(0, 10, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # Main content
    pdf.set_font("Arial", size=12)
    
    # Handle markdown-style headings
    lines = text.split('\n')
    for line in lines:
        if line.startswith('# '):
            # Main heading
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(0, 10, line[2:], ln=True)
            pdf.ln(2)
        elif line.startswith('## '):
            # Subheading
            pdf.set_font("Arial", 'B', size=13)
            pdf.cell(0, 10, line[3:], ln=True)
            pdf.ln(2)
        elif line.startswith('### '):
            # Sub-subheading
            pdf.set_font("Arial", 'B', size=12)
            pdf.cell(0, 10, line[4:], ln=True)
            pdf.ln(2)
        elif line.strip() == '':
            # Empty line
            pdf.ln(5)
        else:
            # Regular text
            pdf.set_font("Arial", size=11)
            # Encode text to handle special characters
            try:
                # First try to clean problematic characters
                cleaned_line = ''.join(c if ord(c) < 128 or ord(c) > 159 else ' ' for c in line)
                encoded_line = cleaned_line.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 6, encoded_line)
            except Exception as e:
                logger.error(f"Error encoding text for PDF: {str(e)}")
                pdf.multi_cell(0, 6, "[Error encoding text]")

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output.getvalue()

# Function to troubleshoot PDF processing issues
def troubleshoot_pdf_processing(pdf_path):
    """
    Troubleshoot PDF processing issues by analyzing the content and chunking
    """
    # Extract and analyze the PDF text
    text, errors = extract_text_from_pdf(pdf_path)
    
    logger.info(f"=== PDF TROUBLESHOOTING REPORT ===")
    logger.info(f"Extracted text length: {len(text)} characters")
    logger.info(f"Extraction errors: {errors}")
    
    # Check for paragraph breaks
    paragraph_count = text.count('\n\n')
    logger.info("Paragraph breaks found: {}".format(paragraph_count))
    
    # Check for sentence breaks
    sentence_count = len(re.findall(r'[.!?]\s+', text))
    logger.info(f"Approximate sentences found: {sentence_count}")
    
    # Try cleaning and see if that affects paragraph counts
    cleaned_text = clean_text(text)
    logger.info(f"Cleaned text length: {len(cleaned_text)} characters")
    logger.info("Paragraph breaks after cleaning: {}".format(cleaned_text.count('\n\n')))
    
    # Try different chunking methods
    chunks_2000 = improved_semantic_text_chunking(cleaned_text, max_chunk_size=2000)
    logger.info(f"Chunking with size 2000 produces {len(chunks_2000)} chunks")
    
    chunks_4000 = improved_semantic_text_chunking(cleaned_text, max_chunk_size=4000)
    logger.info(f"Chunking with size 4000 produces {len(chunks_4000)} chunks")
    
    # Log the results
    result = {
        "text_length": len(text),
        "cleaned_text_length": len(cleaned_text),
        "paragraph_count": paragraph_count,
        "sentence_count": sentence_count,
        "chunks_2000": len(chunks_2000),
        "chunks_4000": len(chunks_4000)
    }
    
    logger.info(f"=== END OF TROUBLESHOOTING REPORT ===")
    return result

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if files are in the request
        if 'files[]' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        # Try both 'files[]' and 'file' keys
        files = request.files.getlist('files[]') if 'files[]' in request.files else [request.files['file']]
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400

        uploaded_files = []
        failed_files = []
        last_text = None
        last_filename = None

        for file in files:
            try:
                if not file or not file.filename:
                    continue
                    
                if not file.filename.lower().endswith('.pdf'):
                    failed_files.append({
                        'name': file.filename,
                        'error': 'Invalid file format. Please upload PDF files only.'
                    })
                    continue

                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Ensure the upload directory exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # Save the file
                file.save(filepath)
                logger.info(f"File saved: {filename}")
                
                # Verify file was saved and is readable
                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    failed_files.append({
                        'name': filename,
                        'error': 'Failed to save file or file is empty'
                    })
                    continue
                
                # Extract and process text
                try:
                    text = extract_text_from_pdf(filepath)
                    if not text or len(text.strip()) == 0:
                        failed_files.append({
                            'name': filename,
                            'error': 'Could not extract text from the PDF'
                        })
                        continue
                except Exception as e:
                    logger.error(f"Error extracting text from {filename}: {str(e)}")
                    failed_files.append({
                        'name': filename,
                        'error': f'Error processing PDF: {str(e)}'
                    })
                    continue
                    
                # Clean and chunk text
                text = clean_text(text)
                chunks = improved_semantic_text_chunking(text)
                
                # Add to FAISS index
                try:
                    add_to_faiss_index(filename, chunks)
                    uploaded_files.append(filename)
                    
                    # Store last successful file's info for session
                    last_text = text
                    last_filename = filename
                    
                    logger.info(f"Successfully processed {filename}")
                except Exception as e:
                    logger.error(f"Error indexing {filename}: {str(e)}")
                    failed_files.append({
                        'name': filename,
                        'error': f'Failed to index file: {str(e)}'
                    })
            except Exception as e:
                failed_files.append({
                    'name': file.filename,
                    'error': str(e)
                })

        # Store the last successful file's text in session
        if last_text and last_filename:
            session['pdf_text'] = last_text
            session['pdf_filename'] = last_filename
            logger.info(f"Stored text from {last_filename} in session for summary generation")

        # Prepare response message
        if len(uploaded_files) > 0:
            message = f'Successfully processed {len(uploaded_files)} file(s)'
            if failed_files:
                message += f', {len(failed_files)} file(s) failed'
        else:
            if failed_files:
                return jsonify({
                    'error': 'All files failed to process',
                    'failed_files': failed_files
                }), 400
            else:
                return jsonify({'error': 'No valid files were uploaded'}), 400

        response_data = {
            'message': message,
            'files': uploaded_files,
        }
        
        if failed_files:
            response_data['failed_files'] = failed_files

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    try:
        # Get text from session
        text = session.get('pdf_text')
        if not text:
            logger.error("No PDF text found in session")
            return jsonify({'error': 'Please upload a PDF first'}), 400
            
        # Get PDF filename for context
        pdf_filename = session.get('pdf_filename')
        if not pdf_filename:
            logger.error("No PDF filename found in session")
            return jsonify({'error': 'PDF file information not found'}), 400
            
        logger.info(f"Generating summary for PDF: {pdf_filename}")
        logger.info(f"Text length: {len(text)} characters")
            
        summary = summarize_text_with_headings(text)
        if not summary:
            logger.error("Summary generation failed")
            return jsonify({'error': 'Failed to generate summary'}), 500
            
        # Store summary in session
        session['current_summary'] = summary
        logger.info(f"Successfully generated summary of length: {len(summary)} characters")
        return jsonify({'summary': summary})
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/list_pdfs")
def list_pdfs():
    """List all PDF files in the upload folder"""
    try:
        # Get all PDF files from the upload folder
        pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                    if f.endswith('.pdf')]
        return jsonify(pdf_files)
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/view_pdf")
def view_pdf():
    """Serve a PDF file for viewing"""
    try:
        filename = request.args.get('filename')
        if not filename:
            return "No filename provided", 400
            
        # Ensure the filename is secure
        filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return "File not found", 404
            
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving PDF: {str(e)}")
        return str(e), 500

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logger.error('Invalid request payload: Missing or malformed JSON')
            return jsonify({'error': 'Invalid request payload'}), 400
    except Exception as e:
        logger.error(f'Error parsing JSON: {str(e)}')
        return jsonify({'error': 'Failed to parse JSON'}), 400

    user_query = data['message'].strip()
    if not user_query:
        return jsonify({'error': 'Empty query provided'}), 400

    try:
        response = summarize_with_rag(user_query)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            logger.error('Invalid request payload: Missing or malformed JSON')
            return jsonify({'error': 'Invalid request payload'}), 400
    except Exception as e:
        logger.error(f'Error parsing JSON: {str(e)}')
        return jsonify({'error': 'Failed to parse JSON'}), 400

    query = data['query'].strip()
    if not query:
        return jsonify({'error': 'Empty keyword provided'}), 400

    try:
        chunks = get_relevant_chunks(query)
        return jsonify({'results': chunks})
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_relevant_chunks(query, k=3, min_similarity=0.3):
    """
    Get the most relevant chunks for a query with improved context handling
    """
    try:
        # Get query embedding
        query_embedding = get_embedder().encode([query])
        faiss.normalize_L2(query_embedding)

        # Search in FAISS index
        D, I = faiss_index.search(query_embedding, k=min(k, faiss_index.ntotal))
        
        if len(I) == 0 or len(I[0]) == 0:
            logger.warning("No chunks found for query")
            return []

        # Filter by similarity threshold
        results = []
        for score, idx in zip(D[0], I[0]):
            if score < min_similarity:
                continue
                
            if idx >= len(metadata):
                logger.warning("Invalid chunk index: {}".format(idx))
                continue

            chunk_data = metadata[idx]
            results.append({
                "text": chunk_data["text"],
                "score": float(score),
                "source": chunk_data["pdf_filename"]
            })

        # Sort by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info("Found {} relevant chunks for query".format(len(results)))
        return results

    except Exception as e:
        logger.error("Error getting relevant chunks: {}".format(str(e)))
        return []

@app.route('/download_summary', methods=['GET'])
def download_summary():
    try:
        # Get summary from session
        summary = session.get('current_summary')
        if not summary:
            return jsonify({'error': 'No summary available. Generate a summary first.'}), 400

        logger.info("Creating PDF for summary")
        pdf_bytes = text_to_pdf(summary, "Document Summary")
        
        # Create a BytesIO object and write the PDF to it
        pdf_buffer = BytesIO(pdf_bytes)
        pdf_buffer.seek(0)
        
        # Send the file
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"summary_{int(time.time())}.pdf"
        )
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error creating PDF: {str(e)}'}), 500

@app.route('/clear_database', methods=['POST'])
def clear_database():
    global faiss_index, metadata
    try:
        # Reset the index and metadata
        faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata = []
        faiss.write_index(faiss_index, index_file)
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info("Vector database cleared successfully")
        return jsonify({'success': True, 'message': 'Vector database cleared successfully!'})
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error clearing database: {str(e)}'}), 500

@app.route('/delete_file', methods=['POST'])
def delete_file():
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400

        # Delete from uploads directory
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted file: {filepath}")

        # Delete from FAISS index
        try:
            delete_from_faiss_index(filename)
        except Exception as e:
            logger.error(f"Error deleting from FAISS index: {str(e)}")
            return jsonify({'error': f'Error deleting from index: {str(e)}'}), 500

        return jsonify({'message': f'Successfully deleted {filename}'})

    except Exception as e:
        logger.error(f"Error in delete_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

def delete_from_faiss_index(filename):
    global faiss_index, metadata
    try:
        # Filter metadata to remove chunks from the specified file
        metadata = [chunk for chunk in metadata if chunk['pdf_filename'] != filename]
        
        # Rebuild the FAISS index
        embeddings = get_embedder().encode([chunk['text'] for chunk in metadata])
        faiss.normalize_L2(embeddings)
        faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        faiss_index.add(embeddings)
        
        # Update chunk IDs in metadata
        for i, chunk in enumerate(metadata):
            chunk['chunk_id'] = i
        
        # Save updated index and metadata
        faiss.write_index(faiss_index, index_file)
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Successfully deleted {filename} from FAISS index")
    except Exception as e:
        logger.error(f"Error deleting from FAISS index: {str(e)}")
        raise e

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)