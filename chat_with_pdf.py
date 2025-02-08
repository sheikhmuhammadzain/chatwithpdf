import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai  # Using the google-generativeai library

# -----------------------------------------------------------------------------
# HARD-CODE YOUR GEMINI API KEY HERE (not recommended for production)
import os
import dotenv  # Import dotenv to load .env variables

# Load the .env file
dotenv.load_dotenv()

# Retrieve the API key from the environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Error: GEMINI_API_KEY is not set. Please check your .env file.")
# -----------------------------------------------------------------------------

def extract_text_from_pdf(pdf_file):
    """
    Extracts all text from a PDF file (expects a file-like object).
    """
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
    return text

def chunk_text(text, max_words=100):
    """
    Splits text into chunks of up to max_words words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def build_faiss_index(chunks, embedder):
    """
    Computes embeddings for each text chunk and builds a FAISS index.
    """
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    # Create a flat L2 index (simple but effective for small datasets)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def retrieve_context(query, embedder, index, chunks, top_k=3):
    """
    Retrieves the top_k most relevant chunks for the given query.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    # Gather the corresponding text chunks.
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

def call_llm(prompt, model="gemini-pro"):
    """
    Calls the Google Gemini API (via the Generative AI library) with the given prompt.
    Uses the specified model and generation parameters.
    """
    # Configure the API key (hard-coded above)
    genai.configure(api_key=GEMINI_API_KEY)

    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_output_tokens": 150,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    try:
        # Create a GenerativeModel instance with the desired model name and configurations.
        model_genai = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response = model_genai.generate_content(prompt)  # Call the API

        # Attempt to extract the generated text from the response.
        if response and hasattr(response, "text") and response.text:
            return response.text
        elif response and hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].text if response.candidates[0].text else "LLM API did not return any generated text in candidates."
        else:
            return "LLM API did not return any generated text."
    except Exception as e:
        return f"Error calling LLM API: {e}"

# -----------------------------------------------------------------------------
# Streamlit User Interface
st.title("Chat with Your PDF Developed By Zain Sheikh")
st.write("Upload a PDF file and ask questions about its content.")

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
    
    if text is None or not text.strip():
        st.error("No text could be extracted from the PDF.")
    else:
        st.success("PDF text extracted successfully!")
        # Chunk the text
        chunks = chunk_text(text, max_words=100)
        st.write(f"Total chunks created: {len(chunks)}")
        
        # Build FAISS index from the chunks using a sentence transformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        index, _ = build_faiss_index(chunks, embedder)
        
        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        st.write("### Chat with the PDF")
        user_query = st.text_input("Enter your question about the PDF:")
        if st.button("Submit Question") and user_query:
            # Retrieve relevant context for the query
            retrieved_chunks = retrieve_context(user_query, embedder, index, chunks, top_k=3)
            context = "\n".join(retrieved_chunks)
            prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
            
            st.info("Calling Gemini API for an answer...")
            answer = call_llm(prompt)
            
            # Append to chat history
            st.session_state.chat_history.append((user_query, answer))
        
        # Display chat history
        if st.session_state.chat_history:
            st.write("## Chat History")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")
