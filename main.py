import sys
# Override standard sqlite3 with pysqlite3 for ChromaDB compatibility on Render
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
from fastapi import FastAPI, HTTPException
# ... (the rest of your imports and code remain exactly the same)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from mistralai.client import Mistral

# --- Configuration ---
CHROMA_DB_PATH = "chroma_data"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
EMBED_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-large-latest"

if not MISTRAL_API_KEY:
    print("Error: MISTRAL_API_KEY environment variable not set.")
    sys.exit(1)

# Initialize FastAPI App
app = FastAPI(
    title="Upazila RAG API",
    description="Backend API for cross-lingual Bangladesh Upazila knowledge retrieval"
)

# Enable CORS so your WordPress server/frontend can communicate with it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Clients
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = chroma_client.get_collection(name="upazila_rag_collection")
except ValueError:
    print(f"Error: Chroma collection 'upazila_rag_collection' not found at {CHROMA_DB_PATH}")
    sys.exit(1)

# --- Pydantic Data Models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class SourceMetadata(BaseModel):
    wp_id: int
    title: str
    chunk_index: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

# --- Helper Functions ---
def get_chroma_context(query: str, top_k: int):
    """Embeds the query and fetches matching text chunks from ChromaDB."""
    # 1. Embed user query via Mistral
    embed_response = mistral_client.embeddings.create(
        model=EMBED_MODEL,
        inputs=[query]
    )
    query_embedding = embed_response.data[0].embedding
    
    # 2. Semantic query against ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results

def ask_mistral(query: str, context_text: str) -> str:
    """Sends context and query to Mistral using cross-lingual RAG instructions."""
    
    prompt = f"""You are a highly polite, professional, and helpful AI assistant specializing in information about Bangladesh Upazilas. 
    Your primary objective is to provide accurate and relevant answers to the user's queries based exclusively on the provided CONTEXT.

    Strict Instructions:
    1. Language & Tone: You must always respond in pure, formal Bengali. You are required to address the user with the highest level of respect, utilizing the formal pronoun 'আপনি' (Apni) in all interactions.
    2. Cross-Lingual Matching: The provided CONTEXT is written in Bengali. If the user asks a question in English or writes an Upazila name in English (e.g., "Thakurgaon"), you must internally map and translate it to its Bengali equivalent (e.g., "ঠাকুরগাঁও") to accurately extract information from the CONTEXT.
    3. Conciseness: Keep your answers brief, meaningful, and strictly to the point. Do not overwhelm the user with unnecessary information. Answer only what is asked.
    4. No Hallucination: You must answer using ONLY the facts present in the CONTEXT below. Do not assume, guess, or incorporate any outside knowledge.
    5. Strict Fallback Protocol: If the CONTEXT does not contain information about the specific Upazila the user is asking about, do not attempt to answer. You must output EXACTLY the following Bengali sentence and nothing else:
       "দুঃখিত, এই উপজেলার তথ্য সংগ্রহের কাজ চলছে , আপনি যদি এই উপজেলা সম্পর্কে তথ্য প্রদানে অংশগ্রহণ করতে চান তাহলে আমাদের সাথে যোগাযোগ করুন ।"
    6. Mandatory Slogan: You must append the following exact phrase at the very end of every successful response: 
       "১৫ বছরের বিস্ময়কর অগ্রযাত্রা বদলে যাওয়া এক আধুনিক বাংলাদেশের গল্প"

    CONTEXT:
    {context_text}
    
    QUESTION:
    {query}
    """
    
    messages = [
        {"role": "system", "content": "You are a strict, highly polite cross-lingual RAG assistant. Rely exclusively on the provided context, respond ONLY in formal Bengali using 'আপনি' (Apni), and execute the fallback rules with absolute precision."},
        {"role": "user", "content": prompt}
    ]
    
    chat_response = mistral_client.chat.complete(
        model=CHAT_MODEL,
        messages=messages
    )
    return chat_response.choices[0].message.content

# --- API Endpoints ---
@app.get("/")
def health_check():
    """Simple endpoint for Render to monitor service health."""
    return {"status": "healthy", "database": "connected"}

@app.post("/api/chat", response_model=QueryResponse)
def run_rag_pipeline(request: QueryRequest):
    """Main RAG endpoint to submit a query and get a synthesized answer with sources."""
    try:
        # Step 1: Retrieve from Vector DB
        raw_results = get_chroma_context(request.query, request.top_k)
        
        documents = raw_results['documents'][0]
        metadatas = raw_results['metadatas'][0]
        
        if not documents:
            return QueryResponse(
                answer="I don't have enough information in my database to answer that.",
                sources=[]
            )
            
        # Format the context block for the LLM
        context_text = "\n\n---\n\n".join(documents)
        
        # Step 2: Generate Answer via Mistral
        answer = ask_mistral(request.query, context_text)
        
        # Format sources to send back to WordPress
        formatted_sources = [
            {"wp_id": meta["wp_id"], "title": meta["title"], "snippet": doc[:150] + "..."}
            for doc, meta in zip(documents, metadatas)
        ]
        
        return QueryResponse(answer=answer, sources=formatted_sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
