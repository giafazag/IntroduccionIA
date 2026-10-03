import os
from fastapi import FastAPI, UploadFile, File, Form
import google.generativeai as genai
import chromadb
from pypdf import PdfReader

# Configura tu API Key de Google AI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AQ.Ab8RN6JnxOAsHqoN-cyBu5Kl5AnNNQ_cy8Lg8s274PvXFaGtVg")
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="RAG Backend API")

# Inicializar cliente de ChromaDB persistente
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Usamos un modelo nativo de embeddings para Chroma o registramos el comportamiento
collection = chroma_client.get_or_create_collection(name="rag_documents")

def get_google_embedding(text: str):
    """Genera embeddings usando la API oficial de Google GenAI."""
    response = genai.embed_content(
        model="models/text-embedding-004",
        contents=text,
        task_type="retrieval_document"
    )
    return response['embedding']

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Divide el texto en fragmentos (chunks) con solape."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Endpoint para cargar un PDF, procesarlo y añadirlo a ChromaDB."""
    try:
        pdf_reader = PdfReader(file.file)
        full_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        if not full_text.strip():
            return {"status": "error", "message": "El documento está vacío o no se pudo extraer texto."}
        
        chunks = split_text(full_text)
        
        # Guardar fragmentos y embeddings en ChromaDB
        for i, chunk in enumerate(chunks):
            embedding = get_google_embedding(chunk)
            chunk_id = f"{file.filename}_chunk_{i}"
            
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": file.filename}],
                ids=[chunk_id]
            )
            
        return {"status": "success", "message": f"Procesado exitosamente: {len(chunks)} fragmentos guardados."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/query")
async def query_rag(question: str = Form(...)):
    """Endpoint para consultar al sistema RAG."""
    try:
        # 1. Obtener embedding de la pregunta de consulta
        query_response = genai.embed_content(
            model="models/text-embedding-004",
            contents=question,
            task_type="retrieval_query"
        )
        query_embedding = query_response['embedding']
        
        # 2. Recuperar los fragmentos más relevantes de ChromaDB (k=3)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        retrieved_docs = results.get("documents", [[]])[0]
        context = "\n---\n".join(retrieved_docs) if retrieved_docs else "No se encontró contexto relevante."
        
        # 3. Generar la respuesta usando Gemini aumentándola con el contexto
        prompt = f"""
        Eres un asistente inteligente. Responde a la pregunta del usuario utilizando únicamente el contexto provisto a continuación. 
        Si el contexto no contiene la respuesta, di que no posees la información suficiente.

        Contexto:
        {context}

        Pregunta: {question}
        Respuesta:
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        llm_response = model.generate_content(prompt)
        
        return {
            "status": "success",
            "answer": llm_response.text,
            "sources": results.get("metadatas", [[]])[0]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
