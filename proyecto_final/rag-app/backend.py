import os
from fastapi import FastAPI, UploadFile, File, Form
from google import genai
from google.genai import types
import chromadb
from pypdf import PdfReader

# Inicializa el nuevo cliente oficial de Google GenAI
# Lee automáticamente la variable de entorno GEMINI_API_KEY o GOOGLE_API_KEY
client = genai.Client()

app = FastAPI(title="RAG Backend API (Actualizado)")

# Inicializar cliente de ChromaDB persistente
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_documents")

def get_google_embedding(text: str):
    """Genera embeddings usando el nuevo SDK google-genai."""
    response = client.models.embed_content(
        model="gemini-embedding-001", 
        contents=text,
    )
    # El nuevo SDK devuelve una estructura de objetos, accedemos mediante atributos
    return response.embeddings[0].values

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
            return {"status": "error", "message": "El documento está vacío."}
        
        chunks = split_text(full_text)
        
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
    """Endpoint para consultar al sistema RAG usando el nuevo SDK."""
    try:
        # 1. Obtener embedding de la pregunta
        query_embedding = get_google_embedding(question)
        
        # 2. Recuperar los fragmentos de ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        retrieved_docs = results.get("documents", [[]])
        context = "\n---\n".join(retrieved_docs[0]) if retrieved_docs and retrieved_docs[0] else "No hay contexto."
        
        # 3. Generar la respuesta usando Gemini 2.5 o 1.5 con el nuevo cliente
        prompt = f"""
        Eres un asistente inteligente. Responde a la pregunta del usuario utilizando únicamente el contexto provisto a continuación. 
        Si el contexto no contiene la respuesta, di que no posees la información suficiente.

        Contexto:
        {context}

        Pregunta: {question}
        Respuesta:
        """
        
        llm_response = client.models.generate_content(
            model='gemini-2.5-flash',    # <-- Nombre de modelo actualizado
            contents=prompt,
        )
        
        return {
            "status": "success",
            "answer": llm_response.text,
            "sources": results.get("metadatas", [[]])[0]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
