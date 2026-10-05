# 🤖 Sistema RAG (Streamlit + FastAPI + ChromaDB + Google AI)

Este proyecto implementa un sistema de **Generación Aumentada por Recuperación (RAG)** totalmente desacoplado. Permite subir documentos en formato PDF, procesar su contenido, indexar sus vectores en una base de datos local y realizar consultas conversacionales utilizando los modelos de última generación de Google.

---

##  Arquitectura del Sistema

* **Frontend (Streamlit):** Interfaz de usuario interactiva que proporciona un chat fluido y un panel lateral para la gestión y carga de archivos PDF.
* **Backend (FastAPI):** API REST encargada de la extracción de texto, fragmentación (*chunking*), almacenamiento de vectores y orquestación del flujo RAG.
* **Base de Datos Vectorial (ChromaDB):** Almacenamiento e indexación persistente de los fragmentos de texto en vectores locales.
* **Modelos de Google GenAI (SDK `google-genai`):**
  * `gemini-embedding-001` (Generación de vectores/embeddings).
  * `gemini-3.8-flash` (Interactions API para responder preguntas basadas en el contexto).

---

## Requisitos e Instalación

### 1. Clonar el proyecto y preparar los archivos
Asegúrate de tener los siguientes archivos en tu directorio de trabajo:
* `backend.py` (Servidor de la API)
* `frontend.py` (Interfaz de usuario)
* `requirements.txt` (Dependencias del sistema)

### 2. Instalar dependencias
Abre una terminal en la ruta de tu proyecto e instala los paquetes necesarios ejecutando:

```bash
pip install -r requirements.txt
```

---

## Configuración de las Credenciales

El sistema utiliza el nuevo SDK oficial de Google, el cual busca de forma predeterminada la variable de entorno `GEMINI_API_KEY`. 

Antes de ejecutar los servicios, obtén tu clave gratuita en [Google AI Studio](https://google.com) y asígnala en tu terminal según tu sistema operativo:

* **En Windows (PowerShell):**
  ```powershell
  $env:GEMINI_API_KEY="AIzaSyTuClaveAqui..."
  ```
* **En Windows (CMD tradicional):**
  ```cmd
  set GEMINI_API_KEY="AIzaSyTuClaveAqui..."
  ```
* **En Linux / macOS:**
  ```bash
  export GEMINI_API_KEY="AIzaSyTuClaveAqui..."
  ```

---

##  Instrucciones de Ejecución

El sistema requiere que mantengas **dos terminales abiertas al mismo tiempo**:

### Paso 1: Iniciar el Servidor Backend (FastAPI)
En tu primera terminal, levanta el servidor web con Uvicorn:
```powershell
python -m uvicorn backend:app --reload
```
Sabrás que inició correctamente cuando observes la línea: `Application startup complete`. El backend estará escuchando en `http://127.0.0.1:8000`.

### Paso 2: Iniciar la Interfaz Web (Streamlit)
Abre una **segunda terminal**, asegúrate de que la variable de entorno esté configurada también allí, y ejecuta el frontend llamando al módulo de Python:
```powershell
python -m streamlit run frontend.py
```
Esto abrirá automáticamente una pestaña en tu navegador web en la dirección local `http://localhost:8501`.

---

##  Modo de Uso

1. **Subir Conocimiento:** En la barra lateral izquierda de Streamlit, arrastra o selecciona un archivo PDF.
2. **Indexar:** Haz clic en el botón **"Procesar e Indexar Documento"**. El backend guardará de forma persistente los vectores en una carpeta local llamada `chroma_db/`.
3. **Preguntar:** En la caja de chat central, hazle preguntas al documento. El bot te responderá basándose estrictamente en la información contenida en el PDF e indicará las fuentes consultadas en la parte inferior.
