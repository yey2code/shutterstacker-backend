import os
import result
import base64
import json
import subprocess
import shutil
import uuid
import logging
import asyncio
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
from ftplib import FTP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Coolify/dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
TEMP_DIR = "/app/temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Mount temp dir for serving images to frontend
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

# --- Pydantic Models ---

class AnalyzeRequest(BaseModel):
    session_id: str
    api_key: str
    context_map: Optional[Dict[str, str]] = {}

class MetadataItem(BaseModel):
    filename: str
    title: str
    description: str
    keywords: str
    category: str

class EmbedUploadRequest(BaseModel):
    session_id: str
    project_name: str # kept for compatibility/folder naming if needed, though session_id is primary
    metadata: List[MetadataItem]
    ftp_user: str
    ftp_pass: str
    ftp_host: str = "ftp.shutterstock.com"

# --- Helper Functions ---

def cleanup_session(session_id: str):
    """Deletes the session directory."""
    session_path = os.path.join(TEMP_DIR, session_id)
    if os.path.exists(session_path):
        try:
            shutil.rmtree(session_path)
            logger.info(f"Cleaned up session: {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

# --- Endpoints ---

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accepts list of files. Creates a new session ID.
    Saves files to /app/temp/{session_id}/.
    Returns session_id and list of filenames.
    """
    session_id = str(uuid.uuid4())
    session_path = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)

    file_list = []

    for file in files:
        try:
            file_location = os.path.join(session_path, file.filename)
            # Safe async file write
            content = await file.read()
            with open(file_location, 'wb') as f:
                f.write(content)
            
            # Construct accessible URL (assuming frontend can access /temp via proxy or direct)
            # For this setup, we return filename and frontend constructs URL: API_URL + /temp/ + session_id + / + filename
            file_list.append(file.filename)
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            # Continue with other files or raise? Continuing seems better for UX.

    return {"session_id": session_id, "files": file_list}

@app.post("/analyze")
async def analyze_images(request: AnalyzeRequest):
    """
    1. Iterates images in session.
    2. Sends to Gemini with optional context.
    3. Returns JSON list of metadata.
    """
    session_path = os.path.join(TEMP_DIR, request.session_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")

    results = []

    # Get list of image files
    try:
        image_files = [f for f in os.listdir(session_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error reading session dir: {e}")

    # Gemini Setup
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={request.api_key}"
    headers = {'Content-Type': 'application/json'}

    for filename in image_files:
        filepath = os.path.join(session_path, filename)
        
        # Get Context
        user_context = request.context_map.get(filename, "")
        
        system_prompt = (
            "Analyze this image for stock photography. "
            "Return PURE JSON (no markdown formatting) with keys: Title, Description, Keywords (comma separated string), Category (Choose from standard stock categories). "
            f"Additional Context provided by user: '{user_context}'. Override visual inferences if this context contradicts them."
        )

        try:
             with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

             payload = {
                "contents": [{
                    "parts": [
                        {"text": system_prompt},
                        {"inline_data": {
                            "mime_type": "image/jpeg", # simplified
                            "data": encoded_string
                        }}
                    ]
                }]
            }
             
             response = requests.post(url, headers=headers, json=payload)
             response.raise_for_status()
             data = response.json()
             
             # Extract text response - handling potential structure variations
             try:
                text_content = data['candidates'][0]['content']['parts'][0]['text']
             except (KeyError, IndexError):
                 logger.error(f"Unexpected Gemini response structure: {data}")
                 raise ValueError("Invalid API response format")

             # Cleanup json
             clean_json = text_content.replace("```json", "").replace("```", "").strip()
             metadata = json.loads(clean_json)
             
             results.append({
                 "filename": filename,
                 "title": metadata.get("Title", ""),
                 "description": metadata.get("Description", ""),
                 "keywords": metadata.get("Keywords", ""),
                 "category": metadata.get("Category", "")
             })

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results.append({
                "filename": filename,
                "title": "Error Processing",
                "description": str(e),
                "keywords": "",
                "category": ""
            })

    return {"results": results}

@app.post("/embed-and-upload")
async def embed_and_upload(request: EmbedUploadRequest, background_tasks: BackgroundTasks):
    """
    1. Embed metadata via ExifTool.
    2. Upload to FTP.
    3. Trigger cleanup.
    """
    session_path = os.path.join(TEMP_DIR, request.session_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")

    embed_errors = []
    
    # --- Embedding ---
    for item in request.metadata:
        image_path = os.path.join(session_path, item.filename)
        if not os.path.exists(image_path):
            continue

        cmd = [
            "exiftool",
            "-overwrite_original",
            f"-Title={item.title}",
            f"-Description={item.description}",
            f"-Keywords={item.keywords}",
            f"-Category={item.category}", 
            f"-IPTC:Caption-Abstract={item.description}",
            f"-IPTC:Keywords={item.keywords}",
            f"-XMP:Title={item.title}",
            f"-XMP:Description={item.description}",
            f"-XMP:Subject={item.keywords}",
            image_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            embed_errors.append(f"ExifTool failed for {item.filename}: {e.stderr}")

    if embed_errors:
        logger.warning(f"Embedding errors: {embed_errors}")
        # We continue to upload what we can, or return error? 
        # Requirement says "execute embedding... return logs". We will return logs but proceed.

    # --- FTP Upload ---
    uploaded_files = []
    upload_errors = []

    try:
        with FTP(request.ftp_host) as ftp:
            ftp.login(user=request.ftp_user, passwd=request.ftp_pass)
            
            for item in request.metadata:
                filename = item.filename
                file_path = os.path.join(session_path, filename)
                
                if not os.path.exists(file_path): 
                    continue

                try:
                    with open(file_path, "rb") as f:
                        ftp.storbinary(f"STOR {filename}", f)
                    uploaded_files.append(filename)
                except Exception as e:
                    upload_errors.append(f"FTP Upload failed for {filename}: {e}")
                    
    except Exception as e:
         return {"status": "failed", "error": f"FTP Connection failed: {str(e)}", "embed_errors": embed_errors}

    # Queue Cleanup
    background_tasks.add_task(cleanup_session, request.session_id)

    return {
        "status": "completed",
        "uploaded": uploaded_files,
        "upload_errors": upload_errors,
        "embed_errors": embed_errors
    }
