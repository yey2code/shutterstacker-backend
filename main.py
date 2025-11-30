import os
import base64
import json
import subprocess
import shutil
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd
from ftplib import FTP

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
MOUNTED_PICTURES_DIR = "/mnt/pictures"
PROCESSED_DIR = "/app/processed"

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

class ProcessRequest(BaseModel):
    source_folder: str
    selected_images: List[str]
    project_name: str
    api_key: str

class MetadataItem(BaseModel):
    filename: str
    title: str
    description: str
    keywords: str
    category: str

class EmbedRequest(BaseModel):
    project_name: str
    metadata: List[MetadataItem]

class UploadRequest(BaseModel):
    project_name: str
    ftp_user: str
    ftp_pass: str

@app.get("/api/list-images")
def list_images(path: str = ""):
    """Lists images in the Docker-mounted directory."""
    full_path = os.path.join(MOUNTED_PICTURES_DIR, path)
    
    # Security check to prevent traversing up
    if not os.path.abspath(full_path).startswith(MOUNTED_PICTURES_DIR):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(full_path):
         raise HTTPException(status_code=404, detail="Path not found")

    items = []
    try:
        for entry in os.scandir(full_path):
            if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append({"name": entry.name, "type": "file", "path": os.path.join(path, entry.name)})
            elif entry.is_dir():
                items.append({"name": entry.name, "type": "directory", "path": os.path.join(path, entry.name)})
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    return items

@app.post("/api/process-with-gemini")
def process_with_gemini(request: ProcessRequest):
    """
    1. Creates project directory.
    2. Copies images.
    3. Sends to Gemini.
    4. Saves CSV.
    """
    project_path = os.path.join(PROCESSED_DIR, request.project_name)
    os.makedirs(project_path, exist_ok=True)
    
    results = []
    
    for image_rel_path in request.selected_images:
        # Source path (from mounted volume)
        src_path = os.path.join(MOUNTED_PICTURES_DIR, image_rel_path)
        filename = os.path.basename(image_rel_path)
        dst_path = os.path.join(project_path, filename)
        
        # Copy file
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {src_path}: {e}")
            continue

        # Encode image
        with open(dst_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Call Gemini API
        # Using specific version 001 for stability
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={request.api_key}"
        headers = {'Content-Type': 'application/json'}
        
        prompt_text = "Analyze this image for stock photography. Return JSON with keys: Title, Description, Keywords (comma separated), Category (Choose from Shutterstock categories)."
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {
                        "mime_type": "image/jpeg", # Assuming JPEG for simplicity, could detect
                        "data": encoded_string
                    }}
                ]
            }]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            if not response.ok:
                print(f"Gemini API Error: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()
            
            # Extract text response
            text_content = data['candidates'][0]['content']['parts'][0]['text']
            
            # Parse JSON from text (Gemini might wrap in markdown code blocks)
            clean_json = text_content.replace("```json", "").replace("```", "").strip()
            metadata = json.loads(clean_json)
            
            results.append({
                "Filename": filename,
                "Title": metadata.get("Title", ""),
                "Description": metadata.get("Description", ""),
                "Keywords": metadata.get("Keywords", ""),
                "Category": metadata.get("Category", "")
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({
                "Filename": filename,
                "Title": "Error",
                "Description": str(e),
                "Keywords": "",
                "Category": ""
            })

    # Save to CSV
    csv_path = os.path.join(project_path, "shutterstock.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    return {"status": "success", "csv_path": csv_path, "project_name": request.project_name}

@app.get("/api/get-metadata")
def get_metadata(project_name: str):
    """Returns the CSV data as JSON."""
    csv_path = os.path.join(PROCESSED_DIR, project_name, "shutterstock.csv")
    print(f"Attempting to read CSV from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        raise HTTPException(status_code=404, detail=f"Project CSV not found at {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        # Handle NaN values which JSON standard doesn't support
        df = df.fillna("")
        return df.to_dict(orient="records")
    except pd.errors.EmptyDataError:
        print("CSV is empty")
        return []
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {str(e)}")

@app.post("/api/embed-metadata")
def embed_metadata(request: EmbedRequest):
    """Embeds metadata into images using exiftool."""
    project_path = os.path.join(PROCESSED_DIR, request.project_name)
    
    success_count = 0
    errors = []

    for item in request.metadata:
        image_path = os.path.join(project_path, item.filename)
        if not os.path.exists(image_path):
            errors.append(f"File not found: {item.filename}")
            continue
            
        # Construct exiftool command
        # Mapping to standard IPTC/XMP fields commonly used by stock sites
        cmd = [
            "exiftool",
            "-overwrite_original",
            f"-Title={item.title}",
            f"-Description={item.description}",
            f"-Keywords={item.keywords}",
            f"-Category={item.category}", # Custom field, might not map directly to standard XMP without config, but useful
            f"-IPTC:Caption-Abstract={item.description}",
            f"-IPTC:Keywords={item.keywords}",
            f"-XMP:Title={item.title}",
            f"-XMP:Description={item.description}",
            f"-XMP:Subject={item.keywords}",
            image_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            success_count += 1
        except subprocess.CalledProcessError as e:
            errors.append(f"Exiftool error for {item.filename}: {e.stderr}")

    return {"status": "completed", "success_count": success_count, "errors": errors}

@app.post("/api/upload")
def upload_to_ftp(request: UploadRequest):
    """Uploads images to Shutterstock FTP."""
    project_path = os.path.join(PROCESSED_DIR, request.project_name)
    
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")

    uploaded_files = []
    errors = []

    try:
        with FTP("ftp.shutterstock.com") as ftp:
            ftp.login(user=request.ftp_user, passwd=request.ftp_pass)
            
            for filename in os.listdir(project_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.eps', '.mov', '.mp4')): # Shutterstock accepted formats
                    file_path = os.path.join(project_path, filename)
                    try:
                        with open(file_path, "rb") as f:
                            ftp.storbinary(f"STOR {filename}", f)
                        uploaded_files.append(filename)
                    except Exception as e:
                        errors.append(f"Failed to upload {filename}: {str(e)}")
                        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FTP Connection failed: {str(e)}")

    return {"status": "completed", "uploaded": uploaded_files, "errors": errors}
