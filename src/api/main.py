from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import secrets
import json
import os
import torch
import sys
from typing import List, Dict, Optional
from src.api.logging_config import logger, log_request, log_performance, log_prediction
import time
from pathlib import Path
import gc
import psutil
import traceback
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import functions from run_inference.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from run_inference import predict_entities

# Get configuration from environment variables with defaults
MODEL_ID = os.environ.get("MODEL_ID", "Harshhhhhhh/NER")
USERNAME = os.environ.get("API_USERNAME", "admin")
PASSWORD = os.environ.get("API_PASSWORD", "password123")
ENABLE_AUTH = os.environ.get("ENABLE_AUTH", "true").lower() == "true"

# Create FastAPI app
app = FastAPI(
    title="Named Entity Recognition API",
    description="API for recognizing entities in text using a BERT NER model from Hugging Face",
    version="1.0.0"
)

# Global variables for lazy loading
tokenizer = None
model = None
id_to_tag = None

def log_memory_usage(label):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage ({label}): {memory_info.rss / 1024 / 1024:.2f} MB")

def initialize_model():
    """Load model from Hugging Face with aggressive memory optimizations"""
    global tokenizer, model, id_to_tag
    
    if tokenizer is not None and model is not None:
        return
    
    # Aggressive garbage collection before loading model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log_memory_usage("before_model_load")
    
    try:
        # Load directly from Hugging Face with optimizations
        logger.info(f"Loading model from Hugging Face: {MODEL_ID}")
        start_time = time.time()
        
        # Step 1: Load tokenizer first
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        gc.collect()
        
        # Step 2: Load model with 8-bit quantization
        logger.info("Loading model with quantization...")
        from transformers import BertForTokenClassification
        # Load with 8-bit quantization for memory efficiency
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID,
            torchscript=True,
            low_cpu_mem_usage=True,
            return_dict=False
        )
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Step 3: Apply dynamic quantization - significantly reduces memory usage
        logger.info("Applying dynamic quantization...")
        model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm}, 
            dtype=torch.qint8
        )
        
        # Force model to CPU mode and clear CUDA cache if needed
        model = model.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Step 4: Load ID to tag mapping
        if hasattr(model.config, "id2label"):
            id_to_tag = model.config.id2label
        else:
            num_labels = model.config.num_labels
            id_to_tag = {i: f"TAG_{i}" for i in range(num_labels)}
        
        # Free up memory
        gc.collect()
        
        logger.info(f"Model loaded and optimized in {time.time() - start_time:.2f} seconds")
        log_memory_usage("after_model_load")
        
    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}\n{traceback.format_exc()}")
        raise RuntimeError(f"Could not initialize model: {str(e)}")

security = HTTPBasic()

class TextRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    start: int
    end: int
    label: str

class PredictionResponse(BaseModel):
    entities: List[Entity]
    original_text: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if not ENABLE_AUTH:
        return "anonymous"
        
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.on_event("startup")
async def startup_event():
    """Initialize the model at startup if LAZY_LOADING is false"""
    if os.environ.get("LAZY_LOADING", "true").lower() != "true":
        try:
            initialize_model()
        except Exception as e:
            logger.error(f"Failed to preload model: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", 
        model_id=MODEL_ID,
        model_loaded=(model is not None)
    )

@app.get("/", dependencies=[Depends(authenticate)] if ENABLE_AUTH else [])
def root():
    return {"message": "BERT NER Model API is running. Use /predict endpoint for entity recognition."}

@app.post("/predict", response_model=PredictionResponse)
@log_request
async def predict(request_data: TextRequest, request: Request, credentials = Depends(authenticate) if ENABLE_AUTH else None):
    text = request_data.text
    start_time = time.time()
    
    try:
        log_memory_usage("before_prediction")
        
        # Ensure model is loaded
        if model is None or tokenizer is None or id_to_tag is None:
            initialize_model()
            
        with log_performance("text_processing"):
            # Process text in smaller chunks for memory efficiency
            max_length = 384  # Reduced from 512 for memory efficiency
            entity_dicts = []
            
            if len(text) > max_length:
                # Process long text in overlapping chunks with smaller overlap
                chunks = []
                for i in range(0, len(text), max_length - 30):
                    chunk = text[i:i + max_length]
                    chunks.append((i, chunk))
                
                for offset, chunk in chunks:
                    # Clear memory between chunk processing
                    if offset > 0:
                        gc.collect()
                    
                    chunk_entities = predict_entities(chunk, tokenizer, model, id_to_tag)
                    
                    # Adjust entity positions based on chunk offset
                    for entity in chunk_entities:
                        entity["start"] += offset
                        entity["end"] += offset
                    entity_dicts.extend(chunk_entities)
            else:
                entity_dicts = predict_entities(text, tokenizer, model, id_to_tag)
        
        # Convert to Entity objects
        entities = [
            Entity(
                text=ent["text"],
                start=ent["start"],
                end=ent["end"],
                label=ent["label"]
            ) for ent in entity_dicts
        ]
        
        processing_time = time.time() - start_time
        log_memory_usage("after_prediction")
        
        # Force garbage collection
        gc.collect()
        
        return PredictionResponse(
            entities=entities,
            original_text=text,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")