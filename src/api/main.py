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
import torch.quantization

# Import functions from run_inference.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from run_inference import load_model, predict_entities

# Get configuration from environment variables with defaults
MODEL_ID = os.environ.get("MODEL_ID", "Harshhhhhhh/NER")
USERNAME = os.environ.get("API_USERNAME", "admin")
PASSWORD = os.environ.get("API_PASSWORD", "password123")
ENABLE_AUTH = os.environ.get("ENABLE_AUTH", "true").lower() == "true"
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/models")

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

def initialize_model():
    """Load model with memory optimizations"""
    global tokenizer, model, id_to_tag
    
    if tokenizer is not None and model is not None:
        return
    
    # Force garbage collection before loading model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create cache directory if it doesn't exist
    model_dir = os.path.join(MODEL_CACHE_DIR, MODEL_ID.replace("/", "_"))
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if model exists in cache
    model_exists = os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
    tokenizer_exists = os.path.exists(os.path.join(model_dir, "vocab.txt"))
    
    if model_exists and tokenizer_exists:
        logger.info(f"Loading model from local cache: {model_dir}")
        start_time = time.time()
        try:
            # Use load_model from run_inference.py
            tokenizer, model, id_to_tag = load_model(model_dir)
            
            # Verify components were actually loaded
            if tokenizer is None or model is None or id_to_tag is None:
                raise ValueError("One or more model components failed to load")
                
            logger.info(f"Model loaded from cache in {time.time() - start_time:.2f} seconds")
            return
        except Exception as e:
            logger.warning(f"Failed to load from cache, will download: {str(e)}")
    
    # Download from Hugging Face if not in cache
    logger.info(f"Downloading model from Hugging Face: {MODEL_ID}")
    start_time = time.time()
    
    try:
        # Try with standard BertTokenizer first if Fast version fails
        from transformers import BertTokenizer, BertTokenizerFast, BertForTokenClassification
        
        # Try loading with fast tokenizer first
        try:
            tokenizer = BertTokenizerFast.from_pretrained(MODEL_ID)
        except Exception as e:
            logger.warning(f"Fast tokenizer failed, falling back to standard tokenizer: {str(e)}")
            tokenizer = BertTokenizer.from_pretrained(MODEL_ID)
            
        model = BertForTokenClassification.from_pretrained(MODEL_ID)
        
        # Save to cache
        logger.info(f"Saving model to cache: {model_dir}")
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        
        # Create ID to tag mapping
        if hasattr(model.config, "id2label"):
            id_to_tag = model.config.id2label
            # Save mappings to cache
            with open(os.path.join(model_dir, "tag_mappings.json"), "w") as f:
                json.dump({"id_to_tag": id_to_tag}, f)
        else:
            num_labels = model.config.num_labels
            id_to_tag = {i: f"TAG_{i}" for i in range(num_labels)}
            
        model.eval()  # Set model to evaluation mode
        
        # Add memory optimizations after model is loaded
        model.config.torchscript = True
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Move model to CPU explicitly to control memory
        model = model.cpu()
        
        # Additional memory cleanup
        gc.collect()
        
        # Verify components were successfully loaded
        if tokenizer is None or model is None or id_to_tag is None:
            raise ValueError("One or more model components failed to initialize")
            
        logger.info(f"Model downloaded and cached in {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}", exc_info=True)
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
    """Perform startup tasks - preload the model if PRELOAD_MODEL is true"""
    if os.environ.get("PRELOAD_MODEL", "false").lower() == "true":
        try:
            initialize_model()
        except Exception as e:
            logger.error(f"Failed to preload model: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for AWS load balancers"""
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
        # Ensure model is loaded
        if model is None or tokenizer is None or id_to_tag is None:
            initialize_model()
            
        # Double-check that model loaded successfully
        if model is None or tokenizer is None or id_to_tag is None:
            raise RuntimeError("Failed to initialize model components")
            
        with log_performance("text_tokenization"):
            # Use predict_entities from run_inference.py
            entity_dicts = predict_entities(text, tokenizer, model, id_to_tag)
            
            # Log the found entities for debugging
            logger.info(f"Found {len(entity_dicts)} entities in text of length {len(text)}")
            if entity_dicts:
                for entity in entity_dicts:
                    logger.info(f"Entity: {entity['text']} - {entity['label']}")
        
        log_prediction(len(text), entity_dicts, "bert-base")
        
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
        
        return PredictionResponse(
            entities=entities,
            original_text=text,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")