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
import psutil
import traceback

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

def log_memory_usage(label):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage ({label}): {memory_info.rss / 1024 / 1024:.2f} MB")

def initialize_model():
    """Load model with aggressive memory optimizations"""
    global tokenizer, model, id_to_tag
    
    if tokenizer is not None and model is not None:
        return
    
    # Aggressive garbage collection before loading model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log_memory_usage("before_model_load")
    
    # Create cache directory if it doesn't exist
    model_dir = os.path.join(MODEL_CACHE_DIR, MODEL_ID.replace("/", "_"))
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if model exists in cache
    model_exists = os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
    tokenizer_exists = os.path.exists(os.path.join(model_dir, "vocab.txt"))
    
    try:
        # Load with more aggressive optimization
        if model_exists and tokenizer_exists:
            logger.info(f"Loading model from local cache: {model_dir}")
            start_time = time.time()
            
            # Load in parts to reduce memory pressure
            logger.info("Loading tokenizer...")
            tokenizer = load_model_component(model_dir, "tokenizer")
            gc.collect()
            
            logger.info("Loading model...")
            model = load_model_component(model_dir, "model")
            gc.collect()
            
            # Load ID to tag mapping
            tag_file = os.path.join(model_dir, "tag_mappings.json")
            if os.path.exists(tag_file):
                with open(tag_file, "r") as f:
                    mappings = json.load(f)
                    id_to_tag = mappings.get("id_to_tag", {})
                    # Convert string keys to integers
                    id_to_tag = {int(k): v for k, v in id_to_tag.items()}
            else:
                raise ValueError("Tag mappings not found in cache")
                
            logger.info(f"Model loaded from cache in {time.time() - start_time:.2f} seconds")
        else:
            # Download from Hugging Face
            logger.info(f"Downloading model from Hugging Face: {MODEL_ID}")
            start_time = time.time()
            
            # Load in sequence to reduce peak memory usage
            logger.info("Loading tokenizer...")
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(MODEL_ID)
            tokenizer.save_pretrained(model_dir)
            gc.collect()
            
            logger.info("Loading model...")
            from transformers import BertForTokenClassification
            model = BertForTokenClassification.from_pretrained(MODEL_ID, 
                                                            torchscript=True,
                                                            return_dict=False)
            gc.collect()
            
            # Save model before optimizing to save memory during save
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
        
        # Optimize model regardless of load path
        logger.info("Optimizing model for inference...")
        model.eval()  # Set model to evaluation mode
        
        # Apply quantization for reduced memory usage
        logger.info("Applying quantization...")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Embedding}, dtype=torch.qint8
        )
        
        # Force model to CPU mode
        model = model.cpu()
        
        # Free up as much memory as possible
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        log_memory_usage("after_model_load")
        
    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}\n{traceback.format_exc()}")
        raise RuntimeError(f"Could not initialize model: {str(e)}")

def load_model_component(model_dir, component_type):
    """Load model component with minimal memory footprint"""
    try:
        if component_type == "tokenizer":
            from transformers import BertTokenizer
            return BertTokenizer.from_pretrained(model_dir)
        elif component_type == "model":
            from transformers import BertForTokenClassification
            return BertForTokenClassification.from_pretrained(model_dir, 
                                                           torchscript=True,
                                                           return_dict=False)
    except Exception as e:
        logger.error(f"Failed to load {component_type}: {str(e)}")
        raise

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
        log_memory_usage("before_prediction")
        
        # Ensure model is loaded
        if model is None or tokenizer is None or id_to_tag is None:
            initialize_model()
            
        with log_performance("text_tokenization"):
            # Process text in smaller chunks if very long
            max_length = 512
            entity_dicts = []
            
            if len(text) > max_length:
                # Process long text in overlapping chunks
                chunks = []
                for i in range(0, len(text), max_length - 50):
                    chunk = text[i:i + max_length]
                    chunks.append((i, chunk))
                
                for offset, chunk in chunks:
                    chunk_entities = predict_entities(chunk, tokenizer, model, id_to_tag)
                    # Adjust entity positions based on chunk offset
                    for entity in chunk_entities:
                        entity["start"] += offset
                        entity["end"] += offset
                    entity_dicts.extend(chunk_entities)
                    
                    # Clean up after each chunk
                    gc.collect()
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
        
        # Force garbage collection again
        gc.collect()
        
        return PredictionResponse(
            entities=entities,
            original_text=text,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")