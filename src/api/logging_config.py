import logging
import logging.handlers
import os
import json
import uuid
import time
import traceback
import sys
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

# Configuration from environment variables with defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "standard")  # standard, json
LOG_RETENTION = int(os.getenv("LOG_RETENTION_DAYS", "30"))
MAX_LOG_SIZE_MB = int(os.getenv("MAX_LOG_SIZE_MB", "10"))
LOGS_DIR = os.getenv("LOGS_DIR", "logs")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Map string log levels to actual log levels
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Custom JSON formatter for structured logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "path": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Add thread and process info in debug mode
        if record.levelno <= logging.DEBUG:
            log_data.update({
                "thread": record.threadName,
                "process": record.processName
            })
            
        # Add context attributes if they exist
        for attr in ["request_id", "user", "execution_time", "model_version", "entity_count"]:
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)
                
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "trace": traceback.format_exception(*record.exc_info)
            }
                
        return json.dumps(log_data)

# Configure logging with enhanced features
def setup_logger():
    log_level = LOG_LEVEL_MAP.get(LOG_LEVEL, logging.INFO)
    
    # Initialize the logger
    logger = logging.getLogger("ner_api")
    logger.setLevel(log_level)
    
    # Clear existing handlers to prevent duplicates on reinitialization
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handlers - one for general logs, one for errors
    general_log_file = f"{LOGS_DIR}/api.log"
    error_log_file = f"{LOGS_DIR}/api_errors.log"
    
    # Rotating file handler for general logs
    general_handler = logging.handlers.RotatingFileHandler(
        general_log_file,
        maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,  # MB to bytes
        backupCount=10
    )
    general_handler.setLevel(log_level)
    
    # Rotating file handler specifically for errors and above
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,  # MB to bytes
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    
    # Also log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatters
    if LOG_FORMAT.lower() == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(request_id)s] - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Set formatters
    general_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(general_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    # Add filter to attach request_id if not present
    class RequestContextFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, 'request_id'):
                record.request_id = 'no-request-id'
            return True
    
    logger.addFilter(RequestContextFilter())
    
    return logger

logger = setup_logger()

# Request context manager
class RequestContext:
    _request_id = None
    _user = None
    
    @classmethod
    def get_request_id(cls):
        if cls._request_id is None:
            cls._request_id = str(uuid.uuid4())
        return cls._request_id
    
    @classmethod
    def set_request_id(cls, request_id):
        cls._request_id = request_id
    
    @classmethod
    def clear_request_id(cls):
        cls._request_id = None
    
    @classmethod
    def set_user(cls, user):
        cls._user = user
        
    @classmethod
    def get_user(cls):
        return cls._user

# Log request details decorator
def log_request(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Generate request ID and set in context
        request_id = str(uuid.uuid4())
        RequestContext.set_request_id(request_id)
        
        # Start timing
        start_time = time.time()
        
        # Modified approach - don't try to extract request details from parameters
        # Just log basic information we know for sure
        logger.info(f"Request started: {func.__name__}", 
                   extra={'request_id': request_id})
        
        try:
            # Execute the actual function
            response = await func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log successful response
            logger.info(
                f"Request completed in {execution_time:.3f}s",
                extra={
                    'request_id': request_id,
                    'execution_time': execution_time,
                    'status_code': getattr(response, 'status_code', 0)
                }
            )
            
            return response
            
        except Exception as e:
            # Calculate execution time for error case
            execution_time = time.time() - start_time
            
            # Log the exception with details
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    'request_id': request_id,
                    'execution_time': execution_time
                },
                exc_info=True
            )
            
            # Re-raise the exception
            raise
        finally:
            # Clear the request context
            RequestContext.clear_request_id()
    
    return wrapper

# Performance monitoring context manager
@contextmanager
def log_performance(operation_name):
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        logger.debug(
            f"Performance: {operation_name} completed in {execution_time:.3f}s",
            extra={
                'request_id': RequestContext.get_request_id(),
                'operation': operation_name,
                'execution_time': execution_time
            }
        )

# Log model prediction details
def log_prediction(text_length, entities, model_version):
    logger.info(
        f"Prediction processed: {len(entities)} entities found in {text_length} chars",
        extra={
            'request_id': RequestContext.get_request_id(),
            'text_length': text_length,
            'entity_count': len(entities),
            'model_version': model_version,
            'entity_types': {e['label']: 1 for e in entities}
        }
    )

# Configure exception hook to log unhandled exceptions
def log_unhandled_exception(exc_type, exc_value, exc_traceback):
    logger.critical(
        f"Unhandled exception: {exc_type.__name__}: {exc_value}",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    
sys.excepthook = log_unhandled_exception