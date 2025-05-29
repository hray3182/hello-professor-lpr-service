import logging
import os
import torch
from fastapi import FastAPI, APIRouter
from ultralytics import YOLO
import easyocr

from .config import (MODEL_PATH, DEBUG_SAVE_OCR_IMAGES, DEBUG_IMG_DIR, 
                     PROCESSED_GRAY_PLATES_DIR, PROCESSED_WARPED_COLOR_PLATES_DIR)

# FastAPI App and Routers
app = FastAPI()
entry_router = APIRouter(prefix="/entry", tags=["Entry Stream"])
exit_router = APIRouter(prefix="/exit", tags=["Exit Stream"])

# Logger
logger = logging.getLogger("app") # Consistent with old main.py
# Basic logging configuration, can be expanded or managed by uvicorn
logging.basicConfig(level=logging.INFO)

# Global data structure for streams
streams_data = {}

# Device for models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device for deep learning models: {device}")

# Models
yolo_model = None
easyocr_reader_instance = None

def load_models():
    global yolo_model, easyocr_reader_instance
    logger.info("Loading YOLO model...")
    try:
        abs_model_path = os.path.abspath(MODEL_PATH)
        logger.info(f"Attempting to load YOLO model from: {MODEL_PATH} (Absolute: {abs_model_path})")
        if not os.path.exists(MODEL_PATH): # Check if file exists at relative path from CWD
            logger.error(f"YOLO model file NOT FOUND at relative path: {MODEL_PATH} (Resolved absolute: {abs_model_path})")
            yolo_model = None # Explicitly set to None
            # Continue to attempt EasyOCR loading, but log this critical failure.
        else:
            yolo_model = YOLO(MODEL_PATH)
            yolo_model.to(device)
            logger.info(f"Successfully loaded YOLO model from {MODEL_PATH} to {device}")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}", exc_info=True)
        yolo_model = None

    logger.info("Initializing EasyOCR reader...")
    try:
        easyocr_reader_instance = easyocr.Reader(['en'], gpu=(device == 'cuda'))
        logger.info("EasyOCR reader initialized successfully.")
    except Exception as e_easyocr_init:
        logger.error(f"Failed to initialize EasyOCR reader: {e_easyocr_init}", exc_info=True)
        easyocr_reader_instance = None

def create_debug_directories():
    logger.info("Creating debug directories if they don't exist...")
    if DEBUG_SAVE_OCR_IMAGES:
        if not os.path.exists(DEBUG_IMG_DIR):
            os.makedirs(DEBUG_IMG_DIR)
            logger.info(f"Created directory: {DEBUG_IMG_DIR}")
    if not os.path.exists(PROCESSED_GRAY_PLATES_DIR):
        os.makedirs(PROCESSED_GRAY_PLATES_DIR)
        logger.info(f"Created directory: {PROCESSED_GRAY_PLATES_DIR}")
    if not os.path.exists(PROCESSED_WARPED_COLOR_PLATES_DIR):
        os.makedirs(PROCESSED_WARPED_COLOR_PLATES_DIR)
        logger.info(f"Created directory: {PROCESSED_WARPED_COLOR_PLATES_DIR}")

# Initial setup calls when this module is imported
# load_models() # Models will be loaded via lifespan event in the new main.py
# create_debug_directories() # Directories will be created via lifespan or on first use if needed 