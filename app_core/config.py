# --- Configuration ---
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.7 # Overall confidence for "ok" status
YOLO_CONF_FOR_OCR = 0.4  # Minimum YOLO confidence to attempt OCR
LP_FORMAT_REGEX = r"^[A-Z]{3}-[0-9]{4}$"

# Image Preprocessing for OCR
OCR_FIXED_WIDTH = 256
OCR_FIXED_HEIGHT = 64
DEBUG_SAVE_OCR_IMAGES = True
# DEBUG_SHOW_OCR_IMAGE = False # REMOVED

DEBUG_IMG_DIR = "debug_ocr_images"
PROCESSED_GRAY_PLATES_DIR = "processed_gray_plates"
PROCESSED_WARPED_COLOR_PLATES_DIR = "processed_warped_color_plates" 