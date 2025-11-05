import io
import os
import shutil
from PIL import Image, ImageDraw
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import cv2
import numpy as np
import torch
from tqdm import tqdm
import re
# Local imports
import models
import database
from database import engine

# --- AI Model & Processor Setup ---
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

# Path to your new, smarter YOLOv8 model
YOLO_MODEL_PATH = 'best.pt' 

models.Base.metadata.create_all(bind=engine)

# --- File Storage Setup ---
TICKETS_DIR = "tickets"
TEMP_LINES_DIR = "temp_lines"
DEBUG_DIR = "debug_output"
os.makedirs(TICKETS_DIR, exist_ok=True)
os.makedirs(TEMP_LINES_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

app = FastAPI(title="Advanced Handwritten Scanner API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(f"/{TICKETS_DIR}", StaticFiles(directory=TICKETS_DIR), name="tickets")
app.mount(f"/{DEBUG_DIR}", StaticFiles(directory=DEBUG_DIR), name="debug")

# --- Security and Authentication ---
SECRET_KEY = "a_very_secret_key_change_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- AI Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading Hugging Face TrOCR model...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten') 
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(device)
print("✅ TrOCR model loaded successfully.")

print("Loading custom YOLOv8 model for table cell detection...")
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"❌ CRITICAL ERROR: YOLO model not found at '{YOLO_MODEL_PATH}'")
    yolo_model = None
else:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ Custom YOLOv8 model loaded successfully.")


# --------------------------------------------------------
# --- Currency Symbol Fix ---
# --------------------------------------------------------
def correct_currency_symbols(text: str) -> str:
    """
    Corrects common OCR misinterpretations of currency symbols like:
    - S10, s.50, s0.00, so.00 → $10, $.50, $0.00, $0.00
    """
    # 1️⃣ Replace 's' or 'S' (possibly followed by o or 0) before digits or dots with $
    text = re.sub(r'\b[sS][oO0]?(?=\s?[\d.])', '$', text)

    # 2️⃣ Replace 'o' or 'O' inside numbers with '0'
    text = re.sub(r'(?<=\d)[oO](?=\d)', '0', text)

    return text


# ------------------------------------------------------------------- #
# --- IMAGE PREPROCESSING (UNCHANGED) ---
# ------------------------------------------------------------------- #

def basic_preprocess(image: Image.Image) -> Image.Image:
    open_cv_image = np.array(image.convert("RGB"))
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binarized = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if np.mean(binarized) < 128:
        binarized = cv2.bitwise_not(binarized)
    final_image_rgb = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_image_rgb)

# ------------------------------------------------------------------- #
# --- HORIZONTAL & VERTICAL LINE REMOVAL (UNCHANGED) ---
# ------------------------------------------------------------------- #

def remove_lines(image: Image.Image) -> Image.Image:
    binarized = np.array(image.convert("L"))
    inverted_binarized = cv2.bitwise_not(binarized)

    # --- Detect horizontal lines ---
    h_kernel_width = max(50, binarized.shape[1] // 30)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))
    detected_horizontal_lines = cv2.morphologyEx(
        inverted_binarized, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    dilated_horizontal_lines = cv2.dilate(
        detected_horizontal_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )

    # --- Detect vertical lines ---
    v_kernel_height = max(50, binarized.shape[0] // 30)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_height))
    detected_vertical_lines = cv2.morphologyEx(
        inverted_binarized, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )
    dilated_vertical_lines = cv2.dilate(
        detected_vertical_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )

    all_lines_mask = cv2.add(dilated_horizontal_lines, dilated_vertical_lines)
    cleaned_inverted = cv2.subtract(inverted_binarized, all_lines_mask)
    final_binarized = cv2.bitwise_not(cleaned_inverted)
    final_image_rgb = cv2.cvtColor(final_binarized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_image_rgb)

# ------------------------------------------------------------------- #
# --- TABLE DETECTION & OCR (UNCHANGED) ---
# ------------------------------------------------------------------- #

def enhance_cell_image(cell_cv_image):
    if cell_cv_image.shape[0] < 10 or cell_cv_image.shape[1] < 10:
        return None
    if len(cell_cv_image.shape) == 3 and cell_cv_image.shape[2] == 3:
        gray = cv2.cvtColor(cell_cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_cv_image
    target_height = 64
    aspect_ratio = target_height / gray.shape[0]
    new_width = int(gray.shape[1] * aspect_ratio)
    interp = cv2.INTER_CUBIC if aspect_ratio > 1 else cv2.INTER_AREA
    resized = cv2.resize(gray, (new_width, target_height), interpolation=interp)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    contrasted = clahe.apply(resized)
    _, binarized_image = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binarized_image) < 128:
        binarized_image = cv2.bitwise_not(binarized_image)
    num_black_pixels = np.sum(binarized_image == 0)
    total_pixels = binarized_image.shape[0] * binarized_image.shape[1]
    if total_pixels == 0:
        return None
    text_percentage = num_black_pixels / total_pixels
    if text_percentage < 0.015: 
        return None
    final_image_rgb = cv2.cvtColor(binarized_image, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_image_rgb)

def recognize_cell_text(cell_image: Image.Image):
    if cell_image is None or cell_image.width < 5 or cell_image.height < 5:
        return ""
    try:
        pixel_values = processor(images=cell_image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=300)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""

def get_y_overlap(box1, box2):
    b1_top, b1_bottom = box1[1], box1[3]
    b2_top, b2_bottom = box2[1], box2[3]
    overlap_top = max(b1_top, b2_top)
    overlap_bottom = min(b1_bottom, b2_bottom)
    overlap_height = max(0, overlap_bottom - overlap_top)
    if overlap_height == 0:
        return 0
    min_height = min(b1_bottom - b1_top, b2_bottom - b2_top)
    if min_height == 0:
        return 0
    return overlap_height / min_height

def extract_table_data_yolo(image: Image.Image, debug_dir_path: str):
    print("Running primary table extraction with YOLO (on line-included image)...")
    if yolo_model is None:
        print("⚠️ YOLO model is not loaded. Skipping table detection.")
        return None
    original_image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = original_image_cv.shape 
    cv2.imwrite(os.path.join(debug_dir_path, "1_image_for_yolo_detection.png"), original_image_cv)
    print("Detecting table cells...")
    results = yolo_model.predict(original_image_cv, conf=0.9, verbose=False)
    if not results or results[0].boxes is None or not results[0].boxes.xyxy.nelement():
        print("No cells detected by YOLO.")
        return None
    cell_boxes = results[0].boxes.cpu().numpy().xyxy.astype(int).tolist()
    if not cell_boxes:
        print("No cell boxes in results.")
        return None
    min_x = min(b[0] for b in cell_boxes)
    min_y = min(b[1] for b in cell_boxes)
    max_x = max(b[2] for b in cell_boxes)
    max_y = max(b[3] for b in cell_boxes)
    padding = 10 
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(w, max_x + padding)
    max_y = min(h, max_y + padding)
    table_bbox = [min_x, min_y, max_x, max_y]
    print(f"Detected {len(cell_boxes)} cells. Reconstructing table structure...")
    rows = []
    processed_indices = set()
    box_list_with_indices = sorted(enumerate(cell_boxes), key=lambda item: item[1][1])
    for i, box in box_list_with_indices:
        if i in processed_indices:
            continue
        current_row = [box]
        processed_indices.add(i)
        for j, other_box in box_list_with_indices:
            if j in processed_indices:
                continue
            if get_y_overlap(box, other_box) > 0.5:
                current_row.append(other_box)
                processed_indices.add(j)
        current_row.sort(key=lambda b: b[0])
        rows.append(current_row)
    print("Removing ALL lines from table area for clean OCR...")
    cleaned_image_pil = remove_lines(image)
    cleaned_image_cv = cv2.cvtColor(np.array(cleaned_image_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(debug_dir_path, "2_image_for_cell_extraction_CLEANED.png"), cleaned_image_cv)
    table_data = []
    print("Enhancing and performing OCR on detected cells (from cleaned image)...")
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    for i, row_boxes in enumerate(tqdm(rows, desc="Reading Table Rows")):
        row_text = []
        for j, box in enumerate(row_boxes):
            x1, y1, x2, y2 = box
            cell_padding = 2
            cell_image_cv = cleaned_image_cv[
                max(0, y1 - cell_padding):min(cleaned_image_cv.shape[0], y2 + cell_padding),
                max(0, x1 - cell_padding):min(cleaned_image_cv.shape[1], x2 + cell_padding)
            ]
            enhanced_cell_pil = enhance_cell_image(cell_image_cv)
            if enhanced_cell_pil:
                enhanced_cell_pil.save(os.path.join(debug_dir_path, f"cell_{i:02d}_{j:02d}.png"))
            draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
            raw_text = recognize_cell_text(enhanced_cell_pil)
            row_text.append(raw_text)
        table_data.append(row_text)
    draw_img.save(os.path.join(debug_dir_path, "3_detected_cells_on_original.png"))
    return {
        "extracted_table": table_data,
        "table_bbox": table_bbox,
        "debug_output_path": debug_dir_path
    }

# ------------------------------------------------------------------- #
# --- CELL SEGMENTATION (UNCHANGED) ---
# ------------------------------------------------------------------- #

def extract_lines_data(image_path: str, unique_filename: str):
    scan_temp_dir = os.path.join(TEMP_LINES_DIR, unique_filename)
    os.makedirs(scan_temp_dir, exist_ok=True)
    try:
        print("Segmenting non-table text...")
        cell_data_by_row = segment_lines(image_path, scan_temp_dir)
        if not cell_data_by_row: 
            print("No non-table text found after segmentation.")
            return None
        total_cells = sum(len(row["paths"]) for row in cell_data_by_row)
        print(f"Contour OCR started on {len(cell_data_by_row)} lines ({total_cells} cells)...")
        all_extracted_lines = []
        with tqdm(total=total_cells, desc="Reading Cells (Contour)") as pbar:
            for row_data in cell_data_by_row:
                row_text = []
                for cell_path in row_data["paths"]:
                    text = recognize_line(cell_path)
                    row_text.append(text)
                    pbar.update(1)
                all_extracted_lines.append({
                    "row_text": row_text,
                    "y": row_data["y"]
                })
        return {"all_lines": all_extracted_lines}
    finally:
        if os.path.exists(scan_temp_dir): 
            shutil.rmtree(scan_temp_dir)

def segment_lines(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None: 
        print("Segment_lines: Could not read image.")
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary_otsu) < 128:
        binary_otsu = cv2.bitwise_not(binary_otsu)
    binary = cv2.bitwise_not(binary_otsu)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        print("Segment_lines: No contours found.")
        return []
    word_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 15]
    if not word_boxes: 
        print("Segment_lines: No word boxes found after filtering contours (area < 15).")
        return []
    word_boxes.sort(key=lambda b: b[1])
    lines_data = []
    current_line = []
    if word_boxes:
        current_line.append(word_boxes[0])
        sample_heights = [h for _, _, _, h in word_boxes[:50]]
        avg_height = np.mean(sample_heights) if sample_heights else 20
        for box in word_boxes[1:]:
            last_box_y_center = current_line[-1][1] + current_line[-1][3] / 2
            current_box_y_center = box[1] + box[3] / 2
            if abs(current_box_y_center - last_box_y_center) < avg_height * 1.0:
                current_line.append(box)
            else:
                sorted_line_boxes = sorted(current_line, key=lambda b: b[0])
                lines_data.append({"boxes": sorted_line_boxes, "y": sorted_line_boxes[0][1]})
                current_line = [box]
        if current_line:
            sorted_line_boxes = sorted(current_line, key=lambda b: b[0])
            lines_data.append({"boxes": sorted_line_boxes, "y": sorted_line_boxes[0][1]})
    all_lines_data = [] 
    all_widths = []
    for _, _, w, h in word_boxes:
        if h > avg_height * 0.5:
            all_widths.append(w)
    avg_char_width = np.mean(all_widths) if all_widths else 10
    gap_threshold = avg_char_width * 2.0 
    for line_data in lines_data:
        line = line_data["boxes"]
        line_y = line_data["y"]
        if not line: continue
        cells_in_line = [] 
        current_cell_boxes = [line[0]] 
        for i in range(len(line) - 1):
            current_word_box = line[i]
            next_word_box = line[i+1]
            gap = next_word_box[0] - (current_word_box[0] + current_word_box[2])
            if gap > gap_threshold:
                cells_in_line.append(current_cell_boxes)
                current_cell_boxes = [next_word_box]
            else:
                current_cell_boxes.append(next_word_box)
        cells_in_line.append(current_cell_boxes)
        all_lines_data.append({"line_cell_boxes": cells_in_line, "y": line_y})
    return crop_and_save_cells(image, all_lines_data, output_dir)

def crop_and_save_cells(image, all_lines_data, output_dir):
    final_lines = []
    padding = 10
    for i, line_data in enumerate(all_lines_data):
        cell_paths_in_row = []
        line_y = line_data["y"]
        for j, cell_boxes in enumerate(line_data["line_cell_boxes"]):
            if not cell_boxes: continue
            x_min = min(b[0] for b in cell_boxes)
            y_min = min(b[1] for b in cell_boxes)
            x_max = max(b[0] + b[2] for b in cell_boxes)
            y_max = max(b[1] + b[3] for b in cell_boxes)
            y1, y2 = max(0, y_min - padding), min(image.shape[0], y_max + padding)
            x1, x2 = max(0, x_min - padding), min(image.shape[1], x_max + padding)
            cell_img = image[y1:y2, x1:x2]
            cell_path = os.path.join(output_dir, f"row_{i:02d}_cell_{j:02d}.png")
            if cell_img.size > 0 and cv2.imwrite(cell_path, cell_img):
                cell_paths_in_row.append(cell_path)
        if cell_paths_in_row:
            final_lines.append({"paths": cell_paths_in_row, "y": line_y})
    return final_lines

def recognize_line(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_cv_gray = np.array(image.convert("L"))
        _, binarized = cv2.threshold(image_cv_gray, 200, 255, cv2.THRESH_BINARY)
        num_black_pixels = np.sum(binarized == 0)
        total_pixels = binarized.shape[0] * binarized.shape[1]
        if total_pixels == 0:
            return ""
        text_percentage = num_black_pixels / total_pixels
        if text_percentage < 0.015:
            return ""
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=100)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception: 
        return ""

# ------------------------------------------------------------------- #
# --- *** UPDATED: STRUCTURED DATA EXTRACTOR *** ---
# ------------------------------------------------------------------- #

def extract_structured_data(raw_text: str) -> dict:
    print("Extracting structured data from raw text...")
    
    def clean_value(value: str):
        if not value:
            return None
        return value.strip(" :-\n\t").strip()

    # --- *** STEP 1: (Fast Path) Find values on the SAME LINE *** ---
    # This is where the bug was. The value regex for material/haul_vendor was bad.
    patterns = {
        "ticket_number": r'(?i)(?:Ticket Number|Ticket#|TICKET NO|Ticket #|Inovice #|Invoice#)\s*[:\-]?\s*([A-Za-z0-9\-\s]+)',
        "ticket_date":   r'(?i)(?:Date)\s*[:\-]?\s*([\d\/\-]{6,10})', 
        "haul_vendor":   r'(?i)(?:Haul Vendor|Vendor|Broker|Trucker|Customer)\s*[:\-]?\s*([A-Za-z&][A-Za-z\s&]*)', # <-- FIXED
        "truck_number":  r'(?i)(?:Truck Number|Truck No|Truck #)\s*[:\-]?\s*([A-Za-z0-9\-]+)',
        "material":      r'(?i)(?:Material(?:\s+hauled)?)\s*[:\-]?\s*([A-Za-z\d\-][A-Za-z\s\d\-]*)', # <-- FIXED
        "job_number":    r'(?i)(?:Job Number|Job No|Job #)\s*[:\-]?\s*([A-Za-z0-9\-]+)',
        "phase_code":    r'(?i)(?:Phase Code)\s*[:\-]?\s*([A-Za-z0-9\-]+)',
        "zone":          r'(?i)(?:Zone)\s*[:\-]?\s*([A-Za-z0-9\-]+)',
        "hours":         r'(?i)(?:Hours)\s*[:\-]?\s*([\d\.]+(?:\s?hrs)?)'
    }
    
    results = {
        "ticket_number": None,
        "ticket_date": None,
        "haul_vendor": None,
        "truck_number": None,
        "material": None,
        "job_number": None,
        "phase_code": None,
        "zone": None,
        "hours": None,
    }

    # Run the "fast path" loop first
    for field, pattern in patterns.items():
        match = re.search(pattern, raw_text)
        if match:
            value = clean_value(match.group(1))
            results[field] = value
            print(f"✅ Found {field} (same-line): {value}")

    # --- *** STEP 2: (Next-Line/Cell Logic) *** ---
    # This also has the fixes for material/haul_vendor
    multi_find_patterns = {
        "ticket_number": (r'(?i)(?:Ticket Number|Ticket#|TICKET NO|Ticket #|Inovice #|Invoice#|Ticket # )', r'([A-Za-z0-9\-\s]+)'),
        "ticket_date":   (r'(?i)(?:Date)', r'([\d\/\-]{6,10})'),
        "haul_vendor":   (r'(?i)(?:Haul Vendor|Vendor|Broker|Trucker|Customer)', r'([A-Za-z&][A-Za-z\s&]*)'), # <-- FIXED
        "truck_number":  (r'(?i)(?:Truck Number|Truck No|Truck #)', r'([A-Za-z0-9\-]+)'),
        "material":      (r'(?i)(?:Material(?:\s+hauled)?)', r'([A-Za-z\d\-][A-Za-z\s\d\-]*)'), # <-- FIXED
        "job_number":    (r'(?i)(?:Job Number|Job No|Job #)', r'([A-Za-z0-9\-]+)'),
        "phase_code":    (r'(?i)(?:Phase Code)', r'([A-Za-z0-9\-]+)'),
        "zone":          (r'(?i)(?:Zone)', r'([A-Za-z0-9\-]+)'),
        "hours":         (r'(?i)(?:Hours)', r'([\d\.]+(?:\s?hrs)?)')
    }

    rows = raw_text.split('\n')
    
    for field, (key_pattern, value_pattern) in multi_find_patterns.items():
        # Only run this search if Step 1 failed (result is still None)
        if results[field] is None:
            try:
                key_re = re.compile(key_pattern)
                value_re = re.compile(value_pattern)

                for i, row in enumerate(rows):
                    # --- *** Next-Cell Logic *** ---
                    cells = row.split('|')
                    for j, cell in enumerate(cells):
                        if key_re.search(cell):
                            # Key found in this cell.
                            # Check if value is in the SAME cell (but wasn't caught by Step 1)
                            # We search the part *after* the key
                            search_area = cell[key_re.search(cell).end():]
                            same_cell_value_match = value_re.search(search_area)
                            
                            if same_cell_value_match:
                                value = clean_value(same_cell_value_match.group(1))
                                if value:
                                    results[field] = value
                                    print(f"✅ Found {field} (same-cell): {value}")
                                    break # Found it, move to next field
                            
                            # If not in same cell, check NEXT cell
                            elif (j + 1) < len(cells):
                                next_cell = cells[j + 1]
                                value_match = value_re.search(next_cell)
                                if value_match:
                                    value = clean_value(value_match.group(1))
                                    if value:
                                        results[field] = value
                                        print(f"✅ Found {field} (next-cell): {value}")
                                        break # Found it, move to next field
                    
                    if results[field] is not None:
                        break # Value was found in a cell, stop searching this field

                    # --- *** Next-Line Logic *** ---
                    if results[field] is None and key_re.search(row):
                        # Key found. Check the NEXT line.
                        if (i + 1) < len(rows):
                            next_row = rows[i + 1]
                            value_match = value_re.search(next_row)
                            if value_match:
                                value = clean_value(value_match.group(1))
                                if value:
                                    results[field] = value
                                    print(f"✅ Found {field} (next-line): {value}")
                                    break # Found it, move to the next field
            
            except Exception as e:
                print(f"⚠️ Error during multi-find search for {field}: {e}")

    # --- *** STEP 3: (Fallback Path) Find values with NO KEY *** ---
    fallback_patterns = {
        "ticket_date": r'(\d{1,2}\/\d{1,2}\/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})'
    }

    for field, pattern in fallback_patterns.items():
        if results[field] is None:
            match = re.search(pattern, raw_text)
            if match:
                value = clean_value(match.group(1)) # Group(1) is the value
                results[field] = value
                print(f"✅ Found {field} (fallback): {value}")
                
    if results["haul_vendor"]:
        results["haul_vendor"] = results["haul_vendor"].split('\n')[0].strip()

    if results["hours"]:
        if isinstance(results["hours"], str):
             results["hours"] = re.sub(r'[^0-9\.]', '', results["hours"])
        try:
            results["hours"] = float(results["hours"])
        except (ValueError, TypeError):
            results["hours"] = None

    print(f"Structured data results: {results}")
    return results


# ------------------------------------------------------------------- #
# --- AUTHENTICATION (UNCHANGED) ---
# ------------------------------------------------------------------- #
def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def get_password_hash(password): return pwd_context.hash(password)
def create_access_token(data: dict):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({**data, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username: raise exc
    except JWTError: raise exc
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user: raise exc
    return user

# ------------------------------------------------------------------- #
# --- API ENDPOINTS (UNCHANGED) ---
# ------------------------------------------------------------------- #
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(form: OAuth2PasswordRequestForm=Depends(), db: Session=Depends(database.get_db)):
    if db.query(models.User).filter(models.User.username == form.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    new_user = models.User(username=form.username, hashed_password=get_password_hash(form.password))
    db.add(new_user); db.commit(); db.refresh(new_user)
    return {"message": "User registered successfully"}

@app.post("/token")
def login(form: OAuth2PasswordRequestForm=Depends(), db: Session=Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    return {
        "access_token": create_access_token({"sub": user.username}),
        "token_type": "bearer",
        "user_id": user.id
    }

# ------------------------------------------------------------------- #
# --- /scan ENDPOINT (UNCHANGED) ---
# ------------------------------------------------------------------- #
@app.post("/scan")
async def scan_ticket(file: UploadFile=File(...), current_user: models.User=Depends(get_current_user), db: Session=Depends(database.get_db)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{current_user.id}_{os.path.basename(file.filename)}"
    saved_image_path = os.path.join(TICKETS_DIR, unique_filename)
    debug_scan_dir = os.path.join(DEBUG_DIR, unique_filename)
    os.makedirs(debug_scan_dir, exist_ok=True)

    try:
        file_content = await file.read()
        
        original_image_pil = Image.open(io.BytesIO(file_content)).convert("RGB")
        print("Preprocessing image (basic)...")
        basic_processed_pil = basic_preprocess(original_image_pil)
        print("✅ Basic preprocessing complete.")
        
        basic_processed_pil.save(saved_image_path, format='PNG')
            
        table_result = extract_table_data_yolo(basic_processed_pil, debug_scan_dir)
        
        image_for_contours = basic_processed_pil.copy()
        table_data = None
        table_y_start = float('inf')

        if table_result:
            print("✅ Table found via YOLO! Will combine with other text.")
            table_data = table_result["extracted_table"]
            table_bbox = table_result["table_bbox"]
            table_y_start = table_bbox[1]
            draw = ImageDraw.Draw(image_for_contours)
            draw.rectangle(table_bbox, fill="white")
            image_for_contours.save(os.path.join(debug_scan_dir, "4_erased_table_from_basic.png"))
        else:
            print("⚠️ No table found via YOLO. Processing full page with contour-based segmentation.")

        print("Applying horizontal & vertical line removal to non-table areas...")
        image_for_contours_cleaned = remove_lines(image_for_contours)
        print("✅ Line removal complete for non-table areas.")
        
        temp_contour_path = os.path.join(debug_scan_dir, "temp_for_contours_CLEANED.png")
        image_for_contours_cleaned.save(temp_contour_path, format='PNG')
        
        print("--- Calling extract_lines_data (for non-table text) ---")
        line_result = extract_lines_data(temp_contour_path, unique_filename)
        print("--- Returned from extract_lines_data ---")
        
        contour_lines = []
        if not line_result or not line_result.get("all_lines"):
            if table_data:
                print("⚠️ Contour extraction failed, but YOLO found a table. Saving table data only.")
            else:
                raise HTTPException(status_code=400, detail="Could not detect any text in the image.")
        else:
            print(f"✅ Contour extraction found {len(line_result['all_lines'])} lines of text.")
            contour_lines = line_result["all_lines"]

        final_data_rows = []
        table_inserted = False

        for line in contour_lines:
            if not table_inserted and table_data and line["y"] >= table_y_start:
                final_data_rows.extend(table_data)
                table_inserted = True
            final_data_rows.append(line["row_text"])

        if not table_inserted and table_data:
            final_data_rows.extend(table_data)

        if not final_data_rows:
            raise HTTPException(status_code=400, detail="Text extraction resulted in empty content.")

        json_output_rows = []
        for row in final_data_rows:
            corrected_row = [correct_currency_symbols(str(cell)) for cell in row]
            json_output_rows.append(corrected_row)

        db_text_blob = "\n".join([" | ".join(row) for row in json_output_rows])
        
        structured_data = extract_structured_data(db_text_blob)
        
        response_data = {
            "extracted_text_rows": json_output_rows,
            "structured_data": structured_data
        }
        
        image_url_path = f"/{TICKETS_DIR}/{unique_filename}"
        
        new_ticket = models.Ticket(
            raw_text_content=db_text_blob,
            owner_id=current_user.id,
            image_path=image_url_path,
            ticket_number=structured_data.get("ticket_number"),
            ticket_date=structured_data.get("ticket_date"),
            haul_vendor=structured_data.get("haul_vendor"),
            truck_number=structured_data.get("truck_number"),
            material=structured_data.get("material"),
            job_number=structured_data.get("job_number"),
            phase_code=structured_data.get("phase_code"),
            zone=structured_data.get("zone"),
            hours=structured_data.get("hours")
        )
        
        db.add(new_ticket); db.commit(); db.refresh(new_ticket)
        
        response_data["image_url"] = image_url_path
        response_data["ticket_id"] = new_ticket.id
        
        return {"filename": file.filename, **response_data}

    except Exception as e:
        if isinstance(e, HTTPException): raise e
        import traceback
        print("--- UNEXPECTED ERROR TRACEBACK ---")
        traceback.print_exc()
        print("-----------------------------------")
        print(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if os.path.exists(debug_scan_dir):
            try:
                pass # Keep debug dir for inspection
                shutil.rmtree(debug_scan_dir) # Uncomment to clean up
            except OSError as e:
                print(f"Error removing debug directory {debug_scan_dir}: {e.strerror}")


# ------------------------------------------------------------------- #
# --- /update-ticket-text ENDPOINT (UNCHANGED) ---
# ------------------------------------------------------------------- #
@app.post("/update-ticket-text")
def update_ticket_text(
    request: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    print(f"🔄 Update ticket request received: {request}")
    
    if "ticket_id" not in request or "raw_text" not in request:
        raise HTTPException(status_code=400, detail="ticket_id and raw_text are required")
        
    ticket_id = request["ticket_id"]
    new_raw_text = request["raw_text"]
    
    ticket = db.query(models.Ticket).filter(
        models.Ticket.id == ticket_id,
        models.Ticket.owner_id == current_user.id
    ).first()
    
    if not ticket:
        print(f"❌ Ticket {ticket_id} not found for user {current_user.id}")
        raise HTTPException(status_code=404, detail="Ticket not found")
        
    print(f"📝 Updating ticket {ticket_id}...")
    
    ticket.raw_text_content = new_raw_text
    new_structured_data = extract_structured_data(new_raw_text)
    
    ticket.ticket_number = new_structured_data.get("ticket_number")
    ticket.ticket_date = new_structured_data.get("ticket_date")
    ticket.haul_vendor = new_structured_data.get("haul_vendor")
    ticket.truck_number = new_structured_data.get("truck_number")
    ticket.material = new_structured_data.get("material")
    ticket.job_number = new_structured_data.get("job_number")
    ticket.phase_code = new_structured_data.get("phase_code")
    ticket.zone = new_structured_data.get("zone")
    ticket.hours = new_structured_data.get("hours")
    
    db.commit()
    db.refresh(ticket)
    
    response_data = {
        "message": "Ticket updated successfully",
        "ticket": {
            "id": ticket.id,
            "image_url": ticket.image_path,
            "raw_text_content": ticket.raw_text_content,
            "ticket_number": ticket.ticket_number,
            "ticket_date": ticket.ticket_date,
            "haul_vendor": ticket.haul_vendor,
            "truck_number": ticket.truck_number,
            "material": ticket.material,
            "job_number": ticket.job_number,
            "phase_code": ticket.phase_code,
            "zone": ticket.zone,
            "hours": ticket.hours
        }
    }
    
    print(f"✅ Update successful for ticket {ticket_id}")
    return response_data

# ------------------------------------------------------------------- #
# --- /tickets ENDPOINT (UNCHANGED) ---
# ------------------------------------------------------------------- #
@app.get("/tickets")
def get_tickets(current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    tickets_from_db = db.query(models.Ticket).filter(
        models.Ticket.owner_id == current_user.id
    ).order_by(models.Ticket.created_at.desc()).all()
    
    response = []
    
    for ticket in tickets_from_db:
        response.append({
            "id": ticket.id,
            "image_url": ticket.image_path,
            "created_at": ticket.created_at,
            "raw_text_content": ticket.raw_text_content,
            "ticket_number": ticket.ticket_number,
            "ticket_date": ticket.ticket_date,
            "haul_vendor": ticket.haul_vendor,
            "truck_number": ticket.truck_number,
            "material": ticket.material,
            "job_number": ticket.job_number,
            "phase_code": ticket.phase_code,
            "zone": ticket.zone,
            "hours": ticket.hours
        })
        
    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to the Advanced Handwritten Scanner API"}

# Example for running with uvicorn (if you run this file directly)
if __name__ == "__main__":
    import uvicorn
    print("--- Starting FastAPI Server ---")
    print(f"YOLO Model: {YOLO_MODEL_PATH} ({'LOADED' if yolo_model else 'NOT FOUND'})")
    print(f"TrOCR Model: microsoft/trocr-large-handwritten (LOADED)")
    print(f"Using Device: {device}")
    print("---------------------------------")
    uvicorn.run(app, host="0.0.0.0", port=8000)



    #Extracting separate fields from image