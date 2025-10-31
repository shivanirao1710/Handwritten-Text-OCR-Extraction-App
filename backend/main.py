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

# Path to your custom YOLOv8 model
YOLO_MODEL_PATH = 'bordered.pt'  # Custom YOLO model path

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
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')   # or use microsoft/trocr-large-handwritten 
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
    - 5o.00, 5O.00, 5 10 → $0.00, $10
    """

    # 1️⃣ Replace 's' or 'S' (possibly followed by o or 0) before digits or dots with $
    text = re.sub(r'\b[sS][oO0]?(?=\s?[\d.])', '$', text)

    # 2️⃣ Replace '5' at the start of a number block (common $ misread)
    # Example: '5o.00' or '5 10' → '$0.00' / '$10'
    text = re.sub(r'\b5\s?([0-9oO.]+)', 
                  lambda m: '$' + m.group(1).replace('o', '0').replace('O', '0'),
                  text)

    # 3️⃣ Replace 'o' or 'O' inside numbers with '0'
    text = re.sub(r'(?<=\d)[oO](?=\d)', '0', text)

    return text



# ------------------------------------------------------------------- #
# --- IMAGE PREPROCESSING (NOW SPLIT INTO TWO FUNCTIONS) ---
# ------------------------------------------------------------------- #

def basic_preprocess(image: Image.Image) -> Image.Image:
    """
    Converts input image to a standardized white-background,
    black-text format. DOES NOT remove lines.
    """
    open_cv_image = np.array(image.convert("RGB"))
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Global binarization
    _, binarized = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Global check: Standardize to BLACK text (0) on WHITE background (255)
    if np.mean(binarized) < 128:
        # It's white text on black bg, invert it
        binarized = cv2.bitwise_not(binarized)

    final_image_rgb = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_image_rgb)

def remove_lines(image: Image.Image) -> Image.Image:
    """
    Takes a pre-processed (black text on white bg) PIL image
    and removes horizontal and vertical lines.
    Returns a new line-removed PIL image.
    """
    # Convert PIL Image to OpenCV format (BGR)
    binarized = np.array(image.convert("L"))

    # We must invert it to (white text on black bg) for morphological operations.
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

    # --- Combine and remove lines ---
    all_lines_mask = cv2.add(dilated_horizontal_lines, dilated_vertical_lines)
    
    cleaned_inverted = cv2.subtract(inverted_binarized, all_lines_mask)
    
    # Invert back to black text on white background
    final_binarized = cv2.bitwise_not(cleaned_inverted)

    final_image_rgb = cv2.cvtColor(final_binarized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_image_rgb)


# ------------------------------------------------------------------- #
# --- TABLE DETECTION & OCR (MODIFIED) ---
# ------------------------------------------------------------------- #

def enhance_cell_image(cell_cv_image):
    """
    Enhances a single cell image and, crucially, standardizes it to
    be BLACK TEXT on a WHITE BACKGROUND, regardless of its original format.
    
    NOTE: This now runs on a cell that has *already* had lines removed.
    """
    if cell_cv_image.shape[0] < 10 or cell_cv_image.shape[1] < 10:
        return None
    
    if len(cell_cv_image.shape) == 3 and cell_cv_image.shape[2] == 3:
        gray = cv2.cvtColor(cell_cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_cv_image # Assume it's already grayscale

    # --- Resize and Contrast ---
    target_height = 64
    aspect_ratio = target_height / gray.shape[0]
    new_width = int(gray.shape[1] * aspect_ratio)
    
    interp = cv2.INTER_CUBIC if aspect_ratio > 1 else cv2.INTER_AREA
    resized = cv2.resize(gray, (new_width, target_height), interpolation=interp)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    contrasted = clahe.apply(resized)
    
    # --- Binarize and Standardize Background ---
    _, binarized_image = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(binarized_image) < 128:
        binarized_image = cv2.bitwise_not(binarized_image)

    final_image_rgb = cv2.cvtColor(binarized_image, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_image_rgb)


def recognize_cell_text(cell_image: Image.Image):
    """Runs TrOCR on a single enhanced cell PIL image."""
    if cell_image is None or cell_image.width < 5 or cell_image.height < 5:
        return ""
    try:
        pixel_values = processor(images=cell_image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=300)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def extract_table_data_yolo(image: Image.Image, debug_dir_path: str):
    """
    Runs YOLO to find table cells on the input image (which has lines).
    Then, it removes lines from the image to extract clean cell text.
    
    'image' is the PIL Image from basic_preprocess (lines INCLUDED).
    """
    print("Running primary table extraction with YOLO (on line-included image)...")
    if yolo_model is None:
        print("⚠️ YOLO model is not loaded. Skipping table detection.")
        return None

    # 1. Prepare image for YOLO (lines included)
    original_image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = original_image_cv.shape 
    
    cv2.imwrite(os.path.join(debug_dir_path, "1_image_for_yolo_detection.png"), original_image_cv)

    # 2. Run YOLO detection
    print("Detecting table cells...")
    results = yolo_model.predict(original_image_cv, conf=0.2, verbose=False)
    if not results or results[0].boxes is None or not results[0].boxes.xyxy.nelement():
        print("No cells detected by YOLO.")
        return None

    # 3. Get all cell boxes and determine the full table bounding box
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
    
    # 4. Sort cells into rows for extraction
    sorted_cell_boxes = sorted(cell_boxes, key=lambda b: (b[1], b[0]))
    print(f"Detected {len(sorted_cell_boxes)} cells. Reconstructing table structure...")
    
    rows = []
    current_row = []
    if sorted_cell_boxes:
        ref_y = sorted_cell_boxes[0][1]
        avg_cell_height = np.mean([b[3] - b[1] for b in sorted_cell_boxes])
        
        for box in sorted_cell_boxes:
            if box[1] > ref_y + avg_cell_height * 0.8:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
                ref_y = box[1]
            else:
                current_row.append(box)
        rows.append(sorted(current_row, key=lambda b: b[0]))

    # --- START: NEW LOGIC ---
    # 5. Create a line-removed version of the image *for cell extraction*
    print("Removing lines from table area for accurate cell OCR...")
    cleaned_pil_image = remove_lines(image) # 'image' is the input PIL
    
    # --- THIS IS THE FIX ---
    # Convert the 3-channel RGB PIL image to a 3-channel BGR OpenCV image
    cleaned_image_cv = cv2.cvtColor(np.array(cleaned_pil_image), cv2.COLOR_RGB2BGR)
    # --- END OF FIX ---
    
    cv2.imwrite(os.path.join(debug_dir_path, "2_image_for_cell_extraction.png"), cleaned_image_cv)
    # --- END: NEW LOGIC ---

    # 6. Perform OCR on each cell, cropping from the CLEANED image
    table_data = []
    print("Enhancing and performing OCR on detected cells (from cleaned image)...")
    draw_img = image.copy() # We still draw boxes on the original for debug
    draw = ImageDraw.Draw(draw_img)

    for i, row_boxes in enumerate(tqdm(rows, desc="Reading Table Rows")):
        row_text = []
        for j, box in enumerate(row_boxes):
            x1, y1, x2, y2 = box
            cell_padding = 2
            
            # ***CRITICAL CHANGE***: Crop from 'cleaned_image_cv', not 'original_image_cv'
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
    
    # 7. Return both the extracted data and the table's bounding box
    return {
        "extracted_table": table_data,
        "table_bbox": table_bbox,
        "debug_output_path": debug_dir_path
    }
# ------------------------------------------------------------------- #
# --- MODIFIED CELL SEGMENTATION (FALLBACK/HYBRID LOGIC) ---
# ------------------------------------------------------------------- #

def extract_lines_data(image_path: str, unique_filename: str):
    """
    Manages the contour-based process: segments image into cells, recognizes text
    in each, and returns structured data with Y-coordinates.
    
    NOTE: This function receives the path to the *fully cleaned* image
    (table erased AND lines removed).
    """
    scan_temp_dir = os.path.join(TEMP_LINES_DIR, unique_filename)
    os.makedirs(scan_temp_dir, exist_ok=True)
    try:
        cell_data_by_row = segment_lines(image_path, scan_temp_dir)
        if not cell_data_by_row: 
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
    """
    Segments an image into lines and then splits those lines into cells.
    Includes the fix to correctly group lines.
    """
    image = cv2.imread(image_path)
    if image is None: 
        return []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        print("Segment_lines: No contours found.")
        return []

    word_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 15]
    if not word_boxes: 
        print("Segment_lines: No word boxes found after filtering contours.")
        return []
            
    word_boxes.sort(key=lambda b: b[1]) # Sort by y-coordinate
    
    lines_data = [] # Will store {"boxes": [...], "y": ...}
    current_line = []
    if word_boxes:
        current_line.append(word_boxes[0])
        sample_heights = [h for _, _, _, h in word_boxes[:50]]
        avg_height = np.mean(sample_heights) if sample_heights else 20

        for box in word_boxes[1:]:
            last_box_y_center = current_line[-1][1] + current_line[-1][3] / 2
            current_box_y_center = box[1] + box[3] / 2
            
            # --- THIS IS THE FIX from the previous step ---
            # Increased the tolerance from 0.7 to 1.0 to better group
            # contours (like letters and 'i' dots) on the same line.
            if abs(current_box_y_center - last_box_y_center) < avg_height * 1.0:
            # --- END OF FIX ---
                current_line.append(box)
            else:
                sorted_line_boxes = sorted(current_line, key=lambda b: b[0])
                lines_data.append({"boxes": sorted_line_boxes, "y": sorted_line_boxes[0][1]})
                current_line = [box]
        
        if current_line:
            sorted_line_boxes = sorted(current_line, key=lambda b: b[0])
            lines_data.append({"boxes": sorted_line_boxes, "y": sorted_line_boxes[0][1]})
            
    all_lines_data = [] 
    
    all_widths = [w for _, _, w, h in word_boxes if h > avg_height * 0.5]
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
        
        cells_in_line.append(current_cell_boxes) # Add the last cell
        all_lines_data.append({"line_cell_boxes": cells_in_line, "y": line_y})
        
    return crop_and_save_cells(image, all_lines_data, output_dir)

def crop_and_save_cells(image, all_lines_data, output_dir):
    """
    Crops and saves each detected cell.
    """
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
    """Recognizes text from a single cropped image path."""
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=100)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception: 
        return ""

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
# --- API ENDPOINTS (MODIFIED) ---
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
        
        # 1. Load and perform BASIC preprocessing (binarize, standardize bg)
        original_image_pil = Image.open(io.BytesIO(file_content)).convert("RGB")
        print("Preprocessing image (basic)...")
        basic_processed_pil = basic_preprocess(original_image_pil)
        print("✅ Basic preprocessing complete.")
        
        # 2. Save the basic preprocessed image (this is the one we save to db path)
        basic_processed_pil.save(saved_image_path, format='PNG')
            
        # 3. --- HYBRID LOGIC ---
        # Run YOLO on the basic image (lines INCLUDED)
        table_result = extract_table_data_yolo(basic_processed_pil, debug_scan_dir)
        
        image_for_contours = basic_processed_pil.copy()
        table_data = None
        table_y_start = float('inf')

        if table_result:
            print("✅ Table found via YOLO! Will combine with other text.")
            table_data = table_result["extracted_table"]
            table_bbox = table_result["table_bbox"]
            table_y_start = table_bbox[1]
            
            # Erase table from the "basic" (lines-included) image
            draw = ImageDraw.Draw(image_for_contours)
            draw.rectangle(table_bbox, fill="white")
            image_for_contours.save(os.path.join(debug_scan_dir, "4_erased_table_from_basic.png"))
        else:
            print("⚠️ No table found via YOLO. Processing full page with contour-based segmentation.")

        # 4. NOW, remove lines from the non-table-area image
        print("Applying line removal to non-table areas...")
        image_for_contours_cleaned = remove_lines(image_for_contours)
        print("✅ Line removal complete for non-table areas.")
        
        # 5. Run contour-based extraction on the *fully cleaned* image
        temp_contour_path = os.path.join(debug_scan_dir, "temp_for_contours_CLEANED.png")
        image_for_contours_cleaned.save(temp_contour_path, format='PNG')
        
        line_result = extract_lines_data(temp_contour_path, unique_filename)
        
        contour_lines = []
        if not line_result or not line_result.get("all_lines"):
            if table_data:
                print("⚠️ Contour extraction failed, but YOLO found a table. Saving table data only.")
            else:
                raise HTTPException(status_code=400, detail="Could not detect any text in the image.")
        else:
            print(f"✅ Contour extraction found {len(line_result['all_lines'])} lines of text.")
            contour_lines = line_result["all_lines"]

        # 6. Combine the results in the correct order
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

        # 7. Format for JSON response AND database
        json_output_rows = []
        for row in final_data_rows:
            corrected_row = [correct_currency_symbols(str(cell)) for cell in row]
            json_output_rows.append(corrected_row)

        db_text = "\n".join([" | ".join(row) for row in json_output_rows])
        response_data = {"extracted_text": json_output_rows}
        
        image_url_path = f"/{TICKETS_DIR}/{unique_filename}"
        new_ticket = models.Ticket(extracted_text=db_text, owner_id=current_user.id, image_path=image_url_path)
        db.add(new_ticket); db.commit(); db.refresh(new_ticket)
        
        response_data["image_url"] = image_url_path
        response_data["ticket_id"] = new_ticket.id
        
        return {"filename": file.filename, **response_data}

    except Exception as e:
        if isinstance(e, HTTPException): raise e
        print(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Clean up debug directory
        if os.path.exists(debug_scan_dir):
            try:
                pass # Keep debug dir for inspection
                shutil.rmtree(debug_scan_dir)
            except OSError as e:
                print(f"Error removing debug directory {debug_scan_dir}: {e.strerror}")

@app.post("/update-ticket-text")
def update_ticket_text(
    request: dict,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    print(f"🔄 Update ticket request received: {request}")
    if "ticket_id" not in request or "extracted_text" not in request:
        raise HTTPException(status_code=400, detail="ticket_id and extracted_text are required")
    ticket_id = request["ticket_id"]
    new_text = request["extracted_text"]
    ticket = db.query(models.Ticket).filter(
        models.Ticket.id == ticket_id,
        models.Ticket.owner_id == current_user.id
    ).first()
    if not ticket:
        print(f"❌ Ticket {ticket_id} not found for user {current_user.id}")
        raise HTTPException(status_code=404, detail="Ticket not found")
    print(f"📝 Updating ticket {ticket_id} text to: {new_text[:100]}...")
    ticket.extracted_text = new_text
    db.commit()
    db.refresh(ticket)
    response_data = {
        "message": "Ticket updated successfully",
        "ticket": {
            "id": ticket.id,
            "extracted_text": ticket.extracted_text,
            "image_url": ticket.image_path
        }
    }
    print(f"✅ Update successful for ticket {ticket_id}")
    return response_data

@app.get("/tickets")
def get_tickets(current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    tickets_from_db = db.query(models.Ticket).filter(models.Ticket.owner_id == current_user.id).all()
    response = []
    for ticket in tickets_from_db:
        response.append({
            "id": ticket.id,
            "extracted_text": ticket.extracted_text,
            "image_url": ticket.image_path
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

    # line detection done