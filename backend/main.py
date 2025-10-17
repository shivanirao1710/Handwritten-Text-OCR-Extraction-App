import io
import os
import shutil
import json
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
YOLO_MODEL_PATH = 'best.pt' # Make sure this 'best.pt' file is in the same directory as your script

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

# 1. TrOCR Model
print("Loading Hugging Face TrOCR model...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(device)
print("✅ TrOCR model loaded successfully.")

# 2. Custom YOLOv8 Model for Cell Detection
print("Loading custom YOLOv8 model for table cell detection...")
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"❌ CRITICAL ERROR: YOLO model not found at '{YOLO_MODEL_PATH}'")
    yolo_model = None
else:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ Custom YOLOv8 model loaded successfully.")

# --------------------------------------------------------
# To identify dollar symbol
# --------------------------------------------------------
def correct_currency_symbols(text: str) -> str:
    """
    Corrects OCR errors where '$' is mistaken for 's' or 'S'.
    This is safe and won't affect regular words like "is".
    
    Examples:
    - "s190" -> "$190"
    - "S 0.00" -> "$ 0.00"
    - "is 5 dollars" -> "is 5 dollars" (No change, which is correct)
    """
    # This regex finds an 's' or 'S' that is at the start of a word boundary
    # AND is followed by a digit or a decimal point (with an optional space).
    # The (?=...) part is a "lookahead" that checks without being part of the match.
    corrected_text = re.sub(r'\b[sS](?=\s?[\d.])', '$', text)
    return corrected_text

# ------------------------------------------------------------------- #
# --- NEW: IMAGE PREPROCESSING ---
# ------------------------------------------------------------------- #

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Converts any input image to a standardized format with a white background
    and black text for optimal OCR performance.

    Args:
        image (Image.Image): The input PIL Image.

    Returns:
        Image.Image: The processed PIL Image with a white background and black text.
    """
    # Convert PIL Image to OpenCV format (BGR)
    open_cv_image = np.array(image.convert("RGB"))
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Apply a slight blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding to binarize the image. This automatically finds
    # the best threshold to separate foreground and background.
    _, binarized = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure the background is white and text is black.
    # We check the average color of the binarized image. If the average is less
    # than 128, it means most of the image is black (dark background),
    # so we need to invert the colors.
    if np.mean(binarized) < 128:
        binarized = cv2.bitwise_not(binarized)

    # Convert the processed grayscale image back to an RGB PIL Image
    # as the rest of the pipeline expects a 3-channel image.
    final_image_rgb = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_image_rgb)


# ------------------------------------------------------------------- #
# --- TABLE DETECTION & EXTRACTION (UNCHANGED) ---
# ------------------------------------------------------------------- #

def enhance_cell_image(cell_cv_image):
    if cell_cv_image.shape[0] < 10 or cell_cv_image.shape[1] < 10:
        return None
    gray = cv2.cvtColor(cell_cv_image, cv2.COLOR_BGR2GRAY)
    target_height = 64
    aspect_ratio = target_height / gray.shape[0]
    new_width = int(gray.shape[1] * aspect_ratio)
    resized = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    contrasted = clahe.apply(resized)
    _, final_image = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_GRAY2RGB)
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

def extract_table_data_yolo(image: Image.Image, debug_dir_path: str):
    print("Running primary table extraction with YOLO...")
    if yolo_model is None:
        print("⚠️ YOLO model is not loaded. Skipping table detection.")
        return None
    # Since the input image is already preprocessed (B&W), we use it directly.
    original_image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(debug_dir_path, "1_preprocessed_for_detection.png"), original_image_cv)
    
    print("Detecting table cells...")
    # The preprocessed image is ideal for detection.
    results = yolo_model.predict(original_image_cv, conf=0.2, verbose=False)

    if not results or results[0].boxes is None or not results[0].boxes.xyxy.nelement():
        return None
    cell_boxes = sorted(results[0].boxes.cpu().numpy().xyxy.astype(int).tolist(), key=lambda b: (b[1], b[0]))
    if not cell_boxes:
        return None

    print(f"Detected {len(cell_boxes)} cells. Reconstructing table structure...")
    rows = []
    current_row = []
    if cell_boxes:
        ref_y = cell_boxes[0][1]
        cell_height = cell_boxes[0][3] - cell_boxes[0][1]
        for box in cell_boxes:
            if box[1] > ref_y + cell_height * 0.8:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
                ref_y = box[1]
            else:
                current_row.append(box)
        rows.append(sorted(current_row, key=lambda b: b[0]))

    table_data = []
    print("Enhancing and performing OCR on detected cells...")
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    for i, row_boxes in enumerate(tqdm(rows, desc="Reading Rows")):
        row_text = []
        for j, box in enumerate(row_boxes):
            x1, y1, x2, y2 = box
            padding = 2
            # Crop from the preprocessed image passed to this function
            cell_image_cv = original_image_cv[
                max(0, y1 - padding):min(original_image_cv.shape[0], y2 + padding),
                max(0, x1 - padding):min(original_image_cv.shape[1], x2 + padding)
            ]
            enhanced_cell_pil = enhance_cell_image(cell_image_cv)
            if enhanced_cell_pil:
                enhanced_cell_pil.save(os.path.join(debug_dir_path, f"cell_{i:02d}_{j:02d}.png"))
            draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
            raw_text = recognize_cell_text(enhanced_cell_pil)
            row_text.append(raw_text)
        table_data.append(row_text)

    draw_img.save(os.path.join(debug_dir_path, "2_detected_cells.png"))
    return {"extracted_table": table_data, "debug_output_path": debug_dir_path}

# ------------------------------------------------------------------- #
# --- MODIFIED CELL SEGMENTATION (FALLBACK LOGIC) ---
# ------------------------------------------------------------------- #

def extract_lines_data(image_path: str, unique_filename: str):
    """
    Manages the fallback process: segments image into cells, recognizes text in each,
    and formats the result as a table-like string.
    """
    scan_temp_dir = os.path.join(TEMP_LINES_DIR, unique_filename)
    os.makedirs(scan_temp_dir, exist_ok=True)
    try:
        # segment_lines returns a 2D list of cell image paths (rows of cells)
        cell_image_paths_by_row = segment_lines(image_path, scan_temp_dir)
        if not cell_image_paths_by_row: 
            return None
        
        # 1. Calculate the total number of cells for an accurate progress bar
        total_cells = sum(len(row) for row in cell_image_paths_by_row)
        print(f"Fallback OCR started on {len(cell_image_paths_by_row)} rows ({total_cells} cells)...")

        table_data = []
        
        # 2. Create a tqdm progress bar instance
        with tqdm(total=total_cells, desc="Reading Cells (Fallback)") as pbar:
            # 3. Loop through each row and each cell to process them
            for row_paths in cell_image_paths_by_row:
                row_text = []
                for cell_path in row_paths:
                    # Recognize text for a single cell
                    text = recognize_line(cell_path)
                    row_text.append(text)
                    # 4. Update the progress bar after each cell is done
                    pbar.update(1)
                table_data.append(row_text)

        # Format the output to be consistent with the YOLO extractor
        table_string = "\n".join([" | ".join(map(str, row)) for row in table_data])
        return {"extracted_text": table_string}
    finally:
        if os.path.exists(scan_temp_dir): 
            shutil.rmtree(scan_temp_dir)

def segment_lines(image_path, output_dir):
    """
    Segments an image into lines and then splits those lines into cells based on
    horizontal spacing. This is for borderless tables.
    Returns a 2D list of cropped cell image paths.
    """
    # This function now receives the path to the already preprocessed image
    image = cv2.imread(image_path)
    if image is None: 
        return []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The image is already binarized, but re-applying thresholding is harmless
    # and ensures consistency if this function is ever called directly.
    binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)[1]
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return []

    word_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 15]
    if not word_boxes: 
        return []
            
    # Group word boxes into lines based on vertical position
    word_boxes.sort(key=lambda b: b[1]) # Sort by y-coordinate
    
    lines = []
    current_line = []
    if word_boxes:
        current_line.append(word_boxes[0])
        avg_height = np.mean([h for _, _, _, h in word_boxes])

        for box in word_boxes[1:]:
            last_box = current_line[-1]
            # If the vertical center of the new box is close to the last one, it's on the same line
            if abs((box[1] + box[3] / 2) - (last_box[1] + last_box[3] / 2)) < avg_height * 0.7:
                current_line.append(box)
            else:
                lines.append(sorted(current_line, key=lambda b: b[0]))
                current_line = [box]
        lines.append(sorted(current_line, key=lambda b: b[0]))
        
    # Split each line into cells based on horizontal gaps
    all_cells_by_row = []
    avg_char_width = np.mean([w for _, _, w, _ in word_boxes])
    gap_threshold = avg_char_width * 2.0 # A gap of ~2.0 avg chars indicates a new column

    for line in lines:
        if not line: continue
        
        cells_in_line = []
        current_cell = [line[0]]
        
        for i in range(len(line) - 1):
            current_word_box = line[i]
            next_word_box = line[i+1]
            gap = next_word_box[0] - (current_word_box[0] + current_word_box[2])
            
            if gap > gap_threshold:
                cells_in_line.append(current_cell)
                current_cell = [next_word_box]
            else:
                current_cell.append(next_word_box)
        
        cells_in_line.append(current_cell)
        all_cells_by_row.append(cells_in_line)
        
    return crop_and_save_cells(image, all_cells_by_row, output_dir)

def crop_and_save_cells(image, rows_of_cells, output_dir):
    """
    Crops and saves each detected cell.
    'rows_of_cells' is a 3D list: [row[cell[word_box]]].
    Returns a 2D list of paths: [row[cell_path]].
    """
    row_paths = []
    padding = 10
    
    for i, row in enumerate(rows_of_cells):
        cell_paths_in_row = []
        for j, cell_boxes in enumerate(row):
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
            row_paths.append(cell_paths_in_row)
            
    return row_paths

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
# --- API ENDPOINTS ---
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
        
        # 1. Load the original image from the uploaded file content
        original_image_pil = Image.open(io.BytesIO(file_content)).convert("RGB")

        # 2. Preprocess the image to standardize it to black text on a white background
        print("Preprocessing image to standardize background and text color...")
        processed_image_pil = preprocess_image_for_ocr(original_image_pil)
        print("✅ Image preprocessing complete.")
        
        # 3. Save the PROCESSED image. This standardized image will now be used
        #    by both the primary (YOLO) and fallback (contour) methods.
        processed_image_pil.save(saved_image_path, format='PNG')
            
        # The rest of the logic now uses the preprocessed image
        table_result = extract_table_data_yolo(processed_image_pil, debug_scan_dir)

        if table_result and table_result.get("extracted_table"):
            print("✅ Table found via YOLO! Processing as a table.")
            table_string = "\n".join([" | ".join(map(str, row)) for row in table_result["extracted_table"]])
            db_text = table_string
            db_text = correct_currency_symbols(db_text)
            # Only include the text, as the frontend will format it
            response_data = {"extracted_text": db_text} 
        else:
            print("⚠️ No table found via YOLO. Falling back to contour-based cell segmentation.")
            # This function reads `saved_image_path`, which now contains the processed image.
            line_result = extract_lines_data(saved_image_path, unique_filename)
            if not line_result or not line_result.get("extracted_text"):
                raise HTTPException(status_code=400, detail="Could not detect any text in the image.")
            print("✅ Fallback processing successful.")
            db_text = line_result["extracted_text"]
            db_text = correct_currency_symbols(db_text)
            response_data = line_result

        image_url_path = f"/{TICKETS_DIR}/{unique_filename}"
        new_ticket = models.Ticket(extracted_text=db_text, owner_id=current_user.id, image_path=image_url_path)
        db.add(new_ticket); db.commit(); db.refresh(new_ticket)
        
        # Add ticket id and image url to the final response
        response_data["image_url"] = image_url_path
        response_data["ticket_id"] = new_ticket.id
        return {"filename": file.filename, **response_data}

    except Exception as e:
        if isinstance(e, HTTPException): raise e
        print(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Keep debug directory for inspection, remove the 'finally' block if you want it deleted
        pass
        if os.path.exists(debug_scan_dir):
            try:
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
    print(f"📝 Updating ticket {ticket_id} text to: {new_text}")
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
    print(f"✅ Update successful: {response_data}")
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

# Works well for colour background
