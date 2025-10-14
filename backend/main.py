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

# ------------------------------------------------------------------- #
# --- TABLE DETECTION & EXTRACTION (FINAL VERSION) ---
# ------------------------------------------------------------------- #

def enhance_cell_image(cell_cv_image):
    """
    Applies a series of OpenCV filters to a cropped cell image to make it
    as clear as possible for the OCR model.
    """
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
    
    # The TrOCR model expects a 3-channel input.
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(final_image_rgb)


def recognize_cell_text(cell_image: Image.Image):
    """Performs OCR on a single cell image using the global TrOCR model."""
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
    Final optimized pipeline: Detects cells, then enhances each cell's image
    before performing OCR for the highest accuracy.
    """
    print("Running final optimized table extraction...")
    if yolo_model is None:
        print("⚠️ YOLO model is not loaded. Skipping table detection.")
        return None

    original_image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2GRAY)
    processed_for_detection = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    processed_for_detection_bgr = cv2.cvtColor(processed_for_detection, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(debug_dir_path, "1_preprocessed_for_detection.png"), processed_for_detection_bgr)
    
    print("Detecting table cells...")
    results = yolo_model.predict(processed_for_detection_bgr, conf=0.2, verbose=False)

    if not results or results[0].boxes is None or results[0].boxes.xyxy is None:
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
# --- LINE SEGMENTATION FUNCTIONS (FALLBACK) ---
# ------------------------------------------------------------------- #
def extract_lines_data(image_path: str, unique_filename: str):
    scan_temp_dir = os.path.join(TEMP_LINES_DIR, unique_filename)
    os.makedirs(scan_temp_dir, exist_ok=True)
    try:
        line_image_paths = segment_lines(image_path, scan_temp_dir)
        if not line_image_paths: return None
        full_text = [recognize_line(p) for p in line_image_paths]
        return {"extracted_text": "\n".join(full_text)}
    finally:
        if os.path.exists(scan_temp_dir): shutil.rmtree(scan_temp_dir)

def segment_lines(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None: return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[1])
    heights = [h for _, _, _, h in bounding_boxes if h > 5]
    if not heights: return []
    avg_height, lines = np.mean(heights), []
    current_line = [bounding_boxes[0]]
    for box in bounding_boxes[1:]:
        if abs((box[1] + box[3] / 2) - (current_line[-1][1] + current_line[-1][3] / 2)) < avg_height:
            current_line.append(box)
        else:
            lines.append(current_line); current_line = [box]
    lines.append(current_line)
    return crop_and_save_lines(image, lines, output_dir)

def crop_and_save_lines(image, lines, output_dir):
    cropped_paths, padding = [], 15
    for i, line_boxes in enumerate(lines):
        if not line_boxes: continue
        x_min, y_min = min(b[0] for b in line_boxes), min(b[1] for b in line_boxes)
        x_max, y_max = max(b[0] + b[2] for b in line_boxes), max(b[1] + b[3] for b in line_boxes)
        y1, y2 = max(0, y_min - padding), min(image.shape[0], y_max + padding)
        x1, x2 = max(0, x_min - padding), min(image.shape[1], x_max + padding)
        line_img = image[y1:y2, x1:x2]
        line_path = os.path.join(output_dir, f"line_{i:03d}.png")
        if cv2.imwrite(line_path, line_img): cropped_paths.append(line_path)
    return cropped_paths

def recognize_line(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception: return ""

# ------------------------------------------------------------------- #
# --- AUTHENTICATION ---
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
    # Added user_id to the login response
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
        # Use a non-blocking way to write the file content
        file_content = await file.read()
        with open(saved_image_path, "wb") as f:
            f.write(file_content)
            
        image_pil = Image.open(io.BytesIO(file_content)).convert("RGB")
        table_result = extract_table_data_yolo(image_pil, debug_scan_dir)

        if table_result:
            print("✅ Table found! Processing as a table.")
            # If the result is a table, we now format it as a string for display
            # You could also keep it as JSON if the frontend can handle it
            table_string = "\n".join([" | ".join(map(str, row)) for row in table_result["extracted_table"]])
            db_text = table_string
            response_data = {"extracted_text": db_text} # Keep response simple
        else:
            print("⚠️ No table found. Falling back to line-by-line segmentation.")
            line_result = extract_lines_data(saved_image_path, unique_filename)
            if not line_result:
                raise HTTPException(status_code=400, detail="Could not detect any text in the image.")
            print("✅ Lines processed successfully.")
            db_text = line_result["extracted_text"]
            response_data = line_result

        # The path stored in DB should be the URL path, not the file system path
        image_url_path = f"/{TICKETS_DIR}/{unique_filename}"
        new_ticket = models.Ticket(extracted_text=db_text, owner_id=current_user.id, image_path=image_url_path)
        db.add(new_ticket); db.commit(); db.refresh(new_ticket)
        
        # Add the saved_path to the response for the frontend
        response_data["image_url"] = image_url_path
        return {"filename": file.filename, **response_data}

    except Exception as e:
        if isinstance(e, HTTPException): raise e
        print(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Clean up the temporary debug directory
        if os.path.exists(debug_scan_dir):
            try:
                shutil.rmtree(debug_scan_dir)
            except OSError as e:
                print(f"Error removing debug directory {debug_scan_dir}: {e.strerror}")

# ADDED: Update ticket endpoint
# Add this endpoint - use a completely different path to avoid conflicts
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
    
    # Find the ticket
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

# This endpoint is updated to return a web-accessible URL
@app.get("/tickets")
def get_tickets(current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    tickets_from_db = db.query(models.Ticket).filter(models.Ticket.owner_id == current_user.id).all()
    
    # Manually construct the response to ensure the field is named `image_url`
    response = []
    for ticket in tickets_from_db:
        response.append({
            "id": ticket.id,
            "extracted_text": ticket.extracted_text,
            "image_url": ticket.image_path # Assumes image_path is stored as a URL path
        })
    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to the Advanced Handwritten Scanner API"}