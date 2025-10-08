import io
import os
import shutil
from PIL import Image
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import cv2
import numpy as np
import torch
from spellchecker import SpellChecker

# Local imports
import models
import database
from database import engine

# --- AI Model & Processor Setup ---
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

models.Base.metadata.create_all(bind=engine)

# --- File Storage Setup ---
TICKETS_DIR = "tickets"
TEMP_LINES_DIR = "temp_lines"  # Directory for temporary line images
DEBUG_DIR = "debug_output"      # Directory for debug images
os.makedirs(TICKETS_DIR, exist_ok=True)
os.makedirs(TEMP_LINES_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


app = FastAPI(title="Advanced Handwritten Scanner API")

app.mount(f"/{TICKETS_DIR}", StaticFiles(directory=TICKETS_DIR), name="tickets")

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
# --- Use the 'base' model as requested ---
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)
print("✅ TrOCR model loaded successfully.")

# --- OCR & Spell Correction Setup ---
print("Loading spell checker...")
try:
    spell_checker = SpellChecker()
    print("✅ Spell checker loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load spell checker: {e}")
    spell_checker = None

# --- Advanced Line Segmentation & OCR Functions (Restored from your script) ---

def preprocess_image(image):
    """Enhanced image preprocessing for better OCR results"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary

def segment_lines_alternative(image, output_dir, unique_filename):
    """Alternative line segmentation using horizontal projections."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    
    horizontal_projection = np.sum(binary, axis=1)
    
    threshold = np.max(horizontal_projection) * 0.1
    lines_y = []
    in_line = False
    start_y = 0
    
    for y, value in enumerate(horizontal_projection):
        if value > threshold and not in_line:
            in_line = True
            start_y = y
        elif value <= threshold and in_line:
            in_line = False
            end_y = y
            lines_y.append((start_y, end_y))
    
    cropped_paths = []
    debug_img = image.copy()
    padding = 10
    
    for i, (start_y, end_y) in enumerate(lines_y):
        height = end_y - start_y
        if height < 10:  # Skip very small lines
            continue
            
        y1 = max(0, start_y - padding)
        y2 = min(image.shape[0], end_y + padding)
        
        line_img = image[y1:y2, :]
        
        if line_img.size == 0:
            continue
            
        line_path = os.path.join(output_dir, f"line_{i:03d}.png")
        if cv2.imwrite(line_path, line_img):
            cropped_paths.append(line_path)
            cv2.rectangle(debug_img, (0, y1), (image.shape[1], y2), (255, 0, 0), 2) # Blue for alternative

    if cropped_paths:
        debug_path = os.path.join(DEBUG_DIR, f"{unique_filename}_debug_alt.png")
        cv2.imwrite(debug_path, debug_img)
        print(f"✅ Alternative Debug image saved to: {debug_path}")

    print(f"✅ Alternative method segmented {len(cropped_paths)} lines")
    return cropped_paths


def segment_lines(image_path, output_dir, unique_filename):
    """Improved line segmentation with a fallback to an alternative method."""
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    binary = preprocess_image(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠️ Warning: No contours found. Trying alternative method...")
        return segment_lines_alternative(image, output_dir, unique_filename)

    # --- CHANGE 1: Lowered the minimum area threshold to avoid filtering out small words ---
    min_area = image.shape[0] * image.shape[1] * 0.00005 
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not filtered_contours:
        print("⚠️ Warning: All contours filtered out. Trying alternative method...")
        return segment_lines_alternative(image, output_dir, unique_filename)

    bounding_boxes = sorted([cv2.boundingRect(c) for c in filtered_contours], key=lambda b: b[1])
    
    lines = []
    if bounding_boxes:
        # Filter out boxes that are likely noise before grouping into lines
        heights = [h for x, y, w, h in bounding_boxes if h > 5] # Basic noise filter
        if not heights:
             return segment_lines_alternative(image, output_dir, unique_filename)
        avg_height = np.mean(heights)

        current_line = [bounding_boxes[0]]
        for box in bounding_boxes[1:]:
            y_center_current = box[1] + box[3] / 2
            y_center_last = current_line[-1][1] + current_line[-1][3] / 2
            if abs(y_center_current - y_center_last) < avg_height * 0.7:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
        lines.append(current_line)
    
    return crop_and_save_lines(image, lines, output_dir, unique_filename)

def crop_and_save_lines(image, lines, output_dir, unique_filename):
    """Crop and save individual lines based on grouped bounding boxes."""
    cropped_paths = []
    debug_img = image.copy() # Create a copy for drawing rectangles
    padding = 15
    for i, line_boxes in enumerate(lines):
        if not line_boxes:
            continue
        
        x_min = min(b[0] for b in line_boxes)
        y_min = min(b[1] for b in line_boxes)
        x_max = max(b[0] + b[2] for b in line_boxes)
        y_max = max(b[1] + b[3] for b in line_boxes)
        
        # --- CHANGE 2: Lowered the minimum size for a cropped line to be processed ---
        if (x_max - x_min) < 10 or (y_max - y_min) < 5:
            continue
            
        y1, y2 = max(0, y_min - padding), min(image.shape[0], y_max + padding)
        x1, x2 = max(0, x_min - padding), min(image.shape[1], x_max + padding)
        
        line_img = image[y1:y2, x1:x2]
        line_path = os.path.join(output_dir, f"line_{i:03d}.png")
        if cv2.imwrite(line_path, line_img):
            cropped_paths.append(line_path)
            # --- Draw rectangle on the debug image ---
            cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
    # --- Save the debug image with drawn rectangles ---
    if cropped_paths:
        debug_path = os.path.join(DEBUG_DIR, f"{unique_filename}_debug.png")
        cv2.imwrite(debug_path, debug_img)
        print(f"✅ Debug image saved to: {debug_path}")

    return cropped_paths

def recognize_line(image_path, processor, model, device):
    """Recognizes text from a single line image."""
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as e:
        print(f"Error recognizing line {image_path}: {e}")
        return ""

def correct_spelling(text, spell_checker_instance):
    """Corrects spelling in the recognized text."""
    if not text or not spell_checker_instance:
        return text
    try:
        words = text.split()
        misspelled = spell_checker_instance.unknown(words)
        corrected_words = [spell_checker_instance.correction(word) if word in misspelled else word for word in words]
        return " ".join(w for w in corrected_words if w is not None)
    except Exception:
        return text

# --- Authentication Utility Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None: raise credentials_exception
    return user

# --- API Endpoints ---
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    new_user = models.User(username=form_data.username, hashed_password=get_password_hash(form_data.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully"}

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.id}

@app.post("/scan")
async def scan_ticket(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_filename = os.path.basename(file.filename)
    unique_filename = f"{timestamp}_{current_user.id}_{safe_filename}"
    saved_image_path = os.path.join(TICKETS_DIR, unique_filename)
    
    scan_temp_dir = os.path.join(TEMP_LINES_DIR, unique_filename)
    os.makedirs(scan_temp_dir, exist_ok=True)

    try:
        with open(saved_image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # --- Using your original, preferred line segmentation method ---
        print(f"Segmenting lines from {saved_image_path}...")
        line_image_paths = segment_lines(
            image_path=saved_image_path, 
            output_dir=scan_temp_dir,
            unique_filename=unique_filename # Pass filename for debug image
        )
        
        if not line_image_paths:
            raise HTTPException(status_code=400, detail="Could not detect any lines of text in the image.")

        print(f"Found {len(line_image_paths)} lines. Running OCR...")
        
        full_text = []
        for line_path in line_image_paths:
            raw_text = recognize_line(line_path, processor, model, device)
            corrected_text = correct_spelling(raw_text, spell_checker)
            full_text.append(corrected_text)
            print(f" - Raw: '{raw_text}' -> Corrected: '{corrected_text}'")

        extracted_text = "\n".join(full_text)

        new_ticket = models.Ticket(
            extracted_text=extracted_text, 
            owner_id=current_user.id,
            image_path=saved_image_path
        )
        db.add(new_ticket)
        db.commit()
        db.refresh(new_ticket)

        return {
            "filename": file.filename, 
            "extracted_text": extracted_text,
            "saved_path": saved_image_path
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        if os.path.exists(saved_image_path):
            os.remove(saved_image_path)
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
    finally:
        # IMPORTANT: Clean up the temporary directory with line images
        if os.path.exists(scan_temp_dir):
            shutil.rmtree(scan_temp_dir)
        
        # --- Clean up the debug images ---
        debug_path_main = os.path.join(DEBUG_DIR, f"{unique_filename}_debug.png")
        if os.path.exists(debug_path_main):
            os.remove(debug_path_main)
            
        debug_path_alt = os.path.join(DEBUG_DIR, f"{unique_filename}_debug_alt.png")
        if os.path.exists(debug_path_alt):
            os.remove(debug_path_alt)


@app.get("/tickets")
def get_user_tickets(current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    return db.query(models.Ticket).filter(models.Ticket.owner_id == current_user.id).all()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Advanced Handwritten Scanner API"}

