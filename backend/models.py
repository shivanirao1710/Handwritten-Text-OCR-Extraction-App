from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, func, Float
from sqlalchemy.orm import relationship

# Absolute import from your database.py file
from database import Base


class User(Base):
    """SQLAlchemy model for the User table."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    tickets = relationship("Ticket", back_populates="owner")


class Ticket(Base):
    """
    SQLAlchemy model for the Ticket table.
    
    This model is updated to store both the raw OCR text dump
    and the new structured (Key-Value) fields.
    """
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    
    # --- UPDATED FIELD ---
    # This now stores the path to the combined PDF
    pdf_path = Column(String, nullable=True) 
    
    created_at = Column(DateTime, default=func.now())
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="tickets")
    
    # This field will now contain the combined text from ALL pages
    raw_text_content = Column(String, nullable=True)

    # --- Structured Data Fields (Unchanged) ---
    ticket_number = Column(String, index=True, nullable=True)
    ticket_date = Column(String, nullable=True) # Storing as string for simplicity
    haul_vendor = Column(String, nullable=True)
    truck_number = Column(String, nullable=True)
    material = Column(String, nullable=True)
    job_number = Column(String, nullable=True)
    phase_code = Column(String, nullable=True)
    zone = Column(String, nullable=True)
    hours = Column(Float, nullable=True)