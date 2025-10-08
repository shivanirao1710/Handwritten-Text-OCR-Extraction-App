from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship

# Changed from relative to absolute import
from database import Base


class User(Base):
    """SQLAlchemy model for the User table."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    tickets = relationship("Ticket", back_populates="owner")


class Ticket(Base):
    """SQLAlchemy model for the Ticket table."""
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    extracted_text = Column(String, index=True, nullable=False)
    # This line is the crucial addition
    image_path = Column(String, nullable=True)  # Path to the saved image file
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=func.now())

    owner = relationship("User", back_populates="tickets")

