from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- IMPORTANT ---
# Replace this URL with your actual PostgreSQL connection details.
# Format: "postgresql://<user>:<password>@<host>:<port>/<database_name>"
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:shivanirao1710@localhost/ticketdb"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get a DB session for each request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()