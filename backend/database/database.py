import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

Base = declarative_base()

def get_db_url():
    db_type = os.getenv("DB_TYPE", "mysql")
    if db_type == "mysql":
        return os.getenv("MYSQL_URL", "mysql+pymysql://root:root@localhost:3306/medical_db")
    return os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/medical_db")

def create_db_engine():
    url = get_db_url()
    try:
        engine = create_engine(url, pool_pre_ping=True, echo=False)
        logger.info(f"✅ Database connected: {url.split('@')[-1]}")
        return engine
    except Exception as e:
        logger.error(f"❌ DB connection failed: {e}")
        raise

engine       = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from backend.database import models
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Tables created!")
