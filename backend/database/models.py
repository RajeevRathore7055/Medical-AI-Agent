from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Boolean
from sqlalchemy.sql import func
from backend.database.database import Base


class Doctor(Base):
    __tablename__ = "doctors"
    id             = Column(Integer, primary_key=True, index=True)
    name           = Column(String(100), nullable=False)
    specialization = Column(String(100))
    department     = Column(String(100))
    available_days = Column(String(200))
    timing         = Column(String(100))
    room_number    = Column(String(20))
    contact        = Column(String(20))
    experience     = Column(Integer)
    created_at     = Column(DateTime, server_default=func.now())


class Patient(Base):
    __tablename__ = "patients"
    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String(100), nullable=False)
    age          = Column(Integer)
    gender       = Column(String(10))
    blood_group  = Column(String(5))
    contact      = Column(String(20))
    address      = Column(Text)
    admitted_date = Column(Date)
    ward         = Column(String(50))
    bed_number   = Column(String(20))
    doctor_name  = Column(String(100))
    diagnosis    = Column(Text)
    status       = Column(String(20), default="Active")
    created_at   = Column(DateTime, server_default=func.now())


class Appointment(Base):
    __tablename__ = "appointments"
    id              = Column(Integer, primary_key=True, index=True)
    patient_name    = Column(String(100))
    doctor_name     = Column(String(100))
    department      = Column(String(100))
    appointment_date = Column(Date)
    appointment_time = Column(String(20))
    status          = Column(String(20), default="Scheduled")
    notes           = Column(Text)
    created_at      = Column(DateTime, server_default=func.now())


class Billing(Base):
    __tablename__ = "billing"
    id               = Column(Integer, primary_key=True, index=True)
    patient_name     = Column(String(100))
    patient_id       = Column(Integer)
    consultation_fee = Column(Float, default=0)
    medicine_cost    = Column(Float, default=0)
    lab_cost         = Column(Float, default=0)
    room_charges     = Column(Float, default=0)
    total_amount     = Column(Float, default=0)
    paid_amount      = Column(Float, default=0)
    pending_amount   = Column(Float, default=0)
    payment_status   = Column(String(20), default="Pending")
    billing_date     = Column(Date)
    created_at       = Column(DateTime, server_default=func.now())


class Bed(Base):
    __tablename__ = "beds"
    id           = Column(Integer, primary_key=True, index=True)
    ward_name    = Column(String(100))
    bed_number   = Column(String(20))
    bed_type     = Column(String(50))
    is_available = Column(Boolean, default=True)
    patient_name = Column(String(100), nullable=True)
    floor        = Column(Integer)
    created_at   = Column(DateTime, server_default=func.now())


class Department(Base):
    __tablename__ = "departments"
    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String(100), nullable=False)
    head_doctor  = Column(String(100))
    floor        = Column(Integer)
    contact      = Column(String(20))
    total_beds   = Column(Integer, default=0)
    description  = Column(Text)
    created_at   = Column(DateTime, server_default=func.now())


class Staff(Base):
    __tablename__ = "staff"
    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(100), nullable=False)
    role       = Column(String(100))
    department = Column(String(100))
    shift      = Column(String(50))
    contact    = Column(String(20))
    email      = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())
