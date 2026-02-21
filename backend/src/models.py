from pydantic import BaseModel
from typing import Optional

class DiagnoseRequest(BaseModel):
    symptoms: Optional[str] = ""

class Diagnosis(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str
    explanation: str

class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]
