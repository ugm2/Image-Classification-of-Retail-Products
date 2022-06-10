"""Retail Data Models."""
from pydantic import BaseModel

class RetailResponse(BaseModel):
    """Retail Response."""
    prediction: str
    confidence: float
    
class RetailLabels(BaseModel):
    """Retail Labels."""
    labels: list