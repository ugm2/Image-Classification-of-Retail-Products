"""Retail Data Models."""
from pydantic import BaseModel

class RetailResponse(BaseModel):
    """Retail Response."""
    product_prediction: str