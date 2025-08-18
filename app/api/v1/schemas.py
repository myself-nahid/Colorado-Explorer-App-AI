from pydantic import BaseModel
from typing import List

class GuideRequest(BaseModel):
    user_id: str
    prompt: str
    session_id: str  
    
class GuideResponse(BaseModel):
    response: str
    session_id: str