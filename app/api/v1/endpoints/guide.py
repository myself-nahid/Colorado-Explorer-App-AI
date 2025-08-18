from fastapi import APIRouter, HTTPException
from app.api.v1.schemas import GuideRequest, GuideResponse
from app.services.ai_service import ai_guide_agent
from app.utils.history_manager import update_history, get_history
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

@router.post("/generate", response_model=GuideResponse)
async def generate_guide_endpoint(request: GuideRequest):
    """
    Receives a user prompt and generates a personalized Colorado guide.
    This endpoint maintains a conversation history for each user session.
    """
    try:
        response_content = ai_guide_agent.generate_guide(
            request.user_id, request.prompt, request.session_id
        )

        update_history(
            request.user_id,
            request.session_id,
            HumanMessage(content=request.prompt),
            AIMessage(content=response_content)
        )
        
        return GuideResponse(response=response_content, session_id=request.session_id)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while generating the guide.")