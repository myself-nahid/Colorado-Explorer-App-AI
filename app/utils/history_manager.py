from typing import List, Dict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

conversation_history: Dict[str, Dict[str, List[BaseMessage]]] = {}

def get_history(user_id: str, session_id: str) -> List[BaseMessage]:
    """Retrieves the conversation history for a given user and session."""
    return conversation_history.get(user_id, {}).get(session_id, [])

def update_history(user_id: str, session_id: str, human_message: HumanMessage, ai_message: AIMessage):
    """Updates the conversation history."""
    if user_id not in conversation_history:
        conversation_history[user_id] = {}
    if session_id not in conversation_history[user_id]:
        conversation_history[user_id][session_id] = []
    
    conversation_history[user_id][session_id].append(human_message)
    conversation_history[user_id][session_id].append(ai_message)