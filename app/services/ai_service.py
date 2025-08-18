from typing import TypedDict, Annotated, List, Union
import operator
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.services.location_service import location_service
from app.utils.history_manager import get_history

@tool
def search_colorado_places(query: str) -> list:
    """
    Use this tool to find information about places, activities, or locations within Colorado.
    The input should be a descriptive search query.
    For example: 'hiking trails near Denver' or 'best breweries in Fort Collins'.
    """
    print(f"--- Calling Google Maps Tool with query: {query} ---")
    return location_service.search_places_in_colorado(query)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

class AIGuideAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=settings.GEMINI_API_KEY
        )
        self.tools = [search_colorado_places]
        self.model_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _should_continue(self, state: AgentState) -> str:
        """Determines whether to continue the loop or end."""
        last_message = state['messages'][-1]
        if not last_message.tool_calls:
            return "end"
        return "continue"

    def _call_model(self, state: AgentState):
        """The primary node that calls the LLM."""
        response = self.model_with_tools.invoke(state['messages'])
        return {"messages": [response]}

    def _build_graph(self) -> StateGraph:
        """Builds the LangGraph agent."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._call_model)
        tool_node = ToolNode(self.tools)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def generate_guide(self, user_id: str, prompt: str, session_id: str) -> str:
        """Runs the agent to generate a guide."""
        chat_history = get_history(user_id, session_id)
        
        system_prompt = HumanMessage(
            content=(
                "You are a specialized AI assistant for the 'Colorado Explorer App'. "
                "Your name is 'Explorer'. You are friendly, enthusiastic, and an expert on all things Colorado. "
                "Your primary goal is to provide personalized, helpful, and engaging travel recommendations. "
                "Always restrict your answers to the state of Colorado. "
                "If asked about anything outside Colorado, politely state your focus is only on Colorado. "
                "Use the 'search_colorado_places' tool whenever the user asks for specific locations, activities, or recommendations."
            )
        )
        
        current_conversation = [system_prompt] + chat_history + [HumanMessage(content=prompt)]
        
        inputs = {"messages": current_conversation}
        
        final_state = self.graph.invoke(inputs)
        
        response_message = final_state['messages'][-1]
        return response_message.content

ai_guide_agent = AIGuideAgent()