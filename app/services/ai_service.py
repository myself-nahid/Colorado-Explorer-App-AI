from typing import TypedDict, Annotated
import operator
import time 
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_tavily import TavilySearch 

from app.core.config import settings
from app.services.location_service import location_service
from app.utils.history_manager import get_history

@tool
def search_colorado_places(query: str) -> list:
    """
    Use this tool to find specific places, points of interest, businesses, or addresses in Colorado.
    It is best for queries like 'restaurants near Denver', 'Garden of the Gods address', or 'breweries in Fort Collins'.
    This tool provides structured location data like names, addresses, and ratings.
    """
    start_time = time.time()
    print(f"--- Calling Google Maps Tool with query: {query} ---")
    results = location_service.search_places_in_colorado(query)
    end_time = time.time()
    print(f"--- Google Maps Tool took {end_time - start_time:.2f} seconds. ---")
    return results

tavily_web_search = TavilySearch(
    max_results=3, 
    tavily_api_key=settings.TAVILY_API_KEY
)
tavily_web_search.name = "web_search"
tavily_web_search.description = (
    "Use this tool to search the web for general information, real-time events, news, weather, "
    "temporary closures, or detailed descriptions of places and activities. "
    "It is best for questions that Google Maps cannot answer, such as 'Are there any wildfires near Estes Park?', "
    "'What's the history of the Stanley Hotel?', or 'upcoming concerts in Denver'."
)

@tool
def timed_web_search(query: str) -> list:
    """
    Use this tool to search the web for general information, real-time events, news, weather,
    temporary closures, or detailed descriptions of places and activities.
    It is best for questions that Google Maps cannot answer, such as 'Are there any wildfires near Estes Park?',
    'What's the history of the Stanley Hotel?', or 'upcoming concerts in Denver'.
    """
    start_time = time.time()
    print(f"--- Calling Tavily Web Search with query: {query} ---")
    results = tavily_web_search.invoke({"query": query})
    end_time = time.time()
    print(f"--- Tavily Web Search took {end_time - start_time:.2f} seconds. ---")
    return results

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

class AIGuideAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", 
            google_api_key=settings.GEMINI_API_KEY
        )
        
        self.tools = [search_colorado_places, timed_web_search]
        self.model_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _should_continue(self, state: AgentState) -> str:
        last_message = state['messages'][-1]
        if not last_message.tool_calls:
            return "end"
        return "continue"

    def _call_model(self, state: AgentState):
        response = self.model_with_tools.invoke(state['messages'])
        return {"messages": [response]}

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._call_model)
        tool_node = ToolNode(self.tools)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def generate_guide(self, user_id: str, prompt: str, session_id: str) -> str:
        chat_history = get_history(user_id, session_id)
        
        system_prompt = HumanMessage(
            content=(
                "You are a specialized AI assistant for the 'Colorado Explorer App'. "
                "Your name is 'Explorer'. You are friendly, enthusiastic, and an expert on all things Colorado. "
                "Your primary goal is to provide personalized, helpful, and engaging travel recommendations. "
                "Always restrict your answers to the state of Colorado. "
                "You have two types of tools: "
                "1. `search_colorado_places`: Use this for finding specific locations, businesses, and addresses. "
                "2. `timed_web_search`: Use this for everything else, including real-time information like events, weather, news, temporary closures, and general knowledge questions about Colorado's history or culture."
                "IMPORTANT: You must respond in the same language as the user's prompt. If the user asks a question in Spanish, your entire response must be in fluent, natural Spanish."
            )
        )
        
        current_conversation = [system_prompt] + chat_history + [HumanMessage(content=prompt)]
        
        inputs = {"messages": current_conversation}
        
        final_state = self.graph.invoke(inputs)
        
        response_message = final_state['messages'][-1]
        return response_message.content

ai_guide_agent = AIGuideAgent()