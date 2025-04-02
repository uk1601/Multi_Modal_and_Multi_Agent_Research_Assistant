from langgraph.graph.state import CompiledStateGraph

from backend.agents.bg_task_agent.bg_task_agent import bg_task_agent
from backend.agents.chatbot import chatbot
from backend.agents.research_assistant import research_assistant
from backend.agents.multi_modal_rag import research_agent

DEFAULT_AGENT = "multi-modal-rag"


agents: dict[str, CompiledStateGraph] = {
    "chatbot": chatbot,
    "research-assistant": research_assistant,
    "bg-task-agent": bg_task_agent,
    "multi-modal-rag": research_agent,
}
