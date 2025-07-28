"""
## ReAct Agent - Reasoning and Acting Agent

"""

# Library imports
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Load API Keys
load_dotenv()

# Define AgentState
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # add_messages - reducer function

# Define Tools
@tool
def add(a: int, b: int) -> int:
    """This is an addition function to add 2 numbers"""

    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """This is an subtraction function to subtract 2 numbers"""

    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """This is an multiplication function to multiply 2 numbers"""

    return a * b

# Define tools for LLM
tools = [add, subtract, multiply]

# Define LLM
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# Define Node
def model_call(state: AgentState) -> AgentState:
    """Node for model call"""

    system_prompt = SystemMessage(content="You are my AI assistant. Answer my query the best you can.")
    response = model.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}  # New messages are already added by 'add_messages'

# Define Conditional Edge
def should_continue(state: AgentState) -> str:
    """Conditional to check if loop to continue or not"""

    if not(state["messages"][-1]).tool_calls:
        return "exit"
    else:
        return "continue"

# Define Graph
graph = StateGraph(AgentState)
graph.add_node("Agent Node", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("Tool Node", tool_node)

graph.add_edge(START, "Agent Node")
graph.add_conditional_edges(
    "Agent Node",
    should_continue,
    {
        "continue": "Tool Node",
        "exit": END
    }
)
graph.add_edge("Tool Node", "Agent Node")
app = graph.compile()

# Printing Message
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# User input
inputs = {"messages": [("user", "Tell me a joke and then add 56 and 64.")]}
print_stream(app.stream(inputs, stream_mode="values"))
