"""
## Simple Bot

1. Define *state* structure with a list of *HumanMessage* objects
2. Initialize a *GPT-4o* model using LangChain's ChatOpenAI
3. Sending and handling different types of messages
4. Building and compiling the graph of the *Agent*
"""

# Library imports
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load API Keys
load_dotenv()

# Define AgentState
class AgentState(TypedDict):
    messages: List[HumanMessage]

# Define LLM
llm = ChatOpenAI(model="gpt-4o")

# Define Node
def process(state: AgentState) -> AgentState:
    """Function to get LLM response to human input message"""
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

# Define Graph
graph = StateGraph(AgentState)
graph.add_node("Process", process)
graph.add_edge(START, "Process")
graph.add_edge("Process", END)
agent = graph.compile()

# User Inputs
user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("\nEnter: ")