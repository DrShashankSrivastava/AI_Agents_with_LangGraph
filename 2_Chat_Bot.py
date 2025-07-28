"""
## Chatbot

1. Use different message types - *HumanMessage* and *AIMessage*
2. Maintain a full conversation history using both message types
3. Use GPT-4o model using *LangChain's ChatOpenAI*
4. Create a sophisticated conversation loop

"""

# Library imports
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load API Keys
load_dotenv()

# Define AgentState
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# Define LLM
llm = ChatOpenAI(model="gpt-4o")

# Define Node
def process(state: AgentState) -> AgentState:
    """Function to get AI responses to user inputs"""

    response = llm.invoke(state["messages"])
    print(f"AI response: {response.content}")
    state["messages"].append(AIMessage(content=response.content))

    return state

# Define Graph
graph = StateGraph(AgentState)
graph.add_node("Process", process)
graph.add_edge(START, "Process")
graph.add_edge("Process", END)
agent = graph.compile()

# Human and AI conversation
conversation_history = []
user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("\nEnter: ")

# Log the conversation
with open("conversation_log.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    for conversation in conversation_history:
        if isinstance(conversation, HumanMessage):
            file.write(f"You: {conversation.content}\n")
        elif isinstance(conversation, AIMessage):
            file.write(f"AI: {conversation.content}\n")
    file.write("End of Conversation")

print("Conversation saved to conversation_log.txt")