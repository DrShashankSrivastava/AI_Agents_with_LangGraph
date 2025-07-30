"""
## RAG

"""

# Library imports
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from operator import add as add_messages
from langgraph.graph import StateGraph, START, END

# Load API Keys
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Embedding model
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Load pdf document
pdf_path = "IntroAircraftAeroelasticityLoads.pdf"
# Path sanity check
if not os.path.exists(os.path.join(os.getcwd(), pdf_path)):
    raise FileExistsError(f"File {pdf_path} does not exist")
# Loading step
pdf_loader = PyPDFLoader(pdf_path)
# Load sanity check
try:
    pages = pdf_loader.load()
    print(f"PDF {pdf_path.split(".")[-2]} has been loaded and has {len(pages)} pages.")
except Exception as e:
    print(f"Error loading the PDF: {e}")
    raise

# Split and chunk text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pages_split = text_splitter.split_documents(pages)

# Create Chroma Vector Database
try:
    vectorstore = Chroma.from_documents(documents=pages_split,
                                        embedding=embedding,
                                        persist_directory=os.getcwd(),
                                        collection_name="aeroelasticity")
    print("Created ChromaDB vector store.")
except Exception as e:
    print(f"Error setting up ChromaDB vector store: {e}")
    raise

# Define Retriever
retriever = vectorstore.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 5}) # K: Amount of chunks to return

# Define Tools
@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the ingested PDF document
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the ingested document(s)"

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n".join(results)

tools = [retriever_tool]

# Bind tools to LLM Model
llm = llm.bind_tools(tools)

# Define AgentState
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define Conditional Edge
def should_continue(state: AgentState) -> bool:
    """Determine if the agent should continue"""

    return hasattr(state["messages"][-1], 'tool_calls') and len(state["messages"][-1].tool_calls) > 0

# System prompt
system_prompt = """
You are an intelligent AI assistant who answers questions about the ingested document in your knowledge base.
Use the retriever tool available to answer questions about the topics covered in the ingested document. You can make multiple calls if required.
If you need to look up some information before asking a follow up question, you are allowed to do that.
Please always cite the specific parts of th documents you use in your answers.  
"""
