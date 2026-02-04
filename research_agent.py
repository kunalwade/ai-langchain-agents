import os
import json
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI

# --- 1. DEFINE THE AGENT'S STATE ---
# The state is a dictionary that will be passed between nodes in our graph.
# It represents the current "memory" of the agent.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- 2. SETUP THE TOOLS ---
# An agent needs tools to interact with the world.

# Tool 1: A Web Search Tool using Tavily
# This allows the agent to search the internet for real-time information.
web_search_tool = TavilySearchResults(k=3) # k=3 means it will return the top 3 search results

# Tool 2: A RAG Retriever Tool for our PDF
# This reuses our RAG pipeline from the previous project but packages it as a tool.
def create_pdf_retriever(pdf_path):
    """Creates a RAG retriever for a given PDF file."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vector_store.as_retriever()

pdf_file_path = "sample_document.pdf"
if not os.path.exists(pdf_file_path):
    raise FileNotFoundError(f"The file '{pdf_file_path}' was not found. Please add it to your project directory.")

pdf_retriever = create_pdf_retriever(pdf_file_path)

# The create_retriever_tool function wraps our retriever in a tool interface
# that the agent can easily call.
pdf_retriever_tool = create_retriever_tool(
    pdf_retriever,
    "pdf_document_search",
    "Searches and returns information from the provided PDF document about machine learning."
)

# Combine all tools into a list and create a ToolExecutor
tools = [web_search_tool, pdf_retriever_tool]
tool_executor = ToolExecutor(tools)

# --- 3. DEFINE THE GRAPH NODES ---
# Our graph will have three main nodes: the agent, the tool executor, and a safety guardrail.

# Node 1: The Agent (The "Brain")
# This node decides what to do next. It can either call a tool or finish.
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
model_with_tools = llm.bind_tools(tools) # Bind the tools to the model

def agent_node(state):
    """The main agent node. It calls the LLM to decide the next action."""
    print("---AGENT NODE---")
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Node 2: The Tool Executor
# This node runs the tool that the agent decided to use.
def tool_node(state):
    """Executes the tool chosen by the agent and returns the result."""
    print("---TOOL NODE---")
    last_message = state["messages"][-1]
    # We construct a ToolInvocation from the model's response
    tool_call = last_message.tool_calls[0]
    tool_output = tool_executor.invoke(tool_call)
    return {"messages": [HumanMessage(content=str(tool_output), name="tool_output")]}

# Node 3: The Guardrail (Responsible AI)
# This node checks the final output for safety and relevance.
def guardrail_node(state):
    """A final check on the agent's response before returning it to the user."""
    print("---GUARDRAIL NODE---")
    final_response = state["messages"][-1].content
    
    # Example guardrail: Check if the response is relevant or a refusal.
    # In a real application, this could be a more sophisticated check for toxicity, bias, etc.
    if "I cannot answer" in final_response or "not relevant" in final_response:
        print("Guardrail: Detected a refusal or irrelevant response.")
        # You could modify the response or take other actions here.
        # For now, we'll just let it pass but log it.
    
    return state # Pass the state through unchanged for now

# --- 4. DEFINE THE GRAPH EDGES ---
# Edges determine the flow of control between the nodes.

def should_continue(state):
    """Conditional edge: Decides whether to continue calling tools or finish."""
    last_message = state["messages"][-1]
    # If the last message has tool calls, we should continue.
    if last_message.tool_calls:
        return "continue"
    # Otherwise, we can finish.
    return "end"

# --- 5. CONSTRUCT THE GRAPH ---
# Now, we wire everything together.

# Define the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("guardrail", guardrail_node)

# Set the entry point
workflow.set_entry_point("agent")

# Add the conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": "guardrail", # If no tools are called, go to the guardrail
    },
)

# Add the normal edges
workflow.add_edge("tools", "agent") # After executing a tool, go back to the agent to reason about the result
workflow.add_edge("guardrail", END) # After the guardrail, end the process

# Compile the graph into a runnable app
app = workflow.compile()

# --- 6. RUN THE AGENT ---
if __name__ == "__main__":
    query = "What are the latest advancements in battery technology according to the web, and how does the 'attention' mechanism described in the PDF work?"
    
    print(f"--- Starting Agent with Query ---\nQuery: {query}\n")
    
    # Invoke the agent with the user's query
    final_state = app.invoke({"messages": [HumanMessage(content=query)]})
    
    # The final response is the last message in the state
    final_response = final_state["messages"][-1].content
    
    print("\n--- Agent Finished ---")
    print(f"Final Response:\n{final_response}")
