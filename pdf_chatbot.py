import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- 1. SETUP ---
# Ensure the OpenAI API key is set
try:
    client = OpenAI()
except Exception as e:
    print("Error initializing OpenAI client. Make sure the OPENAI_API_KEY environment variable is set.")
    exit()

# --- 2. LOAD AND PROCESS THE DOCUMENT (RAG - Indexing) ---
def get_vectorstore_from_pdf(pdf_path):
    """Loads a PDF, splits it into chunks, creates embeddings, and stores them in a FAISS vector store."""
    print(f"Processing PDF: {pdf_path}")
    
    # Load the document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")

    # Create embeddings and store in FAISS
    # This will make an API call to OpenAI to get embeddings for each chunk.
    vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    print("Created vector store.")
    
    return vector_store

# --- 3. CREATE THE CONVERSATIONAL RAG CHAIN ---
def create_rag_chain(vector_store):
    """Creates the main RAG chain for conversation."""
    
    model = "gpt-4o-mini" # Or your preferred model

    # This first prompt is for a "history-aware retriever". It takes the latest user question
    # and the conversation history, and rephrases the question to be a standalone query.
    # This is crucial for effective retrieval.
    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    # Create the retriever chain
    retriever = vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        ChatOpenAI(model=model), retriever, retriever_prompt
    )

    # This second prompt is the main one that answers the question based on retrieved context.
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant for question-answering tasks. Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be concise.\n\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    # This chain takes the retrieved documents and "stuffs" them into the final prompt.
    question_answer_chain = create_stuff_documents_chain(ChatOpenAI(model=model), qa_prompt)
    
    # This is the final RAG chain. It combines the retriever and the QA chain.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- 4. IMPLEMENT THE CHAT LOOP ---
def run_chat_session(chain):
    """Starts an interactive chat session with the user."""
    
    # Initialize chat history
    chat_history = []
    
    print("\n--- Chat with your PDF ---")
    print("Ask questions about the document. Type 'exit' to end.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Session ended. Goodbye!")
            break
            
        # Invoke the RAG chain. This is where the magic happens.
        # LangChain handles the entire flow:
        # 1. Rephrase question based on history
        # 2. Retrieve relevant documents
        # 3. Stuff documents into the final prompt
        # 4. Generate the answer
        response = chain.invoke({
            "chat_history": chat_history,
            "input": user_input
        })
        
        print(f"\nAssistant: {response['answer']}")
        
        # Update the chat history for the next turn
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(HumanMessage(content=response["answer"]))


# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    pdf_file_path = "sample_document.pdf" # Make sure this file exists
    if not os.path.exists(pdf_file_path):
        print(f"Error: The file '{pdf_file_path}' was not found.")
    else:
        # 1. Process the PDF and create the vector store
        vector_store = get_vectorstore_from_pdf(pdf_file_path)
        
        # 2. Create the RAG chain
        rag_chain = create_rag_chain(vector_store)
        
        # 3. Start the chat session
        run_chat_session(rag_chain)
