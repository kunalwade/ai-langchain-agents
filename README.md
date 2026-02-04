# AI Upskill Projects

This repository contains three advanced AI projects demonstrating practical applications of OpenAI APIs, LangChain, and LangGraph. Each project showcases different capabilities of Large Language Models (LLMs) and modern AI frameworks.

## 📋 Table of Contents

- [Projects Overview](#projects-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Details](#project-details)
  - [1. PDF Chatbot](#1-pdf-chatbot)
  - [2. Research Agent](#2-research-agent)
  - [3. Structured Data Extractor](#3-structured-data-extractor)
- [Environment Setup](#environment-setup)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Projects Overview

| Project | Description | Key Technologies |
|---------|-------------|------------------|
| **PDF Chatbot** | Interactive chatbot for querying PDF documents using RAG | LangChain, FAISS, OpenAI |
| **Research Agent** | Autonomous agent with web search and PDF retrieval capabilities | LangGraph, Tavily, LangChain |
| **Structured Data Extractor** | Extract structured JSON data from unstructured text | OpenAI API, Few-shot Prompting |

---

## 📦 Prerequisites

- **Python**: Version 3.8 or higher
- **OpenAI API Key**: Required for all projects ([Get one here](https://platform.openai.com/api-keys))
- **Tavily API Key**: Required for the Research Agent ([Get one here](https://tavily.com/))

---

## 🔧 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd Upskill
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file or export the following environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export TAVILY_API_KEY="your-tavily-api-key-here"  # Only for Research Agent
```

---

## 📚 Project Details

### 1. PDF Chatbot

**File**: `pdf_chatbot.py`

#### Description
An interactive conversational chatbot that allows you to ask questions about PDF documents. It uses Retrieval-Augmented Generation (RAG) to find relevant information from the document and generate accurate answers with conversation history awareness.

#### Features
- ✅ PDF document processing and chunking
- ✅ Vector embeddings with FAISS for efficient retrieval
- ✅ Conversation history tracking
- ✅ Context-aware question reformulation
- ✅ Interactive command-line interface

#### Prerequisites
- A PDF file named `sample_document.pdf` in the project directory

#### How to Run
```bash
python pdf_chatbot.py
```

#### Usage Example
```
Processing PDF: sample_document.pdf
Split document into 42 chunks.
Created vector store.

--- Chat with your PDF ---
Ask questions about the document. Type 'exit' to end.

You: What is the main topic of this document?
Assistant: The document discusses machine learning fundamentals...

You: Can you explain the attention mechanism?
Assistant: Based on the context, the attention mechanism...

You: exit
Session ended. Goodbye!
```

#### Key Components
- **Vector Store**: Uses FAISS for fast similarity search
- **Text Splitting**: Chunks documents into 1000-character segments with 200-character overlap
- **Model**: Uses `gpt-4o-mini` for cost-effective responses
- **History-Aware Retrieval**: Reformulates questions based on conversation context

---

### 2. Research Agent

**File**: `research_agent.py`

#### Description
An autonomous AI agent built with LangGraph that can perform complex research tasks by combining web search (via Tavily) and PDF document retrieval. The agent automatically decides which tools to use based on the query and includes safety guardrails for responsible AI.

#### Features
- ✅ Web search integration with Tavily API
- ✅ PDF document retrieval (RAG)
- ✅ Multi-step reasoning with tool selection
- ✅ Autonomous decision-making
- ✅ Safety guardrails for output validation
- ✅ State graph architecture for transparent workflow

#### Prerequisites
- A PDF file named `sample_document.pdf` in the project directory
- Tavily API key set in environment variables

#### How to Run
```bash
python research_agent.py
```

#### Usage Example
The script runs with a pre-configured query:
```
Query: What are the latest advancements in battery technology according to the web, 
       and how does the 'attention' mechanism described in the PDF work?
```

To customize the query, edit the `query` variable in the `if __name__ == "__main__":` section:
```python
query = "Your custom research question here"
```

#### Workflow
```
User Query → Agent Node → Tool Selection → Tool Execution → 
Agent Reasoning → Guardrail Check → Final Response
```

#### Key Components
- **Agent Node**: LLM-powered decision maker using `gpt-4-turbo`
- **Tool Node**: Executes selected tools (web search or PDF retrieval)
- **Guardrail Node**: Validates responses for safety and relevance
- **State Graph**: Manages agent state and workflow
- **Tools**:
  - Tavily web search (returns top 3 results)
  - PDF retriever tool for document-specific queries

---

### 3. Structured Data Extractor

**File**: `structured_data_extractor.py`

#### Description
Extracts structured JSON data from unstructured text using few-shot prompting techniques. Ideal for parsing job postings, invoices, forms, or any unstructured text into a standardized JSON format.

#### Features
- ✅ Schema-based extraction
- ✅ Few-shot learning for accurate parsing
- ✅ Handles various input formats (messy, minimal, detailed)
- ✅ Enforced JSON output format
- ✅ Multiple test cases included
- ✅ Temperature 0 for deterministic results

#### Prerequisites
- OpenAI API key

#### How to Run
```bash
python structured_data_extractor.py
```

#### Output Schema
The script extracts the following fields:
```json
{
  "job_title": "string",
  "company_name": "string",
  "location": "string",
  "is_remote": "boolean",
  "min_salary": "integer",
  "max_salary": "integer",
  "required_experience_years": "integer"
}
```

#### Usage Example
**Input:**
```
Job Title: Senior Python Developer
Company: Tech Innovators Inc.
Location: San Francisco, CA (Remote available)
Salary: $140,000 - $170,000 per year
Must have 5+ years of experience with Python and Django.
```

**Output:**
```json
{
  "job_title": "Senior Python Developer",
  "company_name": "Tech Innovators Inc.",
  "location": "San Francisco, CA",
  "is_remote": true,
  "min_salary": 140000,
  "max_salary": 170000,
  "required_experience_years": 5
}
```

#### Customization
To use with different data types:
1. Modify the `json_schema` variable to match your desired output structure
2. Update the `example_input` and `example_output` in the prompt
3. Update the `unstructured_texts` list with your data

---

## 🌍 Environment Setup

### Required Environment Variables

#### For All Projects
```bash
export OPENAI_API_KEY="sk-..."
```

#### For Research Agent Only
```bash
export TAVILY_API_KEY="tvly-..."
```

### Setting Environment Variables

**Option 1: Using .env file** (recommended)
```bash
# Create a .env file
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
EOF

# Install python-dotenv
pip install python-dotenv

# Add to your script
from dotenv import load_dotenv
load_dotenv()
```

**Option 2: Export in terminal**
```bash
export OPENAI_API_KEY="your-key"
export TAVILY_API_KEY="your-key"
```

**Option 3: Add to shell profile** (Linux/Mac)
```bash
echo 'export OPENAI_API_KEY="your-key"' >> ~/.bashrc
echo 'export TAVILY_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

---

## 📦 Dependencies

All required packages are listed in `requirements.txt`:

```
openai                    # OpenAI API client
langchain>=1.1.0         # LangChain framework
langchain-openai         # OpenAI integrations for LangChain
langchain-community      # Community tools and integrations
langchain-text-splitters # Text chunking utilities
pypdf                    # PDF processing
faiss-cpu               # Vector similarity search
```

### Optional Dependencies
- `tavily-python`: For web search in Research Agent
- `langgraph`: For agent workflow (should be included with langchain)
- `python-dotenv`: For .env file support

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "OpenAI API key not found"
**Solution**: Ensure your API key is properly set in environment variables:
```bash
echo $OPENAI_API_KEY  # Should display your key
```

#### 2. "File 'sample_document.pdf' not found"
**Solution**: Add a PDF file named `sample_document.pdf` to the project directory, or modify the `pdf_file_path` variable in the script.

#### 3. "ModuleNotFoundError"
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

#### 4. FAISS installation issues on Windows
**Solution**: Install using conda instead:
```bash
conda install -c conda-forge faiss-cpu
```

#### 5. Rate limiting errors
**Solution**: 
- Reduce the frequency of API calls
- Add `time.sleep()` between calls
- Check your OpenAI account usage limits

#### 6. Memory errors with large PDFs
**Solution**: 
- Reduce `chunk_size` in text splitter
- Process PDFs in smaller sections
- Use `faiss-gpu` for better performance (if GPU available)

---

## 🎯 Use Cases

### PDF Chatbot
- Technical documentation Q&A
- Research paper analysis
- Legal document review
- Educational material comprehension

### Research Agent
- Multi-source research tasks
- Fact-checking with current data
- Comparative analysis (web vs. documents)
- Academic research assistance

### Structured Data Extractor
- Resume parsing
- Job posting standardization
- Invoice data extraction
- Form digitization
- Web scraping post-processing

---

## 🔑 Best Practices

1. **API Keys**: Never commit API keys to version control. Use `.gitignore` for `.env` files.
2. **Cost Management**: Monitor API usage, especially with GPT-4 models.
3. **Error Handling**: All scripts include basic error handling; enhance as needed.
4. **PDF Quality**: Use text-based PDFs for best results (not scanned images).
5. **Model Selection**: 
   - Use `gpt-4o-mini` for cost-effective tasks
   - Use `gpt-4-turbo` for complex reasoning
   - Use `gpt-3.5-turbo` for simple tasks

---

## 📝 License

This project is provided as-is for educational purposes.

---

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Add new examples
- Enhance documentation

---

## 📧 Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review OpenAI [documentation](https://platform.openai.com/docs)
3. Check LangChain [documentation](https://python.langchain.com/)

---

**Happy Coding! 🚀**
