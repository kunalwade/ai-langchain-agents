import os
import json
from openai import OpenAI

# --- 1. SETUP ---
# Initialize the OpenAI client, which will automatically read the API key
# from the OPENAI_API_KEY environment variable.
try:
    client = OpenAI()
except Exception as e:
    print("Error initializing OpenAI client. Make sure the OPENAI_API_KEY environment variable is set.")
    print(e)
    exit()

# --- 2. DEFINE INPUT DATA AND SCHEMA ---
# This is the unstructured text we want to process.
# In a real application, this could come from a file, a database, or an API call.

# Multiple test cases to demonstrate the extraction
unstructured_texts = [
    {
        "name": "Example 1: Detailed Job Posting",
        "text": """
Job Title: Senior Python Developer
Company: Tech Innovators Inc.
Location: San Francisco, CA (Remote available)
Posted: 2 days ago
Salary: $140,000 - $170,000 per year

Responsibilities:
- Design and implement scalable backend services.
- Work with product managers to define new features.
- Write clean, maintainable code and tests.
- Must have 5+ years of experience with Python and Django.
- Experience with cloud platforms like AWS is a plus.
"""
    },
    {
        "name": "Example 2: Brief Job Description",
        "text": """
We're hiring a Full Stack Developer at DataFlow Systems in Austin, Texas.
The position pays between $100k-$130k annually. Looking for someone with 3+ years experience.
This is an on-site position.
"""
    },
    {
        "name": "Example 3: Messy Format",
        "text": """
*** URGENT HIRING ***
DevOps Engineer needed! CloudScale Inc. - Remote OK
Based in Seattle, WA but work from anywhere
$$ 120000 to 150000 per year $$
Must have: 7 years of Kubernetes, Docker, AWS experience
Apply now!!!
"""
    },
    {
        "name": "Example 4: Minimal Information",
        "text": """
ML Engineer position at AI Labs, Boston. Salary 160k. Need 4 years experience minimum.
"""
    }
]

# This is the desired JSON structure for our output.
# Defining a schema helps the LLM understand exactly what we want.
json_schema = {
    "job_title": "string",
    "company_name": "string",
    "location": "string",
    "is_remote": "boolean",
    "min_salary": "integer",
    "max_salary": "integer",
    "required_experience_years": "integer"
}

# --- 3. BUILD THE PROMPT ---
# We will use a few-shot prompt to give the model a clear example.
def build_prompt(text_input, schema):
    # This is our one-shot example. It shows the model exactly how to behave.
    example_input = """
    Job Title: Frontend Engineer at WebCrafters LLC in New York. Salary: 120k. 3 years of React experience needed.
    """
    example_output = {
        "job_title": "Frontend Engineer",
        "company_name": "WebCrafters LLC",
        "location": "New York",
        "is_remote": False,
        "min_salary": 120000,
        "max_salary": 120000,
        "required_experience_years": 3
    }

    # The main prompt combines role-setting, instructions, the schema, the example, and the final input.
    prompt = f"""
    You are an expert data extraction AI. Your task is to extract information from the provided text and format it as a JSON object.
    
    The output MUST strictly follow this JSON schema:
    {json.dumps(schema, indent=2)}

    ---
    Here is an example:

    [TEXT]:
    {example_input}
    
    [JSON]:
    {json.dumps(example_output, indent=2)}
    ---

    Now, extract the information from the following text:

    [TEXT]:
    {text_input}

    [JSON]:
    """
    return prompt

# --- 4. DEFINE THE EXTRACTION FUNCTION ---
def extract_structured_data(text_input, schema):
    """
    Uses the OpenAI API to extract structured data from text based on a schema.
    """
    # Build the detailed prompt with instructions and examples.
    prompt = build_prompt(text_input, schema)

    print("--- Sending Prompt to LLM ---")
    print(prompt)
    print("-----------------------------")

    try:
        # Make the API call to the LLM.
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini (fast & affordable) or try "gpt-3.5-turbo"
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Set to 0 for deterministic, factual output
            response_format={"type": "json_object"} # Enforce JSON output
        )
        
        # Extract the content and parse it as JSON.
        json_output = response.choices[0].message.content
        return json.loads(json_output)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- 5. RUN THE SCRIPT ---
if __name__ == "__main__":
    print("=" * 80)
    print("STRUCTURED DATA EXTRACTION DEMO")
    print("=" * 80)
    
    # Process each example
    for i, example in enumerate(unstructured_texts, 1):
        print(f"\n{'='*80}")
        print(f"Processing {example['name']}")
        print(f"{'='*80}")
        print(f"\n📄 Input Text:")
        print(example['text'])
        
        extracted_data = extract_structured_data(example['text'], json_schema)

        if extracted_data:
            print("\n✅ Successfully Extracted JSON:")
            print(json.dumps(extracted_data, indent=2))
        else:
            print("\n❌ Failed to extract data")
        
        print("-" * 80)
        
        # Add a small pause between API calls to avoid rate limiting
        if i < len(unstructured_texts):
            import time
            time.sleep(1)
