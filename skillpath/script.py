import json
from langchain_huggingface import HuggingFaceEndpoint
import logging
import os
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HuggingFace endpoint
hf_token = os.environ.get('hf_token')
assert hf_token , 'Token not present'
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7, token=hf_token)

# Define the employees data
employees = [
    {"name": "John Doe", "age": 32, "skills": ["Python", "Java", "SQL"], "career_interests": ["Software Engineering", "Data Science", "Product Management"]},
    {"name": "Jane Smith", "age": 28, "skills": ["JavaScript", "React", "Node.js"], "career_interests": ["Front-end Development", "UX Design", "DevOps"]},
    {"name": "Mike Johnson", "age": 45, "skills": ["C++", "Machine Learning", "Data Science"], "career_interests": ["Artificial Intelligence", "Quantitative Analysis", "Business Analytics"]}
]

def generate_prompt():
    return f"""
    Human: Analyze the given employee data and create a JSON document summarizing each employee's profile. For each employee, provide:
    - name: Full name
    - skills: List of skills
    - career_interests: Career interests
    - path: Suggested single career path (consider age)
    - reason: Brief explanation for the path (don't assume gender in explanation)
    - learning_path: Structured learning plan as an array
    - confidence: Numerical score (1-10) for confidence

    Output: Raw JSON object without explanatory text.

    Employee data:
    {json.dumps(employees, indent=2)}

    Assistant:"""

try:
    response = llm.invoke(generate_prompt())
    
    try:
        result = json.loads(response)
        # Process the result here
        for employee in result:
            print(f"\nEmployee: {employee['name']}")
            print(f"Path: {employee['path']}")
            print(f"Reason: {employee['reason']}")
            print(f"Learning Path: {employee['learning_path']}")
            print(f"Confidence: {employee['confidence']}")
        
        logger.info("Result processed successfully")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON received. Error: {str(e)}")
        logger.error(f"Response was: {response[:200]}")
except Exception as e:
    logger.exception(f"An error occurred: {str(e)}")

logger.info("Script execution completed.")
