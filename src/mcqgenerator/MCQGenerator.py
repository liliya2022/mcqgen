import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

# Import necessary packages from updated LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environ
key = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI model
llm = ChatOpenAI(openai_api_key=key, model_name="gpt-3.5-turbo", temperature=0.7)

# Prompt template for quiz generation
quiz_template = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs.
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=quiz_template
)

# Quiz evaluation prompt template
evaluation_template = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students, \
evaluate the complexity of the question and provide a complete analysis of the quiz in 50 words or fewer. \
If the quiz is not appropriate for the cognitive and analytical abilities of the students, \
update the quiz questions as needed, changing the tone to fit the students' abilities. \
Quiz_MCQs:
{quiz}

Expert review of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=evaluation_template
)

# Define the generate evaluate chain
def generate_evaluate_chain(input_dict):
    # Create the quiz generation chain
    quiz_generation_chain = (
        RunnablePassthrough.assign(
            quiz=quiz_generation_prompt | llm | StrOutputParser()
        )
    )

    # Create the quiz evaluation chain
    quiz_evaluation_chain = (
        RunnablePassthrough.assign(
            review=quiz_evaluation_prompt | llm | StrOutputParser()
        )
    )

    # Combine the chains
    full_chain = quiz_generation_chain | quiz_evaluation_chain

    # Run the chain
    result = full_chain.invoke(input_dict)
    
    return {
        "quiz": result['quiz'],
        "review": result['review']
    }
