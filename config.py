import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
def load_environment():
    load_dotenv()
  
    os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Langsmith tracking

# Initialize the language model
llm = ChatGroq(groq_api_key=st.secrets["GROQ_API_KEY"], model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most detailed and accurate response based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)
