import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import cohere
import json

# Show title and description.
st.title("📄 Nikita's Document Summarizer Lab 2")

# API Keys for OpenAI, Claude, Cohere
openai_api_key = st.secrets["OPENAI_API_KEY"]
claude_api_key = st.secrets["CLAUDE_API_KEY"]
cohere_api_key = st.secrets["COHERE_API_KEY"]

# Summary Type Selection
summary_type = st.selectbox(
    "Select Summary Type",
    ["100 Words", "2 Paragraphs", "5 Bullet Points"]
)

# Language Selection
language = st.sidebar.selectbox(
    "Select Language",
    ["English", "Spanish", "French", "German", "Chinese", "Kannada"]
)
st.sidebar.write(f"Language selected: {language}")

# LLM Model Selection
use_advanced_model = st.checkbox("Use Advanced Model (GPT-4)")
llm_type = st.selectbox(
    "Select LLM Model",
    ["OpenAI GPT-4", "Claude", "Cohere"]
)

if llm_type == "OpenAI GPT-4":
    model = "gpt-4" if use_advanced_model else "gpt-4.1-nano" 
elif llm_type == "Claude":
    model = "claude-3-sonnet-20240229" 
elif llm_type == "Cohere":
    model = "command"  

# Function to read content from URL
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Validate API keys
key_valid = False
if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        openai_client.models.list()  # Check API key validity for OpenAI
        st.success("OpenAI API key is valid ✅")
        key_valid = True
    except Exception as e:
        st.error(f"Invalid OpenAI API key. {e}")
else:
    st.info("No OpenAI API key", icon="🗝️")

# Allow user to input URL
url = st.text_input("Enter URL", value="https://example.com")

# Read the content of the URL
url_content = read_url_content(url)

# Show summary options and process accordingly
if key_valid and url_content:
    if summary_type == "100 Words":
        prompt = f"Summarize the following document in 100 words, in {language}: {url_content}"
    elif summary_type == "2 Paragraphs":
        prompt = f"Summarize the following document in 2 paragraphs, in {language}: {url_content}"
    else:
        prompt = f"Summarize the following document in 5 bullet points, in {language}: {url_content}"

    # Select LLM based on user input
    if llm_type == "OpenAI GPT-4":
        messages = [{"role": "user", "content": prompt}]
        try:
            # OpenAI GPT-4 model
            stream = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
        except Exception as e:
            st.error(f"Error using OpenAI GPT-4: {e}")

    elif llm_type == "Claude":
        try:
            headers = {
                "x-api-key": claude_api_key, 
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            data = {
                "model": model, 
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(
                "https://api.anthropic.com/v1/messages", 
                headers=headers, 
                json=data 
            )
            response.raise_for_status()
            result = response.json()
            st.write(result["content"][0]["text"]) 
        except Exception as e:
            st.error(f"Error using Claude API: {e}")

    elif llm_type == "Cohere":
        try:
            cohere_client = cohere.Client(cohere_api_key)
            response = cohere_client.generate(
                model=model,  
                prompt=prompt,
                max_tokens=500,
                temperature=0.7
            )
            st.write(response.generations[0].text)
        except Exception as e:
            st.error(f"Error using Cohere API: {e}")
else:
    st.info("No URL content to summarize or invalid API key.", icon="❗")
