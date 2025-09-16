import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import anthropic
import cohere

# Show title and description.
st.title("# Nikita's ChatBot HW 3")

# Sidebar options for URLs
url1 = st.sidebar.text_input("URL 1:")
url2 = st.sidebar.text_input("URL 2:")

# Sidebar option for LLM selection
llm_vendor = st.sidebar.selectbox("Which Vendor?", ("OpenAI", "Anthropic", "Cohere"))

if llm_vendor == "OpenAI":
    openAI_model = st.sidebar.selectbox("Which Model?", ("gpt-4o", "gpt-4o-mini"))
    model = openAI_model
elif llm_vendor == "Anthropic":
    anthropic_model = st.sidebar.selectbox("Which Model?", ("claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"))
    model = anthropic_model
else:  # Cohere
    cohere_model = st.sidebar.selectbox("Which Model?", ("command-r-plus", "command-r"))
    model = cohere_model

# Sidebar option for memory type
memory_type = st.sidebar.selectbox("Memory Type:", ("buffer_6", "summary", "token_2000"))

# Function to get URL content
@st.cache_data
def get_url_content(url):
    if not url:
        return ""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text[:3000]  # Limit content
    except:
        return ""

# Get content from URLs
url1_content = get_url_content(url1)
url2_content = get_url_content(url2)

# Initialize clients
openai_api_key = st.secrets["OPENAI_API_KEY"]

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

if 'anthropic_client' not in st.session_state:
    try:
        st.session_state.anthropic_client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
    except:
        st.session_state.anthropic_client = None

if 'cohere_client' not in st.session_state:
    try:
        st.session_state.cohere_client = cohere.Client(api_key=st.secrets["COHERE_API_KEY"])
    except:
        st.session_state.cohere_client = None

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Function to manage memory
def manage_memory(messages, memory_type):
    if memory_type == "buffer_6":
        # Keep initial message + last 6 user messages and responses
        if len(messages) > 13:  # 1 initial + 6*2 messages
            return [messages[0]] + messages[-12:]
        return messages
    elif memory_type == "token_2000":
        # Simple token approximation - keep last messages under ~2000 tokens
        total_chars = 0
        kept_messages = [messages[0]]  # Keep initial message
        for msg in reversed(messages[1:]):
            total_chars += len(msg["content"])
            if total_chars > 8000:  # Rough token approximation
                break
            kept_messages.append(msg)
        return [messages[0]] + list(reversed(kept_messages[1:]))
    else:  # summary
        # Keep initial + last 4 messages, summarize middle
        if len(messages) > 8:
            return messages[:1] + [{"role": "system", "content": "Previous conversation summarized"}] + messages[-4:]
        return messages

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("What is up?"):
    # Add URL content to system message if available
    system_message = ""
    if url1_content:
        system_message += f"URL 1 content: {url1_content}\n\n"
    if url2_content:
        system_message += f"URL 2 content: {url2_content}\n\n"
    
    if system_message:
        system_message += "Answer questions based on this URL content."
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Manage conversation memory
    managed_messages = manage_memory(st.session_state.messages, memory_type)
    
    # Add system message with URL content
    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.extend(managed_messages)
    
    # Get streaming response based on vendor
    response = ""
    with st.chat_message("assistant"):
        if llm_vendor == "OpenAI":
            client = st.session_state.openai_client
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                stream=True
            )
            response = st.write_stream(stream)
        elif llm_vendor == "Anthropic":
            client = st.session_state.anthropic_client
            stream = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[msg for msg in api_messages if msg["role"] != "system"],
                system=next((msg["content"] for msg in api_messages if msg["role"] == "system"), ""),
                stream=True
            )
            response = st.write_stream(stream)
        else:  # Cohere
            client = st.session_state.cohere_client
            # Convert messages for Cohere format
            chat_history = []
            system_msg = ""
            for msg in api_messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    chat_history.append({"role": "USER", "message": msg["content"]})
                elif msg["role"] == "assistant":
                    chat_history.append({"role": "CHATBOT", "message": msg["content"]})
            
            stream = client.chat_stream(
                model=model,
                message=prompt,
                chat_history=chat_history[:-1],  # Exclude current message
                preamble=system_msg
            )
            response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})