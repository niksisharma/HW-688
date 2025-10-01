import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import json
import os
import glob
from bs4 import BeautifulSoup
import re

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Configuration
HTML_FOLDER = "html_files"
CACHE_FILE = "clubs_with_embeddings.csv"

def parse_html_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        club_info = {
            'file_name': os.path.basename(file_path),
            'club_name': '',
            'description': '',
            'full_text': ''
        }
        
        # Extract club name
        h1_tag = soup.find('h1')
        if h1_tag:
            club_info['club_name'] = h1_tag.get_text(strip=True)
        elif soup.find('title'):
            title_text = soup.find('title').get_text(strip=True)
            club_info['club_name'] = title_text.replace(" - 'Cuse Activities", "").strip()
        else:
            club_info['club_name'] = os.path.splitext(os.path.basename(file_path))[0]
        
        # Extract summary/description from script
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'window.initialAppState' in script.string:
                try:
                    summary_match = re.search(r'"summary":"([^"]*)"', script.string)
                    if summary_match:
                        club_info['description'] = summary_match.group(1).replace('\\r\\n', ' ').replace('\\n', ' ')
                except:
                    pass
        
        # Extract all text as fallback
        club_info['full_text'] = soup.get_text(separator=' ', strip=True)
        
        if not club_info['description']:
            words = club_info['full_text'].split()[:100]
            club_info['description'] = ' '.join(words)
        
        return club_info
    
    except Exception as e:
        return None

def load_clubs_from_html(folder_path):
    html_files = glob.glob(os.path.join(folder_path, "*.html")) + \
                 glob.glob(os.path.join(folder_path, "*.htm"))
    
    if not html_files:
        st.error(f"No HTML files found in {folder_path}")
        return None
    
    clubs_data = []
    for html_file in html_files:
        club_info = parse_html_file(html_file)
        if club_info:
            clubs_data.append(club_info)
    
    if not clubs_data:
        return None
    
    return pd.DataFrame(clubs_data)

def get_embedding(text, model="text-embedding-ada-002"):
    text = str(text).replace("\n", " ").strip()
    if not text:
        text = "empty"
    
    try:
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def generate_embeddings_for_dataframe(df):
    df['combined_text'] = (
        "Club Name: " + df['club_name'].astype(str) + ". " +
        "Description: " + df['description'].astype(str)
    )
    
    embeddings = []
    progress_bar = st.progress(0)
    
    for idx, text in enumerate(df['combined_text']):
        embedding = get_embedding(text)
        if embedding is None:
            embedding = np.zeros(1536)
        embeddings.append(embedding)
        progress_bar.progress((idx + 1) / len(df))
    
    progress_bar.empty()
    df['embedding'] = embeddings
    return df

def load_data():
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE)
        if 'embedding' in df.columns and isinstance(df['embedding'].iloc[0], str):
            df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)))
        return df
    else:
        df = load_clubs_from_html(HTML_FOLDER)
        if df is None:
            return None
        
        df = generate_embeddings_for_dataframe(df)
        
        df_save = df.copy()
        df_save['embedding'] = df_save['embedding'].apply(lambda x: json.dumps(x.tolist()))
        df_save.to_csv(CACHE_FILE, index=False)
        
        return df

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_relevant_info(query, df):
    query_embedding = get_embedding(query)
    
    if query_embedding is None:
        return "Error generating query embedding"
    
    # Calculate similarity scores
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x))
    
    # Get top 5 most similar clubs
    top_results = df.nlargest(5, 'similarity')
    
    # Format results
    formatted_text = "Here are the most relevant clubs:\n\n"
    
    for idx, row in top_results.iterrows():
        formatted_text += f"Club Name: {row.get('club_name', 'Unknown')}\n"
        formatted_text += f"Description: {row.get('description', 'No description')}\n"
        formatted_text += f"Similarity Score: {row['similarity']:.4f}\n"
        formatted_text += "-" * 80 + "\n\n"
    
    return formatted_text

def chat_with_context(user_message, conversation_history, df):
    # Get relevant information using vector search
    relevant_info = get_relevant_info(user_message, df)
    
    # Create system message with retrieved context
    system_message = {
        "role": "system",
        "content": f"""You are a helpful assistant that answers questions about student clubs at Syracuse University. 
        Use the following retrieved information to answer the user's question accurately and helpfully.
        
        RETRIEVED INFORMATION:
        {relevant_info}
        """
    }
    
    # Build messages with conversation history
    messages = [system_message] + conversation_history + [{"role": "user", "content": user_message}]
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error. Please try again."

def main():
    st.title("Nikita's Syracuse University Club Q&A Chatbot")
    st.markdown("Ask me anything about student clubs at Syracuse University!")
    
    # Load data
    if 'df' not in st.session_state:
        with st.spinner("Loading club data..."):
            st.session_state.df = load_data()
    
    if st.session_state.df is None:
        st.error(f"Unable to load data. Make sure '{HTML_FOLDER}' folder exists with HTML files.")
        return
    
    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask me about Syracuse clubs..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_context(
                    user_input, 
                    st.session_state.conversation_history[:-1],
                    st.session_state.df
                )
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()