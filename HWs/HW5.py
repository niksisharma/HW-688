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
        
        # Initialize club info
        club_info = {
            'file_name': os.path.basename(file_path),
            'club_name': '',
            'description': '',
            'summary': '',
            'category': '',
            'contact_email': '',
            'contact_name': '',
            'meeting_day': '',
            'meeting_time': '',
            'meeting_location': '',
            'website': '',
            'full_text': ''
        }
        
        # Extract club name from h1 or title
        h1_tag = soup.find('h1')
        if h1_tag:
            club_info['club_name'] = h1_tag.get_text(strip=True)
        elif soup.find('title'):
            title_text = soup.find('title').get_text(strip=True)
            club_info['club_name'] = title_text.replace(" - 'Cuse Activities", "").strip()
        
        if not club_info['club_name']:
            club_info['club_name'] = os.path.splitext(os.path.basename(file_path))[0]
        
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'window.initialAppState' in script.string:
                # Try to extract summary from the script
                try:
                    json_match = re.search(r'"summary":"([^"]*)"', script.string)
                    if json_match:
                        club_info['summary'] = json_match.group(1).replace('\\r\\n', ' ').replace('\\n', ' ')
                    
                    # Extract primary contact email
                    email_match = re.search(r'"primaryEmailAddress":"([^"]*)"', script.string)
                    if email_match:
                        club_info['contact_email'] = email_match.group(1)
                    
                    # Extract contact name
                    fname_match = re.search(r'"firstName":"([^"]*)"', script.string)
                    lname_match = re.search(r'"lastName":"([^"]*)"', script.string)
                    if fname_match and lname_match:
                        club_info['contact_name'] = f"{fname_match.group(1)} {lname_match.group(1)}"
                except:
                    pass
        
        # Extract description from "Additional Information" section
        all_divs = soup.find_all('div', style=lambda value: value and 'font-weight: bold' in value)
        for div in all_divs:
            label = div.get_text(strip=True).rstrip(':')
            next_div = div.find_next_sibling('div')
            if next_div:
                value = next_div.get_text(strip=True)
                if value and value != 'No Response':
                    if 'Description' in label:
                        club_info['description'] = value
                    elif 'Website' in label:
                        club_info['website'] = value
                    elif 'Meeting Day' in label:
                        club_info['meeting_day'] = value
                    elif 'Meeting time' in label.lower():
                        club_info['meeting_time'] = value
                    elif 'Meeting Location' in label:
                        club_info['meeting_location'] = value
        
        # Extract all text content as fallback
        club_info['full_text'] = soup.get_text(separator=' ', strip=True)
        
        # Use summary as description if description is empty
        if not club_info['description'] and club_info['summary']:
            club_info['description'] = club_info['summary']
        
        # If still no description, use a portion of full text
        if not club_info['description'] and club_info['full_text']:
            # Get first reasonable chunk of text
            words = club_info['full_text'].split()[:100]
            club_info['description'] = ' '.join(words)
        
        return club_info
    
    except Exception as e:
        st.error(f"Error parsing {file_path}: {e}")
        return None

def load_clubs_from_html(folder_path):
   
    html_files = glob.glob(os.path.join(folder_path, "*.html")) + \
                 glob.glob(os.path.join(folder_path, "*.htm"))
    
    if not html_files:
        st.error(f"No HTML files found in {folder_path}")
        return None
    
    clubs_data = []
    progress_bar = st.progress(0)
    st.info(f"Parsing {len(html_files)} HTML files...")
    
    for idx, html_file in enumerate(html_files):
        club_info = parse_html_file(html_file)
        if club_info:
            clubs_data.append(club_info)
        progress_bar.progress((idx + 1) / len(html_files))
    
    progress_bar.empty()
    
    if not clubs_data:
        st.error("No club data could be extracted from HTML files")
        return None
    
    df = pd.DataFrame(clubs_data)
    return df

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
    # Create combined text for embedding
    df['combined_text'] = (
        "Club Name: " + df['club_name'].astype(str) + ". " +
        "Description: " + df['description'].astype(str) + ". " +
        "Summary: " + df['summary'].astype(str)
    )
    
    if 'meeting_day' in df.columns:
        df['combined_text'] += ". Meeting Day: " + df['meeting_day'].astype(str)
    if 'meeting_time' in df.columns:
        df['combined_text'] += ". Meeting Time: " + df['meeting_time'].astype(str)
    if 'meeting_location' in df.columns:
        df['combined_text'] += ". Meeting Location: " + df['meeting_location'].astype(str)
    
    # Generate embeddings
    st.info("Generating embeddings... This may take a few minutes.")
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
    try:
        if os.path.exists(CACHE_FILE):
            st.info("Loading cached club data with embeddings...")
            df = pd.read_csv(CACHE_FILE)
            
            # Convert embedding strings to numpy arrays
            if 'embedding' in df.columns and isinstance(df['embedding'].iloc[0], str):
                df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)))
            
            return df
        else:
            st.info("Loading club data from HTML files...")
            df = load_clubs_from_html(HTML_FOLDER)
            
            if df is None:
                return None
            
            # Generate embeddings
            df = generate_embeddings_for_dataframe(df)
            
            # Save cache
            df_save = df.copy()
            df_save['embedding'] = df_save['embedding'].apply(lambda x: json.dumps(x.tolist()))
            df_save.to_csv(CACHE_FILE, index=False)
            
            st.success("Embeddings generated and cached!")
            return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_relevant_documents(query, df, top_k=5):
    query_embedding = get_embedding(query)
    
    if query_embedding is None:
        return "Error generating query embedding", pd.DataFrame()
    
    # Calculate similarity scores
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x))
    
    # Get top k most similar documents
    top_results = df.nlargest(top_k, 'similarity')
    
    formatted_text = "Here are the most relevant clubs based on the query:\n\n"
    
    for idx, row in top_results.iterrows():
        formatted_text += f"Club Name: {row.get('club_name', 'Unknown')}\n"
        
        if row.get('description') and row.get('description') not in ['No Response', '', 'nan']:
            formatted_text += f"Description: {row.get('description')}\n"
        elif row.get('summary') and row.get('summary') not in ['No Response', '', 'nan']:
            formatted_text += f"Summary: {row.get('summary')}\n"
        
        if row.get('contact_email') and row.get('contact_email') not in ['No Response', '', 'nan']:
            formatted_text += f"Contact: {row.get('contact_email')}\n"
        
        if row.get('contact_name') and row.get('contact_name') not in ['No Response', '', 'nan']:
            formatted_text += f"Contact Person: {row.get('contact_name')}\n"
        
        if row.get('meeting_day') and row.get('meeting_day') not in ['No Response', '', 'nan']:
            formatted_text += f"Meeting Day: {row.get('meeting_day')}\n"
        
        if row.get('meeting_time') and row.get('meeting_time') not in ['No Response', '', 'nan']:
            formatted_text += f"Meeting Time: {row.get('meeting_time')}\n"
        
        if row.get('meeting_location') and row.get('meeting_location') not in ['No Response', '', 'nan']:
            formatted_text += f"Meeting Location: {row.get('meeting_location')}\n"
        
        if row.get('website') and row.get('website') not in ['No Response', '', 'nan']:
            formatted_text += f"Website: {row.get('website')}\n"
        
        formatted_text += f"Similarity Score: {row['similarity']:.4f}\n"
        formatted_text += "-" * 80 + "\n\n"
    
    return formatted_text, top_results

def get_relevant_info(query, df):
    relevant_info, _ = search_relevant_documents(query, df, top_k=5)
    return relevant_info

def chat_with_context(user_message, conversation_history, df):
   # Get relevant information using vector search
    relevant_info = get_relevant_info(user_message, df)
    
    # Add relevant info to system prompt
    system_message = {
        "role": "system",
        "content": f"""You are a helpful assistant that answers questions about student clubs at Syracuse University. 
        Use the following retrieved information to answer the user's question accurately and helpfully.
        If the information provided doesn't fully answer the question, say so and provide the best answer you can.
        Be conversational and friendly. When mentioning clubs, include relevant details like meeting times, locations, and contact information when available.
        
        RETRIEVED INFORMATION:
        {relevant_info}
        """
    }
    
    # Build messages list with conversation history
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
        return "I apologize, but I encountered an error while generating a response. Please try again."

def chat_with_function_calling(user_message, conversation_history, df):
    # Define the search function for OpenAI
    search_function = {
        "type": "function",
        "function": {
            "name": "search_relevant_clubs",
            "description": "Search through Syracuse University clubs database to find relevant clubs based on the query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant clubs"
                    }
                },
                "required": ["query"]
            }
        }
    }
    
    system_message = {
        "role": "system",
        "content": """You are a helpful assistant that answers questions about student clubs at Syracuse University. 
        When you need information about clubs, use the search_relevant_clubs function to retrieve relevant data.
        Always base your answers on the retrieved information. Be friendly and provide helpful details."""
    }
    
    messages = [system_message] + conversation_history + [{"role": "user", "content": user_message}]
    
    try:
        # First API call with function calling
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=[search_function],
            tool_choice="auto"
        )
        
        # Check if function was called
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            
            # Execute the search function
            relevant_info = get_relevant_info(user_message, df)
            
            # Add function result to messages
            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "content": relevant_info,
                "tool_call_id": tool_call.id
            })
            
            # Second API call with the function result
            final_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            
            return final_response.choices[0].message.content
        else:
            return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error in function calling approach: {e}")
        return "I apologize, but I encountered an error. Please try again."

def main():
    st.title("🎓 Syracuse University Club Q&A Chatbot")
    st.markdown("Ask me anything about student clubs at Syracuse University!")
    
    # Load data
    if 'df' not in st.session_state:
        with st.spinner("Loading club data and embeddings..."):
            st.session_state.df = load_data()
    
    if st.session_state.df is None:
        st.error("Unable to load data. Please check your data files.")
        st.info(f"Make sure the '{HTML_FOLDER}' folder exists and contains HTML files.")
        return
    
    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    approach = st.sidebar.radio(
        "Select RAG Approach:",
        ["System Prompt (Approach 1)", "Function Calling (Approach 2)"],
        help="Choose how to inject retrieved context into the LLM"
    )
    
    top_k = st.sidebar.slider(
        "Number of clubs to retrieve:",
        min_value=1,
        max_value=10,
        value=5,
        help="How many similar clubs to retrieve for context"
    )
    
    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.messages = []
        st.rerun()
    
    if st.sidebar.button("🔄 Regenerate Embeddings"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.session_state.df = None
        st.rerun()
    
    # Display stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Stats")
    st.sidebar.metric("Total Messages", len(st.session_state.messages))
    st.sidebar.metric("Total Clubs", len(st.session_state.df))
    st.sidebar.info(f"📁 Data source: {HTML_FOLDER}/")
    
    # Example queries
    with st.sidebar.expander("💡 Example Questions"):
        st.markdown("""
        - What technology clubs are available?
        - Tell me about clubs focused on entrepreneurship
        - Which clubs are good for networking?
        - What clubs meet on Tuesdays?
        - Show me clubs related to innovation
        - Are there any clubs for graduate students?
        - What clubs focus on community service?
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask me about Syracuse clubs..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add to message history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching clubs and thinking..."):
                if "Approach 1" in approach:
                    response = chat_with_context(
                        user_input, 
                        st.session_state.conversation_history[:-1],
                        st.session_state.df
                    )
                else:
                    response = chat_with_function_calling(
                        user_input,
                        st.session_state.conversation_history[:-1],
                        st.session_state.df
                    )
                
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
    
    # Show retrieved clubs in expander
    if st.session_state.messages:
        with st.expander("🔍 View Retrieved Clubs"):
            if len(st.session_state.messages) > 0:
                last_user_message = None
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "user":
                        last_user_message = msg["content"]
                        break
                
                if last_user_message:
                    _, top_results = search_relevant_documents(
                        last_user_message, 
                        st.session_state.df, 
                        top_k=top_k
                    )
                    
                    for idx, row in top_results.iterrows():
                        st.markdown(f"**Similarity: {row['similarity']:.4f}**")
                        st.markdown(f"### {row.get('club_name', 'Unknown Club')}")
                        
                        if row.get('description'):
                            st.markdown(f"*{row.get('description')}*")
                        elif row.get('summary'):
                            st.markdown(f"*{row.get('summary')}*")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if row.get('contact_email'):
                                st.markdown(f"📧 {row.get('contact_email')}")
                            if row.get('meeting_day'):
                                st.markdown(f"📅 {row.get('meeting_day')}")
                        with col2:
                            if row.get('meeting_time'):
                                st.markdown(f"⏰ {row.get('meeting_time')}")
                            if row.get('meeting_location'):
                                st.markdown(f"📍 {row.get('meeting_location')}")
                        
                        st.markdown("---")

if __name__ == "__main__":
    main()