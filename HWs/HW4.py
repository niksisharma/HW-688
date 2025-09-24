import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import os
from collections import deque

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Show title and description
st.title("# HW4 - iSchool Student Organizations Chatbot")
st.markdown("Ask questions about iSchool student organizations and get informed answers!")

chromadb_path = "./ChromaDB_for_hw4"

chroma_client = chromadb.PersistentClient(path=chromadb_path)

# Sidebar for LLM selection
st.sidebar.title("🤖 LLM Selection")
llm_options = {
    "GPT-4o Mini": "gpt-4o-mini",
    "GPT-3.5 Turbo": "gpt-3.5-turbo", 
    "GPT-4": "gpt-4"
}

selected_llm_name = st.sidebar.selectbox(
    "Choose your LLM:",
    list(llm_options.keys()),
    index=0
)
selected_llm = llm_options[selected_llm_name]

st.sidebar.markdown(f"**Currently using:** {selected_llm_name}")
st.sidebar.markdown("---")
st.sidebar.markdown("**About this chatbot:**")
st.sidebar.markdown("This chatbot uses RAG (Retrieval Augmented Generation) to answer questions about iSchool student organizations based on HTML documents.")

# Initialize clients
openai_api_key = st.secrets["OPENAI_API_KEY"]

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

# Initialize chat history with memory buffer (store up to 5 Q&A pairs)
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = deque(maxlen=5)  # Store last 5 Q&A pairs

if "messages" not in st.session_state:
    st.session_state.messages = []

def extract_text_from_html(file_path):
    """Extract clean text from HTML file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            soup = BeautifulSoup(file, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
    except Exception as e:
        st.error(f"Error reading HTML file {file_path}: {e}")
        return ""

def chunk_document(text, filename, method="sentence_based"):
    """
    Chunk document into smaller pieces for better RAG performance.
    
    CHUNKING METHOD: Sentence-based chunking with size limits
    WHY THIS METHOD: 
    1. Maintains semantic coherence by keeping sentences intact
    2. Creates manageable chunk sizes (500-1000 chars) that fit well in context windows
    3. Allows for some overlap between chunks to maintain context
    4. Better than simple character splitting as it preserves meaning
    5. More practical than paragraph-based for HTML content which may have irregular structure
    """
    
    if not text.strip():
        return []
    
    sentences = text.split('. ')
    chunks = []
    
    # Create two chunks from each document as required
    mid_point = len(sentences) // 2
    
    # First chunk: first half of sentences
    first_half = '. '.join(sentences[:mid_point])
    if len(first_half) > 50:  # Only add if meaningful content
        chunks.append({
            'text': first_half,
            'chunk_id': f"{filename}_chunk_1",
            'source': filename
        })
    
    # Second chunk: second half of sentences  
    second_half = '. '.join(sentences[mid_point:])
    if len(second_half) > 50:  # Only add if meaningful content
        chunks.append({
            'text': second_half,
            'chunk_id': f"{filename}_chunk_2", 
            'source': filename
        })
    
    return chunks

def add_to_collection(collection, text, doc_id, metadata):
    """Add a document chunk to the ChromaDB collection with OpenAI embeddings"""
    openai_client = st.session_state.openai_client
    
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        
        embedding = response.data[0].embedding
        
        collection.add(
            documents=[text],
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        return True
    except Exception as e:
        st.error(f"Error adding document to collection: {e}")
        return False

def create_hw4_vectordb():
    """Create ChromaDB collection and populate with HTML documents"""
    
    # Only create vector DB if it doesn't exist
    vector_db_exists = False
    try:
        existing_collection = chroma_client.get_collection("HW4_iSchool_Collection")
        if existing_collection.count() > 0:
            vector_db_exists = True
            st.info("Vector database already exists. Using existing database.")
            return existing_collection
    except:
        pass  # Collection doesn't exist yet
    
    if vector_db_exists:
        return existing_collection
    
    try:
        collection = chroma_client.get_or_create_collection(
            name="HW4_iSchool_Collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        st.write("📁 Loading HTML files for iSchool organizations...")
        
        html_directory = "./html_files"  # Adjust path as needed
        
        html_files = []
        
        if os.path.exists(html_directory):
            files = [f for f in os.listdir(html_directory) if f.lower().endswith('.html')]
            if files:
                html_files = files
        
        if not html_files:
            st.error("No HTML files found. Please ensure HTML files are in the ./html_files directory.")
            return None
        
        st.write(f"Found {len(html_files)} HTML files")
        
        processed_chunks = 0
        total_files_processed = 0
        
        for html_filename in html_files:
            html_file_path = os.path.join(html_directory, html_filename)
            
            try:
                # Extract text from HTML
                text_content = extract_text_from_html(html_file_path)
                
                if text_content.strip():
                    # Chunk the document (creates 2 chunks as required)
                    chunks = chunk_document(text_content, html_filename)
                    
                    # Add each chunk to the collection
                    for chunk in chunks:
                        success = add_to_collection(
                            collection, 
                            chunk['text'], 
                            chunk['chunk_id'],
                            {
                                'filename': chunk['source'],
                                'chunk_number': chunk['chunk_id'].split('_')[-1]
                            }
                        )
                        if success:
                            processed_chunks += 1
                    
                    total_files_processed += 1
                    
            except Exception as e:
                st.error(f"Error processing {html_filename}: {e}")
        
        st.success(f"Successfully processed {total_files_processed} HTML files into {processed_chunks} chunks!")
        return collection
            
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return None

def search_vectordb(collection, query, top_k=3):
    """Search the vector database and return relevant document chunks"""
    if collection is None:
        return []
    
    try:
        openai_client = st.session_state.openai_client
        
        # Create embedding for search query
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        # Search the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        relevant_docs = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0
                similarity_score = 1 - distance
                
                relevant_docs.append({
                    'chunk_id': doc_id,
                    'content': document,
                    'filename': metadata.get('filename', 'Unknown'),
                    'chunk_number': metadata.get('chunk_number', '1'),
                    'similarity': similarity_score
                })
        
        return relevant_docs
            
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []

def generate_rag_response_with_memory(user_query, relevant_docs, selected_model):
    """Generate response using RAG with conversation memory"""
    openai_client = st.session_state.openai_client
    
    # Prepare context from retrieved documents
    context_parts = []
    source_info = []
    
    if relevant_docs:
        context_parts.append("Here is relevant information about iSchool student organizations:")
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"\n--- Source {i+1}: {doc['filename']} (chunk {doc['chunk_number']}) ---")
            context_parts.append(doc['content'][:1200])  # Limit content length
            source_info.append(f"• {doc['filename']} - chunk {doc['chunk_number']} (similarity: {doc['similarity']:.3f})")
    
    context = "\n".join(context_parts)
    
    # Prepare conversation history
    memory_context = ""
    if st.session_state.conversation_memory:
        memory_context = "\n\nRecent conversation history:\n"
        for qa_pair in st.session_state.conversation_memory:
            memory_context += f"Q: {qa_pair['question']}\nA: {qa_pair['answer']}\n\n"
    
    system_prompt = """You are a helpful AI assistant chatbot that specializes in answering questions about iSchool student organizations. 

IMPORTANT INSTRUCTIONS:
1. Focus specifically on iSchool student organizations, clubs, and related activities
2. Use the provided document context to give accurate, specific information
3. If the documents contain relevant information, clearly reference that you're using information from the knowledge base
4. If the documents don't contain relevant information, clearly state that you don't have that specific information
5. Keep responses conversational, helpful, and focused on student organizations
6. Consider the conversation history to provide contextual responses
7. Be encouraging about student involvement and organization participation
"""
    
    user_prompt = f"""User Question: {user_query}

{context if context else "No directly relevant documents found in the knowledge base for this specific query."}

{memory_context}

Please provide a helpful response focused on iSchool student organizations."""

    try:
        response = openai_client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1200,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        
        # Add source information if documents were used
        if relevant_docs and len(relevant_docs) > 0:
            assistant_response += f"\n\n📚 **Sources:**\n" + "\n".join(source_info)
        
        # Add to conversation memory
        st.session_state.conversation_memory.append({
            'question': user_query,
            'answer': assistant_response
        })
        
        return assistant_response
        
    except Exception as e:
        return f"Sorry, I encountered an error while generating a response: {e}"

def main():
    # Initialize vector database
    if 'HW4_vectorDB' not in st.session_state:
        st.write("Setting up vector database for iSchool organizations...")
        
        with st.spinner("Loading HTML documents into ChromaDB..."):
            collection = create_hw4_vectordb()
            if collection is not None:
                st.session_state.HW4_vectorDB = collection
                st.success("✅ Vector database ready!")
                st.rerun()
    else:
        st.markdown("## 💬 Chat about iSchool Student Organizations")
        st.markdown(f"Ask questions about iSchool student organizations! Currently using **{selected_llm_name}**.")
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about iSchool student organizations..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner(f"Searching documents and generating response using {selected_llm_name}..."):
                    relevant_docs = search_vectordb(st.session_state.HW4_vectorDB, prompt, top_k=4)
                    
                    # Generate RAG response with selected LLM
                    response = generate_rag_response_with_memory(prompt, relevant_docs, selected_llm)
                    
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Show conversation memory info in sidebar
        if st.session_state.conversation_memory:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Conversation Memory:**")
            st.sidebar.markdown(f"Storing {len(st.session_state.conversation_memory)}/5 recent Q&A pairs")

if __name__ == "__main__":
    main()
