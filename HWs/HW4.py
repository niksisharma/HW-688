import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import PyPDF2
import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Show title and description.
st.title("# Nikita's Lab 4 - RAG Chatbot")

chromadb_path = "./ChromaDB_for_lab"

chroma_client = chromadb.PersistentClient(path=chromadb_path)

# Initialize clients
openai_api_key = st.secrets["OPENAI_API_KEY"]

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def add_to_collection(collection, text, filename):
    """Add a document to the ChromaDB collection with OpenAI embeddings"""
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding],
        metadatas=[{"filename": filename}]
    )

def extract_text_from_pdf_file(file_obj):
    """Extract text from PDF file object"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def create_lab4_vectordb():
    """Create ChromaDB collection and populate with PDF documents from local directory"""
    try:
        collection = chroma_client.get_or_create_collection(
            name="Lab4Collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        st.write("📁 Loading PDF files from repository...")
        
        pdf_directory = "./pdfs"
        
        pdf_files = []
        pdf_path = None
        
        if os.path.exists(pdf_directory):
            files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
            if files:
                pdf_files = files
                pdf_path = pdf_directory
        
        if not pdf_files:
            st.error("No PDF files found in the repository. Please check the file paths.")
            return None
        
        st.write(f"Found {len(pdf_files)} PDF files in: `{pdf_path}`")
        
        processed_count = 0
        for pdf_filename in pdf_files:
            pdf_file_path = os.path.join(pdf_path, pdf_filename)
            
            try:
                with open(pdf_file_path, 'rb') as file:
                    text_content = extract_text_from_pdf_file(file)
                
                if text_content.strip(): 
                    # Add to collection
                    add_to_collection(collection, text_content, pdf_filename)
                    processed_count += 1
                    
            except Exception as e:
                st.error(f"Error processing {pdf_filename}: {e}")
        
        st.success(f"Successfully processed {processed_count}/{len(pdf_files)} PDF files!")
        return collection
            
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return None

def search_vectordb(collection, query, top_k=3):
    """Search the vector database and return relevant documents"""
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
                distance = results['distances'][0][i] if 'distances' in results else 0
                similarity_score = 1 - distance
                
                relevant_docs.append({
                    'filename': doc_id,
                    'content': document,
                    'similarity': similarity_score
                })
        
        return relevant_docs
            
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []

def generate_rag_response(user_query, relevant_docs):
    """Generate response using RAG - combine retrieved documents with LLM"""
    openai_client = st.session_state.openai_client
    
    context_parts = []
    source_info = []
    
    if relevant_docs:
        context_parts.append("Here is relevant information from the knowledge base:")
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"\n--- Document {i+1}: {doc['filename']} ---")
            context_parts.append(doc['content'][:1500])  # Limit content length
            source_info.append(f"• {doc['filename']} (similarity: {doc['similarity']:.3f})")
    
    context = "\n".join(context_parts)
    
    system_prompt = """You are a helpful AI assistant chatbot that answers questions based on the provided documents. 

IMPORTANT INSTRUCTIONS:
1. If you use information from the provided documents, clearly state that you are using knowledge from the document(s)
2. If the provided documents don't contain relevant information for the user's question, clearly state that you don't have that information in the knowledge base
3. Always be clear about whether your response is based on the retrieved documents or your general knowledge
4. Keep your responses conversational and helpful
"""
    
    user_prompt = f"""User Question: {user_query}

{context if context else "No relevant documents found in the knowledge base for this query."}

Please provide a helpful response to the user's question."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000
        )
        
        assistant_response = response.choices[0].message.content
        
        # Add source information if documents were used
        if relevant_docs and len(relevant_docs) > 0:
            assistant_response += f"\n\n📚 **Sources consulted:**\n" + "\n".join(source_info)
        
        return assistant_response
        
    except Exception as e:
        return f"Sorry, I encountered an error while generating a response: {e}"

def main():
    # Initialize vector database
    if 'Lab4_vectorDB' not in st.session_state:
        st.write("Setting up vector database...")
        
        with st.spinner("Loading documents into ChromaDB..."):
            collection = create_lab4_vectordb()
            if collection is not None:
                st.session_state.Lab4_vectorDB = collection
                st.success("✅ Vector database ready!")
                st.rerun() 
    else:
        st.markdown("## 💬 Chat with your Documents")
        st.markdown("Ask questions about the documents in your knowledge base!")
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about the documents"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating response..."):
                    relevant_docs = search_vectordb(st.session_state.Lab4_vectorDB, prompt, top_k=3)
                    
                    # Generate RAG response
                    response = generate_rag_response(prompt, relevant_docs)
                    
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        

if __name__ == "__main__":
    main()
