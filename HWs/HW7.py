import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from anthropic import Anthropic
import os
from typing import List, Tuple
import json

st.set_page_config(page_title="Legal News Intelligence Bot", layout="wide")

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def get_embeddings(texts: List[str], _client, model="text-embedding-3-small"):
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = _client.embeddings.create(input=batch, model=model)
        embeddings.extend([item.embedding for item in response.data])
    return np.array(embeddings)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_news(query: str, df: pd.DataFrame, embeddings: np.ndarray, 
                openai_client, top_k: int = 10) -> pd.DataFrame:
    query_embedding = np.array(openai_client.embeddings.create(
        input=[query], model="text-embedding-3-small"
    ).data[0].embedding)
    
    similarities = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = df.iloc[top_indices].copy()
    results['relevance_score'] = similarities[top_indices]
    return results

def rank_interesting_news(df: pd.DataFrame, llm_provider: str, api_key: str) -> Tuple[List[dict], str]:
    sample_news = df.sample(min(50, len(df))).to_dict('records')
    
    prompt = f"""You are an AI assistant for a large global law firm. Analyze these news stories and rank the top 10 most interesting/relevant for legal professionals.

Consider:
- Regulatory changes and compliance issues
- Litigation and legal disputes
- Corporate governance matters
- Financial crimes and enforcement
- M&A and corporate transactions
- Technology and data privacy laws
- International legal developments

News stories:
{json.dumps(sample_news, indent=2, default=str)}

Return ONLY a JSON array of the top 10 news items with this structure:
[{{"index": <original_index>, "company": "<company_name>", "headline": "<brief_headline>", "reasoning": "<why_interesting_for_lawyers>"}}]"""

    if llm_provider == "OpenAI":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        result = response.choices[0].message.content
    else:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=4000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.content[0].text
    
    start_idx = result.find('[')
    end_idx = result.rfind(']') + 1
    if start_idx != -1 and end_idx > start_idx:
        result = result[start_idx:end_idx]
    
    return json.loads(result), result

def query_specific_topic(query: str, context_news: pd.DataFrame, 
                        llm_provider: str, api_key: str) -> str:
    news_text = "\n\n".join([
        f"Company: {row['company_name']}\nDate: {row['Date']}\n{row['Document']}"
        for _, row in context_news.head(10).iterrows()
    ])
    
    prompt = f"""You are an AI assistant for a large global law firm. Answer this question based on the provided news context.

Question: {query}

Relevant News:
{news_text}

Provide a comprehensive legal analysis highlighting:
- Key legal implications
- Regulatory considerations
- Potential risks or opportunities
- Relevant precedents or trends

Answer:"""

    if llm_provider == "OpenAI":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    else:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2000,
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

st.title("🏛️ Legal News Intelligence Bot")
st.markdown("*Powered by RAG and Multi-LLM Analysis*")

with st.sidebar:
    st.header("Configuration")
    
    uploaded_file = st.file_uploader("Upload News CSV", type=['csv'])
    
    st.subheader("API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", 
                                value=os.getenv('OPENAI_API_KEY', ''))
    anthropic_key = st.text_input("Anthropic API Key", type="password",
                                   value=os.getenv('ANTHROPIC_API_KEY', ''))
    
    llm_provider = st.selectbox("Select LLM Provider", ["OpenAI", "Anthropic"])
    
    st.markdown("---")
    st.markdown("**Models Used:**")
    st.markdown("- OpenAI: GPT-4o-mini")
    st.markdown("- Anthropic: Claude 3.5 Haiku")

if uploaded_file and openai_key:
    df = load_data(uploaded_file)
    
    st.success(f"Loaded {len(df)} news articles from {df['company_name'].nunique()} companies")
    
    if 'embeddings' not in st.session_state:
        with st.spinner("Creating embeddings for RAG..."):
            openai_client = OpenAI(api_key=openai_key)
            st.session_state.embeddings = get_embeddings(
                df['Document'].tolist(), openai_client
            )
            st.session_state.df = df
    
    tab1, tab2, tab3 = st.tabs([
        "🔍 Search Specific Topic", 
        "⭐ Most Interesting News",
        "📊 Model Comparison"
    ])
    
    with tab1:
        st.header("Search News by Topic")
        search_query = st.text_input(
            "Enter your query (e.g., 'regulatory compliance', 'M&A activity', 'data privacy')",
            placeholder="Find news about..."
        )
        
        if search_query:
            with st.spinner("Searching..."):
                openai_client = OpenAI(api_key=openai_key)
                results = search_news(
                    search_query, 
                    st.session_state.df, 
                    st.session_state.embeddings,
                    openai_client
                )
                
                st.subheader(f"Top {len(results)} Results")
                for idx, row in results.iterrows():
                    with st.expander(
                        f"**{row['company_name']}** - {row['Date'].strftime('%Y-%m-%d')} "
                        f"(Relevance: {row['relevance_score']:.3f})"
                    ):
                        st.write(row['Document'])
                        st.markdown(f"[Read more]({row['URL']})")
                
                if llm_provider == "OpenAI" and openai_key:
                    api_key = openai_key
                elif llm_provider == "Anthropic" and anthropic_key:
                    api_key = anthropic_key
                else:
                    st.warning("Please provide API key for selected provider")
                    api_key = None
                
                if api_key:
                    with st.spinner(f"Generating analysis with {llm_provider}..."):
                        analysis = query_specific_topic(
                            search_query, results, llm_provider, api_key
                        )
                        st.subheader("Legal Analysis")
                        st.markdown(analysis)
    
    with tab2:
        st.header("Most Interesting News for Legal Professionals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Rank with OpenAI", disabled=not openai_key):
                with st.spinner("Analyzing with GPT-4o-mini..."):
                    rankings, _ = rank_interesting_news(df, "OpenAI", openai_key)
                    st.session_state.openai_rankings = rankings
        
        with col2:
            if st.button("Rank with Anthropic", disabled=not anthropic_key):
                with st.spinner("Analyzing with Claude 3.5 Haiku..."):
                    rankings, _ = rank_interesting_news(df, "Anthropic", anthropic_key)
                    st.session_state.anthropic_rankings = rankings
        
        if 'openai_rankings' in st.session_state:
            st.subheader("OpenAI Rankings")
            for i, item in enumerate(st.session_state.openai_rankings, 1):
                with st.expander(f"{i}. {item.get('company', 'N/A')} - {item.get('headline', 'N/A')}"):
                    st.write(f"**Reasoning:** {item.get('reasoning', 'N/A')}")
        
        if 'anthropic_rankings' in st.session_state:
            st.subheader("Anthropic Rankings")
            for i, item in enumerate(st.session_state.anthropic_rankings, 1):
                with st.expander(f"{i}. {item.get('company', 'N/A')} - {item.get('headline', 'N/A')}"):
                    st.write(f"**Reasoning:** {item.get('reasoning', 'N/A')}")
    
    with tab3:
        st.header("Model Comparison")
        st.markdown("""
        Compare how different LLMs rank and analyze the same news data.
        Run both models in the "Most Interesting News" tab to see differences.
        """)
        
        if 'openai_rankings' in st.session_state and 'anthropic_rankings' in st.session_state:
            st.subheader("Side-by-Side Comparison")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**OpenAI (GPT-4o-mini)**")
                for i, item in enumerate(st.session_state.openai_rankings[:5], 1):
                    st.write(f"{i}. {item.get('company', 'N/A')}")
            
            with col2:
                st.markdown("**Anthropic (Claude 3.5 Haiku)**")
                for i, item in enumerate(st.session_state.anthropic_rankings[:5], 1):
                    st.write(f"{i}. {item.get('company', 'N/A')}")
            
            st.info("""
            **Key Differences:**
            - Response time
            - Reasoning depth
            - Legal focus accuracy
            - Cost per request
            """)
        else:
            st.info("Generate rankings with both models to see comparison")

else:
    st.info("👈 Please upload a CSV file and provide at least OpenAI API key to begin")
    
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. Upload your news CSV file
        2. Enter API keys for OpenAI and/or Anthropic
        3. Choose your preferred LLM provider
        4. Use the tabs to:
           - Search for specific topics
           - Get ranked interesting news
           - Compare models
        """)