import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tavily import TavilyClient
from langchain.schema.messages import AIMessage
from dotenv import load_dotenv
import os
import traceback


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily = TavilyClient(api_key=TAVILY_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="RAGX", page_icon="🤖", layout="wide")
st.title("🤖 AI Chat Application")

st.markdown(
    """
    <style>
    /* your existing styles here... */
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("RAGX")
    st.header("⚙️ Options")
    model = st.selectbox(
        "Supported Models",
        ["llama3-70b-8192","gemma2-9b-it", "llama-3.1-8b-instant",],
        help="Choose the underlying model that powers the assistant."
    )

    st.caption(f"🧠 Powered by: `{model}`")

    if st.button("🔄 Clear Conversation"):
        st.session_state.history = []

    if st.button("💾 Download Conversation"):
        import json
        st.download_button("Download", json.dumps(st.session_state.get("history", [])), file_name="conversation.json")

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, LangChain, Tavily, FAISS, and Groq.")


llm = ChatGroq(
    model=model,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY
)

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Based on the following context: {context}\n\nAnswer the question: {question}"
)


if "history" not in st.session_state:
    st.session_state.history = []

def get_response(query):
    try:
        
        prompt_text = (
            f"Answer the following question:\n{query}\n\n"
            "If you don't know the answer, respond with only: [NO_ANSWER]"
        )
        
        direct_response = llm.invoke(prompt_text)
        direct_answer = direct_response.content if isinstance(direct_response, AIMessage) else str(direct_response)
        direct_answer = direct_answer.strip()

        if direct_answer == "[NO_ANSWER]":
            
            search_results = tavily.search(query=query, search_depth="advanced")
            documents = [
                {"page_content": result["content"], "metadata": {"source": result["url"]}}
                for result in search_results["results"]
            ]

            texts = [doc["page_content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]

            vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            response = qa_chain({"query": query})

            answer = response["result"]
            sources = [doc.metadata.get("source", "Unknown") for doc in response["source_documents"]]
            return answer, sources

        else:
            
            return direct_answer, []

    except Exception as e:
        st.error(f"Error: {e}")
        st.error(traceback.format_exc())
        return "Sorry, something went wrong. Please try again.", []

query = st.chat_input("Ask a question...")

if query:
    query = query.strip()
    if query:
        st.session_state.history.append({"role": "user", "content": query})


for i, msg in enumerate(st.session_state.history):
    align = "flex-start" if msg["role"] == "assistant" else "flex-end"
    bubble_color = "#232429" if msg["role"] == "assistant" else "#232429"
    text_color = "#ffffff" if msg["role"] == "assistant" else "#e7e0e0"

    with st.container():
        st.markdown(
            f"""
            <div style='display: flex; justify-content: {align}; margin-bottom: 0.5rem;'>
                <div style='background-color: {bubble_color}; color: {text_color}; padding: 10px 15px; border-radius: 10px; max-width: 70%;'>
                    {msg["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.markdown(f"- [{source}]({source})")

    
    if i == len(st.session_state.history) - 1 and msg["role"] == "user" and query:
        with st.spinner("Thinking..."):
            answer, sources = get_response(query)
            st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
