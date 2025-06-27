import streamlit as st
import ollama
from load_rag import SimpleRAG

# Must be first Streamlit command
st.set_page_config(page_title="📚 RAG Chatbot", layout="centered")

# 🎨 Custom background and welcome message
st.markdown("""
    <style>
    body {
        background-color: #e6f2ff;
    }
    .stApp {
        background-image: linear-gradient(to bottom right, #dbeafe, #ffffff);
        padding: 2rem;
    }
    .header {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .subheader {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    </style>
    <div class="header">👋 Welcome to Your Help Chatbot</div>
    <div class="subheader">For Biomedical Engineering Students at Minya University</div>
""", unsafe_allow_html=True)

# Initialize RAG
@st.cache_resource
def load_rag():
    return SimpleRAG()

rag = load_rag()

# Main title and prompt
st.title("📚 Local RAG Chatbot")
st.markdown("Ask questions based on your local document database!")

# Question input
question = st.text_input("Enter your question:")

if st.button("🔍 Get Answer") and question:
    with st.spinner("Retrieving context and generating answer..."):
        contexts = rag.search(question)
        context_str = "\n\n".join(contexts[:3])

        prompt = f"""
Answer the following question using ONLY the provided context.
If the context does not contain relevant information, say so.

Context:
{context_str}

Question:
{question}
"""

        try:
            response = ollama.chat(
                model="llama3.2:1b-instruct-q2_K",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"]
        except Exception as e:
            answer = f"Error calling Ollama: {str(e)}"

    # Display final answer
    st.markdown("### 💬 Final Answer:")
    st.success(answer)
