import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="CBSE 6th Std English Chatbot",
    page_icon="📘",
    layout="centered"
)

st.title("📘 CBSE Class 6 – English RAG Chatbot")
st.write("Ask questions only from the CBSE 6th Std English Book")

# -------------------------------------------------
# Load PDF & Build RAG (cached)
# -------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_rag_pipeline():
    # 1. Load PDF
    from langchain_community.document_loaders import PyPDFLoader

    PDF_PATH = r"D:\LangChain\6th Std English Book 1 -2.pdf"
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # 2. Split text
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # 3. Embeddings
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # 4. Vector Store
    from langchain_community.vectorstores import FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. Prompt
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        """
You are a CBSE Class 6 English teacher.
Answer strictly using the context.
If the answer is not found, say:
"I don't know based on the lesson."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # 6. LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 7. RAG Chain (LCEL)
    from langchain_core.output_parsers import StrOutputParser

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = load_rag_pipeline()

# -------------------------------------------------
# Chat UI
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Ask a question from the lesson...")

if user_input:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(user_input)
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
