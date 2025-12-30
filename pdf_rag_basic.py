from dotenv import load_dotenv
load_dotenv()

# ------------------------------------
# STEP 1: Load PDF
# ------------------------------------
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = r"D:\LangChain\6th Std English Book 1 -2.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages from PDF")

# ------------------------------------
# STEP 2: Split Text
# ------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks")

# ------------------------------------
# STEP 3: Create Embeddings
# ------------------------------------
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# ------------------------------------
# STEP 4: Vector Store (FAISS)
# ------------------------------------
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Vector store created successfully")

# ------------------------------------
# STEP 5: Prompt Template
# ------------------------------------
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
You are a CBSE Class 6 English teacher.
Answer the question strictly using the context.
If the answer is not found in the lesson, say:
"I don't know based on the lesson."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ------------------------------------
# STEP 6: LLM
# ------------------------------------
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# ------------------------------------
# STEP 7: Build RAG Chain (LCEL)
# ------------------------------------
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

# ------------------------------------
# STEP 8: Chatbot Loop
# ------------------------------------
print("\nCBSE 6th Std English RAG Chatbot Ready")
print("Ask questions from the book. Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        print("Chatbot closed.")
        break

    answer = rag_chain.invoke(question)

    print("\nBot:")
    print(answer)
    print("-" * 60)
