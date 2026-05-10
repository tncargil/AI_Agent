import time
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
MODEL_NAME = "deepseek-r1:14b"
EMBED_MODEL = "nomic-embed-text"
TARGET_DIR = os.path.expanduser("~/dotfiles/doom/.doom.d/")
DB_DIR = "./chroma_db"

print(f"--- Loading text files from {TARGET_DIR} ---")
loader = DirectoryLoader(
    TARGET_DIR,
    glob="**/*.el",
    loader_cls=lambda path: TextLoader(path, encoding="utf-8")
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

print(f"--- Creating/Loading Vector Database using {EMBED_MODEL} ---")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(model=EMBED_MODEL),
    persist_directory=DB_DIR
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model=MODEL_NAME, temperature=0)

template = """
You are a helpful assistant. Use the following pieces of retrieved context
to answer the question. If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def query(question):
    print(f"\nQuestion: {question}")
    start_time = time.perf_counter()

    response_text = ""
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
        response_text += chunk

    end_time = time.perf_counter()
    print(f"\n\n[Timed: {end_time - start_time:.4f}s]")

if __name__ == "__main__":

    user_task = input("what is your question regarding these files? ")
    query(user_task)
