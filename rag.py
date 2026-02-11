from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHUNK_SIZE = 600
COLLECTION_NAME = "real_estate"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"

llm = None
vector_store = None

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.9,
            max_tokens=500
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )

def process_urls(urls):
    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector store..."
    vector_store.reset_collection()

    yield "Loading data from URLs..."
    loader = WebBaseLoader(urls)
    documents = loader.load()

    yield "Splitting into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)
    yield "Storing in vector database..."
    ids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=ids)
    yield "Documents stored successfully in vector Database ...."

def generate_answer(query):
    if vector_store is None:
        raise RuntimeError("Vector database not initialized.")
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 15}
    )
    template = """
    You are a financial assistant.

    Answer ONLY from the provided context.
    If the answer is not present, say:
    "I don't know based on the provided articles."

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke(query)
    retrieved_docs = retriever.invoke(query)
    sources = list(
        set(
            doc.metadata.get("source")
            for doc in retrieved_docs
            if doc.metadata.get("source")
        )
    )
    return answer, sources
if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]
    process_urls(urls)
    question = "Tell me what was the 30 year fixed mortagate rate along with the date?"
    answer, sources = generate_answer(question)

    print("\n==============================")
    print("QUESTION:", question)
    print("\nANSWER:\n", answer)
    print("\nSOURCES:")
    for s in sources:
        print("-", s)
