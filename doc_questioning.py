
# agentic_ai_workflow.py
import streamlit as st
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms.fake import FakeListLLM

st.title("📄 Question Answering App (CPU only)")

query = st.text_input("Ask your question:")

# Load and split documents
loader = TextLoader("document.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# Embeddings & FAISS index
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(splits, embeddings)

# Fake LLM for response (no OpenAI API)
llm = FakeListLLM(responses=["Here's your answer from the document."])
chain = load_qa_chain(llm, chain_type="stuff")

if query:
    docs = db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    st.write("### 💬 Answer:")
    st.write(response)
