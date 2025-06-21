
# agentic_ai_workflow.py

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.fake import FakeListLLM

# Step 1: Display title
st.title("🧠 Agentic AI Mini Agent (No API Keys!)")

# Step 2: User input (goal)
goal = st.text_input("Enter your goal/question:", placeholder="e.g., What is Agentic AI?")

# Step 3: Simulated Agent Planning
def agent_plan(goal):
    if "what" in goal.lower():
        return ["Understand goal", "Search document", "Answer based on context"]
    else:
        return ["Read input", "Try to match content", "Respond with info"]

# Step 4: Load & prepare document
loader = TextLoader("my_documents.txt")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(splits, embeddings)

# Step 5: Fake LLM response (no API)
llm = FakeListLLM(responses=[
    "Agentic AI is about systems that make decisions and take actions toward a goal, often using memory and tools."
])
chain = load_qa_chain(llm, chain_type="stuff")

# Step 6: Run the agent
if goal:
    with st.spinner("Agent is planning and thinking..."):
        plan = agent_plan(goal)
        st.subheader("🛠️ Agent Plan:")
        for step in plan:
            st.write(f"• {step}")

        # Agent performs search
        relevant_docs = db.similarity_search(goal)

        # Agent generates response
        result = chain.run(input_documents=relevant_docs, question=goal)

        st.success("✅ Agent Response:")
        st.write(result)
