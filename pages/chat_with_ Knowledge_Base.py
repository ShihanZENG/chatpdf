from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os

def get_vector_store():
    """
    Creates and returns a Qdrant vector store connected to a Qdrant client.
    This store uses OpenAI embeddings to convert text into vectors.
    """
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    embeddings = OpenAIEmbeddings()  # Use OpenAI's embeddings for text processing.
    vector_store = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings,
    )
    return vector_store

def preprocess_question(question):
    """
    Processes the user's question by adding context or modifying phrasing
    to guide the model.
    """
    # Add specific guidance to contextualize the question.
    prompt = f"Considering the information in the provided PDF about historical events, {question}"
    return prompt

def main():
    # Load environment variables from a .env file.
    load_dotenv()

    # Set up the Streamlit web application.
    st.set_page_config(page_title="Ask Qdrant")
    st.header("Ask your remote database 💬")

    # Create the vector store for storing and retrieving information.
    vector_store = get_vector_store()

    # Create a RetrievalQA chain that connects the vector store to OpenAI.
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Take user input through a text box.
    user_question = st.text_input("Ask a question about your Knowledge Base:")
    if user_question:
        # Preprocess the question to add context or guidance.
        processed_question = preprocess_question(user_question)
        st.write(f"Question: {processed_question}")
        # Run the QA chain to get an answer.
        answer = qa.run(processed_question)
        st.write(f"Answer: {answer}")

if __name__ == '__main__':
    main()

