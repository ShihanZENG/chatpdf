import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Qdrant
import qdrant_client
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate

# Define templates for generating responses to questions
template_v1 = PromptTemplate(
    template="""Answer the questions based on the following document context. 
    Please quote as much of the document as possible and make sure your answer is concise. 
    Question: {question}""",
    input_variables=["context", "question"]
)

def get_pdf_text(pdf_docs):
    # Extract text from multiple PDF files
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + ' '
    return text.strip()

def get_text_chunks(raw_text):
    # Split long text into sizes suitable for processing
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

def get_vectorstore(text_chunks, use_qdrant=True):
    # Create or update a vector store based on a block of text
    embeddings = OpenAIEmbeddings()
    if use_qdrant:
        client = qdrant_client.QdrantClient(
            os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        vectorstore = Qdrant(
            client=client,
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            embeddings=embeddings,
        )
    return vectorstore

def get_conversation_chain(vectorstore):
    # Creating a session chain, combining LLM and a retriever
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, pdf_text):
    # Processes user input and displays responses via Streamlit
    full_prompt = template_v1.format(context=pdf_text, question=user_question)  # Generate complete prompts using document content and user questions
    response = st.session_state.conversation({'question': full_prompt})
    st.session_state.chat_history = response['chat_history']

    # 输出聊天历史
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    st.sidebar.header("Upload your PDFs")
    pdf_docs = st.sidebar.file_uploader("Upload your PDFs here", accept_multiple_files=True)
    if pdf_docs:
        pdf_text = get_pdf_text(pdf_docs)  # Get all the text content of the PDF, but do not display the
        st.header("Chat with multiple PDFs :book:")
        user_question = st.text_input("Ask a question about your documents:", key="user_question")
        if user_question:
            handle_userinput(user_question, pdf_text)
        if st.sidebar.button("Process", key="process_button"):
            with st.spinner("Processing..."):
                text_chunks = get_text_chunks(pdf_text)
                st.sidebar.write(text_chunks)
                vectorstore = get_vectorstore(text_chunks)
                st.success("You can ask questions now.")
                st.session_state.conversation = get_conversation_chain(vectorstore)
    else:
        st.warning("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()
