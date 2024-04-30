import streamlit as st

st.set_page_config(
    page_title="Web Teach Tool(CSS, html, javascript)",
    page_icon="",
)

st.title("Build a ChatGPT type application that teaches students how to use HTML、CSS、JavaScript to create web pages and use multiple PDF files as a knowledge base")
st.markdown("""
Here is a guide to the page setup: adjacent to the page setup is the multi-page feature.

On the second page, students are granted the flexibility to upload any number of PDFs. However, please note that the model will only respond to the content that has already been uploaded.

The third page facilitates interaction with the pre-imported knowledge base. The contents of the imported knowledge base include HTML, CSS, and JavaScript.
""")

st.header('knowledge base')
st.markdown('[knowledge base PDF]https://drive.google.com/drive/folders/1S2Va_4vTLB1uZ9NFZs_QhlSd9fGk1Mig?usp=sharing')
st.header('Link to the code for uploading files')
st.markdown('https://colab.research.google.com/drive/1z1uCOtmjRxfXJdNEIMpH5beFpvF5f75_')
st.header('Link to the github repository')
st.markdown('https://github.com/ShihanZENG/chatpdf.git')
# https://drive.google.com/drive/folders/1S2Va_4vTLB1uZ9NFZs_QhlSd9fGk1Mig?usp=sharing
st.sidebar.success("Select a page above.")
