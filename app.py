import streamlit as st
from pdf_extractor import pdfExtractor


# Title of the web app
st.title('PDF File and Text Input')



# Upload PDF file
pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
sidebar_button = st.sidebar.button("Send")

# if sidebar_button:
#     pdf_feed = pdfExtractor(pdf_file)
#     #print(pdf_feed.pdf_retrival())

# Initialize the chat history
if 'chat_history_prompt' not in st.session_state:
    st.session_state.chat_history_prompt = []

if 'chat_history_response' not in st.session_state:
    st.session_state.chat_history_response = []



if pdf_file is not None:
    # Process the uploaded PDF file
    pdf_text = pdfExtractor(pdf_file)
    pdf_text.pdf_retrival()
    st.sidebar.write("Text extracted from PDF:")
    st.sidebar.write(pdf_text)



