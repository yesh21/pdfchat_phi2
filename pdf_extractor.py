from langchain.document_loaders import PyMuPDFLoader
import tempfile
import streamlit as st

from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def pretty_print_docs(docs):
    return f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)])
@staticmethod
def llm_model(model_path = 'phi-2.Q4_K_M.gguf'):
    model_path = 'phi-2.Q4_K_M.gguf'

    llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=40,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            #callback_manager=callback_manager,
            verbose=True,
    )
    return llm


class pdfExtractor:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        
        # output = llm(
        # "Instruct: Why is the sky blue? Output:", # Prompt
        # max_tokens=320, # Generate up to 32 tokens, set to None to generate up to the end of the context window
        # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
        #  echo=True # Echo the prompt back in the output
        # ) # Generate a completion, can also call create_completion
        # print(output)


    def pdf_retrival(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2048,
        chunk_overlap = 32
        )

        splits = text_splitter.split_documents(data)

        embedding = HuggingFaceEmbeddings()

        #persist_directory = './'
        vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        #persist_directory=persist_directory
        )

        # Wrap our vectorstore
        #llm = OpenAI(temperature=0)
        llm = llm_model()
        compressor = LLMChainExtractor.from_llm(llm)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectordb.as_retriever()
        )
        def response(prompt):
            #prompt = "Summarise the 2023 report?"
            compressed_docs = compression_retriever.get_relevant_documents(prompt)
            return compressed_docs
        # Function to display the chat input
        
        messages = st.container()
        if prompt := st.chat_input("Say something"):
                 #messages.chat_message("user").write(prompt)
                st.session_state.chat_history_prompt.append(prompt)
                #pdf_feed.response(prompt)
                #print(st.session_state.chat_history)

                for i, res in enumerate(st.session_state.chat_history_prompt):
                    messages.chat_message("user").write(res)
                    reply = response(prompt)[0].page_content
                    output = llm(
                    f"Instruct: You are an AI assistant for answering questions about the provided context. You are given the following extracted parts of a document. {reply} Provide a detailed answer. If you don't know the answer, just say Hmm, I'm not sure. some information about the question {prompt} Output:", # Prompt
                    max_tokens=5000, # Generate up to 512 tokens, set to None to generate up to the end of the context window
                    #stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
                     echo=True # Echo the prompt back in the output
                    ) # Generate a completion, can also call create_completion
                    st.session_state.chat_history_response.append(output)
                    messages.chat_message("assistant").write(st.session_state.chat_history_response[i])

        #messages.chat_message("assistant").write(f"assistant: {prompt}")