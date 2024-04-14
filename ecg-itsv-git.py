# Streamlit Application for ECG Intelligent Tutoring System
# Lynton Hazelhurst 14th April 2024
#***NB Put in OpenAI API NB***

import os
import streamlit as st

st.set_page_config(page_title="Chat with an Intelligent Tutoring System", page_icon="📃", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the ECG Intelligent Tutor")
st.info("This is an Electrocardiogram Intelligent Tutoring System", icon="💬")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about electrocardiograms (ECG)?"}
    ]
#***NB Put in OpenAI API NB***
os.environ["OPENAI_API_KEY"] = "sk-***"
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage

llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

Settings.llm = llm
Settings.embed_model = embed_model

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        storage_context = StorageContext.from_defaults(persist_dir="./")
        index = load_index_from_storage(storage_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
