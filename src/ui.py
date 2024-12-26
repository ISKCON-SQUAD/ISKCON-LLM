import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import os

from src.app import State

def init_api_keys():
    # Load environment variables
    load_dotenv()
    KEYS_EXPECTED = [
        "OPEN_API_KEY",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT"
    ]
    for key in KEYS_EXPECTED:
        if key not in os.environ:
            st.error(f"Environment variable '{key}' is not set.")
            st.stop()
    # Return the OpenAI API key
    return os.getenv("OPEN_API_KEY")

# Create a Streamlit callback handler
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def start_ui(app):
    # Sidebar for conversation management
    with st.sidebar:
        st.header("Conversations")
        
        # Button to start a new conversation
        if st.button("New Conversation"):
            st.session_state.conversation_id = len(st.session_state.get("conversations", []))
            st.session_state.state = State(messages=[], context="")
            st.rerun()

        # Display existing conversations
        if "conversations" not in st.session_state:
            st.session_state.conversations = []

        for i, conv in enumerate(st.session_state.conversations):
            if st.button(f"Conversation {i + 1}", key=f"conv_{i}"):
                st.session_state.conversation_id = i
                st.session_state.state = conv
                st.rerun()

    # Initialize conversation state if not exists
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = 0

    if "state" not in st.session_state:
        st.session_state.state = State(messages=[], context="")

    # Display current conversation
    st.title(f"Bhagavad Gita RAG Chatbot - Conversation {st.session_state.conversation_id + 1}")

    for message in st.session_state.state["messages"]:
        with st.chat_message(message[0]):
            st.markdown(message[1])

    if prompt := st.chat_input("Ask your question about the Bhagavad Gita:"):
        st.session_state.state["messages"].append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream_container = st.empty()
            with stream_container:
                callback = StreamlitCallbackHandler(stream_container)

                new_state = app.invoke(
                    st.session_state.state,
                    {"callbacks": [callback]}
                )

        st.session_state.state = new_state

        # Update conversations list
        if len(st.session_state.conversations) <= st.session_state.conversation_id:
            st.session_state.conversations.append(st.session_state.state)
        else:
            st.session_state.conversations[st.session_state.conversation_id] = st.session_state.state