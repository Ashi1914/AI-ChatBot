import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Load environment variables from .env file
load_dotenv()

# Verify that API key is set
if not os.environ.get("GROQ_API_KEY"):
    st.set_page_config(page_title="AI ChatBot - Error", page_icon="⚠️", layout="centered")
    st.error("⚠️ **GROQ_API_KEY is missing!** Please create a `.env` file in the project root directory and define `GROQ_API_KEY=your_api_key`.")
    st.info("You can duplicate `.env.example` as `.env` and fill in your API key.")
    st.stop()

st.set_page_config(page_title="AI ChatBot", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at left top, #6a8cff 0%, #c55cff 32%, #ff6eb4 100%);
    }
    .main .block-container {
        background: rgba(255, 255, 255, 0.94);
        border-radius: 28px;
        padding: 2rem 2.5rem 2.5rem;
        box-shadow: 0 30px 80px rgba(25, 28, 61, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.42);
    }
    .chat-bubble {
        border-radius: 28px;
        padding: 16px 20px;
        margin: 10px 0;
        max-width: 82%;
        line-height: 1.7;
        font-size: 1rem;
        word-wrap: break-word;
    }
    .chat-bubble.user {
        background: linear-gradient(135deg, #335dff 0%, #0f2ccf 100%);
        color: #ffffff;
        margin-left: auto;
        box-shadow: 0 18px 42px rgba(50, 90, 255, 0.22);
    }
    .chat-bubble.ai {
        background: linear-gradient(135deg, #f8faff 0%, #eef4ff 100%);
        color: #202840;
        margin-right: auto;
        border: 1px solid rgba(79, 115, 255, 0.2);
        box-shadow: 0 16px 40px rgba(91, 125, 255, 0.08);
    }
    .chat-meta {
        font-size: 0.83rem;
        opacity: 0.72;
        margin-bottom: 6px;
        letter-spacing: 0.02em;
    }
    .section-heading {
        color: #273268;
    }
    .section-subtitle {
        color: #50567b;
    }
    .stSidebar .stButton>button {
        background: linear-gradient(90deg, #4d7bff 0%, #8132ff 100%);
        color: #ffffff;
        border: none;
    }
    .stButton>button:hover {
        opacity: 0.95;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# AI ChatBot")
st.markdown("### A vibrant assistant with conversational memory powered by Groq.")
st.markdown("<div class='section-subtitle'>Type your message below and watch the chat history appear in beautifully styled bubbles.</div>", unsafe_allow_html=True)

@st.cache_resource
def build_chain():
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    chain = prompt | llm

    store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

chain_with_history = build_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.markdown("## Chat Settings")
    session_id = st.text_input("Session ID", value="default", help="Preserve separate conversations by using a custom session ID.")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #6178ff 0%, #d760ff 100%); padding: 18px; border-radius: 20px; color: white; margin-top: 16px;'>
            <strong>Pro tip:</strong> Use different Session IDs to keep separate chat threads.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='padding: 16px; border-radius: 18px; background: #f5f8ff; color: #2b3664;'>
            <strong>How it works:</strong>
            <ul style='margin: 8px 0 0 18px; padding: 0;'>
                <li>Enter a question</li>
                <li>Send it to the model</li>
                <li>View the AI response instantly</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    reply = getattr(response, "content", str(response))
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", reply))

for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f"<div class='chat-meta'>You</div><div class='chat-bubble user'>{text}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='chat-meta'>AI Assistant</div><div class='chat-bubble ai'>{text}</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption("Session history is preserved while Streamlit runs. Refresh to clear the UI.")
