import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Load environment variables from .env file
load_dotenv()

# Verify that API key is set
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set. Please create a .env file and set your GROQ_API_KEY.")

# Initialize the LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# Create the prompt template with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create the chain
chain = prompt | llm

# Set up memory store
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create the chain with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Chat loop
print("AI Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "default"}}
    )
    print(f"AI: {response.content}")