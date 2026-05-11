
import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_URL = os.environ.get("API_URL", "http://localhost:8080/chat")
API_TOKEN = os.environ.get("API_TOKEN", "default-secret-token")
API_KEY_NAME = "X-API-Key"

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Enterprise Copilot (Onboarding de RH)", page_icon="🤖")

st.title("Enterprise Copilot (Onboarding de RH)")
st.markdown("Ask me anything about HR onboarding policies, benefits, and procedures.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare request to FastAPI backend
    headers = {
        API_KEY_NAME: API_TOKEN,
        "Content-Type": "application/json"
    }
    payload = {"question": prompt}

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                
                # Check if the request was successful
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "I received an empty response.")
                    st.markdown(answer)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                elif response.status_code == 401:
                    st.error("Authentication Error (401): The API Key is invalid or missing. Please check your configuration.")
                elif response.status_code == 400:
                     st.error(f"Bad Request (400): {response.json().get('detail', 'Invalid input.')}")
                elif response.status_code == 500:
                    st.error("Server Error (500): An internal error occurred in the Copilot engine. Please try again later.")
                else:
                    st.error(f"Unexpected Error ({response.status_code}): Could not retrieve an answer.")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Could not connect to the backend at {API_URL}. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
