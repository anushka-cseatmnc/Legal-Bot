import streamlit as st
import requests

# Page setup
st.set_page_config(page_title="Legal Bot", layout="wide")
st.title("⚖️ Legal Assistant Chatbot")

# Sidebar toggle for history
show_history = st.sidebar.checkbox("Show Chat History", value=False)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User query input
query = st.text_input("Ask your legal question here...")

# Button to trigger query
if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                res = requests.post("http://localhost:8000/query/", json={"question": query})
                answer = res.json().get("answer", "Something went wrong.")
            except Exception as e:
                answer = f"Error: {e}"

            # Save to session history
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Legal Bot", answer))
    else:
        st.warning("Please enter your question.")

# ✅ Chat History (only if user checks it)
if show_history and st.session_state.chat_history:
    st.sidebar.markdown("### 🗂️ Chat History")
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.sidebar.markdown(f"🧑 **You:** {msg}")
        else:
            st.sidebar.markdown(f"🤖 **Legal Bot:** {msg}")

# ✅ Display latest bot response in clean styled box
if st.session_state.chat_history:
    last_role, last_msg = st.session_state.chat_history[-1]
    if last_role == "Legal Bot":
        st.markdown("### 🧾 Bot's Latest Answer")
        st.markdown(
            f"""
            <div style="background-color:#f0f9ff; padding: 15px; border-radius: 10px; font-size: 17px; line-height: 1.6; color: #333;">
                {last_msg}
            </div>
            """,
            unsafe_allow_html=True
        )
