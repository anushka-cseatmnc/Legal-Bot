import streamlit as st

def message_box(sender, message):
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 LegalBot:** {message}")
