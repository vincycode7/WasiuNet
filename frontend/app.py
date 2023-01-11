import streamlit as st

# sidebar = st.s
st.date_input("pick a date to start safe entry point tracking")
st.error('This is an error', icon="🚨")
if not st.checkbox("proceed"):
    st.stop()
st.success('This is a success message!', icon="✅")
