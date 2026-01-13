import streamlit as st
from rag_pipeline import rag_answer

st.set_page_config(page_title="CrediTrust Complaint Assistant")

st.title("CrediTrust Complaint Analysis Assistant")


question = st.text_input("Enter your question about customer complaints:")

col1, col2 = st.columns(2)

with col1:
    ask = st.button("Ask")

with col2:
    clear = st.button("Clear")

if clear:
    st.session_state["rerun_flag"] = True

if ask and question:
    with st.spinner("Analyzing complaints..."):
        answer, sources = rag_answer(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, src in enumerate(sources[:2], 1):
        st.markdown(
            f"**Source {i}**  \n"
            f"- Product: {src['product']}  \n"
            f"- Complaint ID: {src['complaint_id']}  \n"
            f"- Excerpt: {src['text'][:300]}..."
        )
