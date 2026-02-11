import streamlit as st
from rag import process_urls, generate_answer
st.title("RealEstate Research Tool")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")
placeholder = st.empty()
process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != ""]
    if len(urls) == 0:
        placeholder.warning("You must provide at least one URL to process")
    else:
        for status in process_urls(urls):
            placeholder.info(status)
        placeholder.success("URLs processed successfully ✅")
st.divider()
query = st.text_input("Enter your question")
submit_question = st.button("Submit Question")
if submit_question:
    if not query:
        st.warning("Please enter a question.")
    else:
        try:
            answer, sources = generate_answer(query)
            st.header("Answer:")
            st.write(answer)
            if sources:
                st.subheader("Sources:")
                for source in sources:
                    st.write(source)
        except RuntimeError:
            st.error("You must process URLs first.")
