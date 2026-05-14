import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 RAG Document Chatbot")

# -----------------------------
# Sidebar：上傳文件 + 清除聊天
# -----------------------------
from pathlib import Path

st.sidebar.title("📂 Document Control")

uploaded_file = st.sidebar.file_uploader(
    "Upload .md / .pdf / .txt",
    type=["md", "pdf", "txt"]
)

available_files = [
    f.name
    for f in Path("data/uploads").glob("*")
]

selected_file = st.sidebar.selectbox(
    "Select document (optional)",
    ["All Documents"] + available_files
)

# 👉 清除聊天按鈕（已整合）
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# 上傳文件
if uploaded_file:
    with st.sidebar.spinner("Uploading and indexing..."):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue())
        }

        try:
            res = requests.post(f"{API_URL}/upload", files=files)

            if res.status_code == 200:
                st.sidebar.success("✅ Uploaded & indexed!")
            else:
                st.sidebar.error(f"❌ Upload failed: {res.text}")

        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# -----------------------------
# Chat UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 使用者輸入
if prompt := st.chat_input("Ask something about your documents..."):

    # 顯示 user 訊息
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.write(prompt)

    # 呼叫 API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "question": prompt,
                }
                if selected_file != "All Documents":
                    payload["filename"] = selected_file

                res = requests.post(
                    f"{API_URL}/ask",
                    json=payload
                )

                if res.status_code != 200:
                    st.error(f"API Error: {res.text}")
                else:
                    data = res.json()
                    answer = data.get("answer", "No answer")
                    sources = data.get("sources", [])
                    # 顯示答案
                    st.write(answer)

                    # 顯示來源（RAG核心）
                    if sources:
                        st.markdown("**📚 Sources:**")
                        for s in sources:
                            st.write(f"- {s}")

                    # 存入歷史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

            except Exception as e:
                st.error(f"Connection error: {str(e)}")
