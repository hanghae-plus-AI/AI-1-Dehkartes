import os
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.environ["OPENAI_API_KEY"])

# 크롤링


loader = WebBaseLoader(
	web_paths=("https://spartacodingclub.kr/blog/all-in-challenge_winner",),
	bs_kwargs=dict(
		parse_only=bs4.SoupStrainer(
			class_=("editedContent", "css-j3idia")
		)
	),
	encoding="utf-8"
)
docs = loader.load()

# Knowledge source에서 텍스트 추출
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=1000, 
	chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
	documents=splits, 
	embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)


st.title("Bot")

if "messages" not in st.session_state:
	st.session_state.messages = []

for m in st.session_state["messages"]:
	with st.chat_message(m["role"]):
		st.markdown(m["content"])
		
if user_msg := st.chat_input("What is up?"):
	with st.chat_message("user"):
		st.markdown(user_msg)
		retrieved_docs = retriever.invoke(user_msg)
		prompt = hub.pull("rlm/rag-prompt")
		user_prompt = prompt.invoke({"context": format_docs(retrieved_docs), "question": user_msg})

	st.session_state.messages.append({"role": "user", "content": user_msg})

	with st.chat_message("assistant"):
		result = llm.invoke(user_prompt).content
		st.markdown(result)
		
	st.session_state.messages.append({
		"role": "assistant", 
		"content": result
	})