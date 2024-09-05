from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
import streamlit as st

st.set_page_config(page_title="Assignment 15", page_icon="📃")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    openai_api_key = st.text_input("Input OpenAI API Key:")

if openai_api_key != "":
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=openai_api_key,
    )

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("Assignment 15")
st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!

Input your OpenAI API key and upload your files on the sidebar.
    """
)

with st.sidebar:
    if openai_api_key!="":
        file = st.file_uploader(
            "Upload a .txt, .pdf or .docx file",
            type=["txt", "pdf", "docx"],
        )

if openai_api_key!="" and file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file....")
    if message:
        send_message(message, "human")
        chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")

else:
    st.session_state["messages"] = []
