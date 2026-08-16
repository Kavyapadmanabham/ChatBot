import os
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

load_dotenv()


# ============================================================
# STREAMLIT APP
# ============================================================

st.title("JavaScript RAG Chatbot")


# ============================================================
# PINECONE
# ============================================================

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index_name = os.environ.get("PINECONE_INDEX_NAME")

index = pc.Index(index_name)


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=os.environ.get("GEMINI_API_KEY")
)


# ============================================================
# PINECONE VECTOR STORE
# ============================================================

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


# ============================================================
# GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY")
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

    st.session_state.messages.append(
        SystemMessage(
            content="You are an assistant for question-answering tasks."
        )
    )


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.markdown(message.content)


# ============================================================
# USER INPUT
# ============================================================

prompt = st.chat_input(
    "Ask a question about JavaScript..."
)


# ============================================================
# WHEN USER ASKS A QUESTION
# ============================================================

if prompt:

    # --------------------------------------------------------
    # Display user's question
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )


    # --------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.5
        }
    )


    # --------------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------------

    docs = retriever.invoke(prompt)


    # --------------------------------------------------------
    # COMBINE RETRIEVED DOCUMENTS
    # --------------------------------------------------------

    if docs:

        docs_text = "\n\n".join(
            document.page_content
            for document in docs
        )

    else:

        docs_text = "No relevant information was found in the document."


    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = """
You are a question-answering assistant.

Answer the user's question using ONLY the information
provided in the retrieved context below.

If the answer cannot be found in the context,
say that you don't know.

Keep the answer concise and easy to understand.

Retrieved context:
{context}
"""


    system_prompt_fmt = system_prompt.format(
        context=docs_text
    )


    # --------------------------------------------------------
    # SEND QUESTION + CONTEXT TO GEMINI
    # --------------------------------------------------------

    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt_fmt
            ),

            HumanMessage(
                content=prompt
            )
        ]
    )


    # --------------------------------------------------------
    # EXTRACT GEMINI RESPONSE
    # --------------------------------------------------------

    if isinstance(response.content, list):

        result = "".join(
            item["text"]
            for item in response.content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    else:

        result = response.content


    # --------------------------------------------------------
    # DISPLAY GEMINI ANSWER
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(result)


    # --------------------------------------------------------
    # SAVE ANSWER TO CHAT HISTORY
    # --------------------------------------------------------

    st.session_state.messages.append(
        AIMessage(content=result)
    )