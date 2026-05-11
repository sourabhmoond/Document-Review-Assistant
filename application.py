from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
## from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import os

st.set_page_config(page_title="Legal Document Review App",layout="wide")

st.title("-- Legal Document Review Assistant")
st.write("upload a legal document and ask questions, analyze risks, and summarize clauses")

uploaded_file = st.file_uploader("Upload Legal Document",type=["pdf","txt"])

def load_document(file):
    if file.name.endswith(".pdf"):
        with open("temp.pdf","wb") as f:
            f.write(file.read())
        loader = PyPDFLoader("temp.pdf")
    else:
        with open("temp.txt","wb") as f:
            f.write(file.read())
        loader = TextLoader("temp.txt")
    
    documents = loader.load()
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


os.environ["Gemini_apikey26"] = "Use Your API Key"

print()

def create_vectorstore(docs):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
    #                                           api_key= os.environ['Gemini_apikey26'])
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )                                        
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

LEGAL_PROMPT = PromptTemplate(input_variables = ['context',"question"],
                              template = """
You are a legal expert AI.
Analyze the following legal document context and answer the question.

Provide:
1. Clear answer
2. Legal interpretation
3. Risks (if any)
4. Suggestions

Context:
{context}

Question:
{question}

Answer:
""")


def create_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0.37,
                                 api_key =os.environ["Gemini_apikey26"])
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | LEGAL_PROMPT
        | llm
    )

    return chain

if uploaded_file:
    st.info("Processing document...")

    documents = load_document(uploaded_file)
    split_docs = split_documents(documents)
    vectorstore = create_vectorstore(split_docs)
    qa_chain = create_qa_chain(vectorstore)

    st.success("Document processed successfully!")


    query = st.text_input("Ask a legal document about the document:")

    if query:
        with st.spinner("Analyzing..."):
            response = qa_chain.invoke(query)
            st.subheader("Answer")
            st.write(response.content)

    
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Summarize"):
            response = qa_chain.invoke("Summarize this document in simple terms in under 300 words.")
            st.write(response.content)
 
    with col2:
        if st.button("Identify Risks"):
            response = qa_chain.invoke("Identify risks, liabilities, and red flags in under 300 words.")
            st.write(response.content)
        
    with col3:
        if st.button("Key Clauses"):
            response = qa_chain.invoke(
                "Extract key clauses like termination, liability, indemnity in under 300 words")
            st.write(response.content)

else:
    st.warning("Please upload a legal document to begin.")


