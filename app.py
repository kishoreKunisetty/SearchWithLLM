# import langchain
import streamlit as st
import newspaper
import requests
from newspaper import Article
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain import llms
from langchain.llms import HuggingFaceHub
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import  VectorDBQA

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_CSE_ID"] = os.getenv('GOOGLE_CSE_ID')
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# with st.sidebar as sb:
#     os.environ["GOOGLE_CSE_ID"] = st.text_input(type="password", label="CSE_ID")

search = GoogleSearchAPIWrapper()
Documents = []
article_links = []

def top5_results(query):
    return search.results(query, 5)

@st.cache_resource
def load_tool():
    return Tool(
            name="Google Search",
            description="Search Google for recent results.",
            func=top5_results,
        )

@st.cache_resource
def load_llm():
    return HuggingFaceHub(
            repo_id="huggingfaceh4/zephyr-7b-alpha", 
            model_kwargs={"temperature": 0.9, "max_length": 512,"max_new_tokens":512}
        )

@st.cache_resource
def load_text_splitter():
    return CharacterTextSplitter(separator=' ', chunk_size=1000, chunk_overlap=200)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings()

tool = load_tool()
llm = load_llm()

text_splitter = load_text_splitter()
embeddings = load_embeddings()

# Create Streamlit app
st.title('News Search Agent')

search_query = st.text_input('Enter a search query:')

if search_query:
    response = tool.run(search_query)
    for obj in response:
        if "link" in obj.keys() :
            link = obj.get("link",None)
            if link != None:
                print(f"link : {link}")

                article = Article(link, language="en")
                article.download()
                article.parse()
                article.nlp()

                Documents.append(Document(page_content=article.text, metadata={'source': link, 'page_title': article.title}))
    print(f"[INFO] >>> Documents generated \n\nsplitting into texts")
    texts = text_splitter.split_documents(Documents)

    docsearch = Chroma.from_documents(texts, embeddings)
    print(f"[INFO] >>> Documents are stored in ChromaDB")
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
    result = qa({"query": search_query})
    st.write(result['result'])
    
    for itm in result['source_documents']:
        article_links.append(tuple(itm.metadata.values()))
    article_links = set(article_links)
    # st.write(article_links)
    st.write("\nArticles")
    for i,itm in enumerate(article_links):
        st.markdown(f"{i+1} [{itm[0]}]({itm[1]})")