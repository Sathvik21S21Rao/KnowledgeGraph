from duckduckgo_search import DDGS
import requests
from langchain_community import embeddings
from langchain_community.vectorstores import Chroma
from Internet_Retrieval.prompts import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import langchain_core.messages.ai
from googlesearch import search
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



__import__('pysqlite3')
import sys
from urllib.request import urlopen
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
class InternetSearch:
    def __init__(self,headers_to_split_on,embeddings,llm):
        
        self.header_to_split_on=headers_to_split_on
        self.embeddings=embeddings
        self.llm=llm
        self.prompt=ChatPromptTemplate.from_template(template=internet_retrieval_prompt)
        self.splitter=RecursiveCharacterTextSplitter(chunk_size=1000)

    def search(self, query, max_results=5):
        return list(search(query,num_results=max_results))
    
    def load_html(self,urls):
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        return docs_transformed

    def _load_html_to_vectorstore(self,html):
        
        html=self.splitter.split_documents(html)
        self.vectorstore = Chroma.from_documents(
        documents=html,
        collection_name="test",
        embedding=self.embeddings
    )
        self.retriever=self.vectorstore.as_retriever()
        
    
    
    def retrieve_info(self, query, max_results,chat_history):
        results = self.search(query, max_results=max_results)
        print(results)
        html=self.load_html(results)
        
        self._load_html_to_vectorstore(html)
        rag_chain = (
        {"context": self.retriever, "chat_history": RunnablePassthrough(),"query": RunnablePassthrough()}
        | self.prompt
        | self.llm
    )
        response=rag_chain.invoke({"context": self.retriever, "chat_history": chat_history, "query": query})
        
        if  isinstance(response, langchain_core.messages.ai.AIMessage):
            return response.content
        else:
            return response

if __name__=="__main__":
    """Usage 
        em=embeddings.OllamaEmbeddings(model="nomic-embed-text")
        isearch=InternetSearch([("h1", "Header 1"), ("h2", "Header 2")]
    ,em,llm)
        print(isearch.retrieve_info("Latest about Satya nadela",5,""))
    """
    