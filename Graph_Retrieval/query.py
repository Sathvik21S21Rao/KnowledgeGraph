from langchain_core.prompts import PromptTemplate
import langchain_core.messages.ai
from Graph_Retrieval.prompts import query_template
class Query:
    def __init__(self,context,query,llm,chat_history):
        self.context=context
        self.query=query
        self.llm=llm
        self.chat_history=chat_history
    def format_query(self):
        
        return PromptTemplate(template=query_template,input_variables=["context","query","chat_history"])
    def execute_query(self):
        chain=self.format_query() | self.llm
        response= chain.invoke({"context":self.context,"query":self.query,"chat_history":self.chat_history})
        if isinstance(response,str):
            return response
        
        elif isinstance(response,langchain_core.messages.ai.AIMessage):
            return response.content
