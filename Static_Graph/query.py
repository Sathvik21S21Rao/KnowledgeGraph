from langchain_core.prompts import PromptTemplate
import langchain_core.messages.ai
class Query:
    def __init__(self,context,query,llm):
        self.context=context
        self.query=query
        self.llm=llm
        
    def format_query(self):
        template="""Based on the following context answer the query\n\nContext:{context}\n\n Query: {query}"""
        return PromptTemplate(template=template,input_variables=["context","query"])
    def execute_query(self):
        chain=self.format_query() | self.llm
        response= chain.invoke({"context":self.context,"query":self.query})
        if isinstance(response,str):
            return response
        
        elif isinstance(response,langchain_core.messages.ai.AIMessage):
            return response.content
