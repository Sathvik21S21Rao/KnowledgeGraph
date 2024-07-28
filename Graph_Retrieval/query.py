from langchain_core.prompts import PromptTemplate
import langchain_core.messages.ai
class Query:
    def __init__(self,context,query,llm,chat_history):
        self.context=context
        self.query=query
        self.llm=llm
        self.chat_history=chat_history
    def format_query(self):
        template="""Based on the following context answer the query and the following chat history\n\nContext:{context}\n\nChat history:{chat_history}\n\n Query: {query}"""
        return PromptTemplate(template=template,input_variables=["context","query","chat_history"])
    def execute_query(self):
        chain=self.format_query() | self.llm
        response= chain.invoke({"context":self.context,"query":self.query,"chat_history":self.chat_history})
        if isinstance(response,str):
            return response
        
        elif isinstance(response,langchain_core.messages.ai.AIMessage):
            return response.content
