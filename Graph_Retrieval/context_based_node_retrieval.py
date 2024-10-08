from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os
from pydantic import BaseModel, Field
from typing import List
from gensim.models import Word2Vec
from node2vec import Node2Vec
import langchain_core.messages.ai
from pydantic_core import from_json
from Graph_Retrieval.prompts import *
from Community_Generation.communitySummary import *
from Graph_Retrieval.vector_retreival import VectorStore

class NodeOutput(BaseModel):
    node_names: List[str] = Field(description="The names of the relevant nodes", example=["node1", "node2", "node3"])

class ContextBasedNodeRetrieval:
    def __init__(self, llm, graph,node2vec_model_path,data_dir="node_data",community_data="community_data",create=False,embeddings=None,vectorstore=None):
        self.llm = llm
        self.data_dir = data_dir
        self.community_data=community_data
        self.chat_history = []
        self.graph=graph
        self.node2vec_model_path=node2vec_model_path
        self.create=create
        self.vectorstore=vectorstore

        self.community=CommunitySummary(self.graph,self.llm,self.community_data,self.create)
        
        
    def setup(self):
        if self.create:
            self._create_models()
        else:
            self._load_models()

    def _save_and_index_documents(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        with open(f"{self.data_dir}/node.txt", "w") as f:
            for node in self.graph.nodes():
                f.write(f"{node}\n")
        
    def _create_node2vec_model(self):
        
        node2vec = Node2Vec(self.graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        self.node2vec_model = model
        self.node2vec_model.save(self.node2vec_model_path)

    def _create_models(self):
        self._save_and_index_documents()
        self._create_node2vec_model()
        

    def _load_models(self):
        
        if self.node2vec_model_path:
            self.node2vec_model = Word2Vec.load(self.node2vec_model_path)
        else:
            raise FileNotFoundError("Node2Vec model not found. Either create the model or provide the correct path in the config file.")

    def _retrieve_nodes(self, query):
        
        
        template = retrieve_nodes_prompt
        
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {"context": RunnablePassthrough(), "query": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
            | prompt
            | self.llm
        )
        with open(f"{self.data_dir}/node.txt", "r") as f:
            node_names = f.read().split("\n")
        response = rag_chain.invoke({"context": node_names, "query": query, "chat_history": self.chat_history[-5:]})
        
        if isinstance(response, langchain_core.messages.ai.AIMessage):
            response = response.content
        
        response = response[response.find("{"):response.rfind("}") + 1]
        
        response = NodeOutput.model_validate(from_json(response, allow_partial=True))
        return response
                            

    def _get_node_descriptions(self, nodes)->dict:
        return {node:self.graph.nodes[node].get("description") for node in nodes}
    
    def _get_edge_descriptions(self,edges)->dict:
        return {edge:self.graph.to_undirected()[edge[0]][edge[1]].get("description") for edge in edges}
    
    def _enrich_nodes(self,nodes,score_thresh):
        node_names=[node_name.lower() for node_name in nodes.node_names if node_name.lower() in self.node2vec_model.wv.key_to_index]
        similar_nodes_by_node2vec = [self.node2vec_model.wv.most_similar(node_name) for node_name in node_names]
        
        similar_nodes=set(node_names)
        for i, nodes in enumerate(similar_nodes_by_node2vec):
            similar_nodes.update([node[0] for node in nodes if node[1]>score_thresh])
        return similar_nodes
    
    def _retrieve_edges(self,nodes):
        similar_edges=[]
        for node1 in nodes:
            for node2 in nodes:
                if self.graph.has_edge(node1,node2) and (node2,node1) not in similar_edges:
                    similar_edges.append((node1,node2))
        return similar_edges


    def get_context(self, query, score_thresh=0.8)->str:
        
        similar_nodes=self._retrieve_nodes(query)
        context=""
        
        if len(similar_nodes.node_names)/len(self.graph.nodes)>0.4:
            communities=set()
            for node in similar_nodes.node_names:
                if self.community.node_to_community_mapping.get(node):
                    communities.add(self.community.node_to_community_mapping[node][1])
            for community in communities:
                community=self.community.load_community(community)
                context+=f"Community: {community.community_name}\n Description: {community.community_description}\n"
            return context
            
        similar_nodes=self._enrich_nodes(similar_nodes,score_thresh)
        similar_edges=self._retrieve_edges(similar_nodes)
        
        node_descriptions = self._get_node_descriptions(similar_nodes)
        edge_descriptions = self._get_edge_descriptions(similar_edges)
        context = "Nodes and their descriptions:\n"
        for node, description in node_descriptions.items():
            context += f"{node}: {description}\n"

        context += "\nEdges and their descriptions:\n"
        for edge,description in edge_descriptions.items():
            context += f"{edge[0]} <-> {edge[1]}: {description}\n"
        return context
