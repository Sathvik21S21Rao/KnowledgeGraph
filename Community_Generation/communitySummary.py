
from networkx.algorithms.community import girvan_newman
import os
import pickle
from langchain_core.prompts import PromptTemplate
from pydantic_core import from_json
from pydantic import BaseModel,Field
from Community_Generation.prompts import *
from langchain_core.messages.ai import AIMessage

class Community(BaseModel):
    community_name:str=Field(description="The name of the community",example="Community1")
    community_description:str=Field(description="The description of the community",example="Description of the community")
class CommunitySummary:
    def __init__(self, graph,llm,community_dir,create):
        self.graph = graph
        self.llm=llm
        self.prompt=PromptTemplate(template=community_prompt,input_variables=["input_text"])
        self.chain=self.prompt | self.llm
        self.community_dir=community_dir
        self.node_to_community_mapping={}
        if not os.path.exists(self.community_dir):
            os.makedirs(self.community_dir)
        if create:
            self.generate_summary()
        else:
            self._load_node_to_community_mapping()

    def _girvan_newman_communities(self):
        communities_generator = girvan_newman(self.graph)
        first_level_communities = next(communities_generator)
        return sorted(map(sorted, first_level_communities))
    
    def _save_community(self,community):
        community.community_name=community.community_name.replace(" ","_")
        
        with open(f"{self.community_dir}/{community.community_name}.txt", "w") as f:
            f.write(community.community_description)
    
    def load_community(self,community_name):
        community_name=community_name.replace(" ","_")
        with open(f"{self.community_dir}/{community_name}.txt", "r") as f:
            community_description = f.read()
        return Community(community_name=community_name,community_description=community_description)

    def _load_node_to_community_mapping(self):
        with open(f"{self.community_dir}/node_to_community_mapping.pkl", "rb") as f:
            node_to_community_mapping = pickle.load(f)
        self.node_to_community_mapping=node_to_community_mapping
    
    def _save_community_mapping(self):
        with open(f"{self.community_dir}/node_to_community_mapping.pkl", "wb") as f:
            pickle.dump(self.node_to_community_mapping, f)
    

    def generate_summary(self):
        first_level_communities = self._girvan_newman_communities()

        for community in first_level_communities:
            community_description = ""

            for node in community:
                community_description+= self.graph.nodes[node].get("description","")

            response=self.chain.invoke({"input_text":community_description})
            if isinstance(response,AIMessage):
                response=response.content
            response=response[response.find("{"):response.rfind("}")]+"}"
            community_class=Community.model_validate(from_json(response,allow_partial=True))

            for node in community:
                self.node_to_community_mapping[node]=community_class.community_name

            self._save_community(community_class)
        self._save_community_mapping()
            
