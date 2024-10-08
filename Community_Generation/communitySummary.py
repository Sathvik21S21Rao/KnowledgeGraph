
from networkx.algorithms.community import girvan_newman
import os
import pickle
from langchain_core.prompts import PromptTemplate
from pydantic_core import from_json
from pydantic import BaseModel,Field
from pydantic.types import List
from Community_Generation.prompts import *
from langchain_core.messages.ai import AIMessage

class Community(BaseModel):
    community_name:str=Field(description="The name of the community",example="Community1")
    community_description:str=Field(description="The description of the community",example="Description of the community")

class Update_Community(BaseModel):
    new_sentences:List[str]=Field(description="The new sentences to be added to the existing summary",example=["sentence1","sentence2"])
    indices:List[int]=Field(description="The indices where the new sentences should be inserted",example=[1,2])
    
class CommunitySummary:
    def __init__(self, graph,llm,community_dir,create):
        self.graph = graph
        self.llm=llm
        self.prompt=PromptTemplate(template=community_prompt,input_variables=["input_text"])
        self.chain=self.prompt | self.llm
        self.community_dir=community_dir
        self.node_to_community_mapping={}
        self.index_to_community_mapping={}
        if not os.path.exists(self.community_dir):
            os.makedirs(self.community_dir)
        
        if create:
            self.generate_summary()
        else:
            self._load_node_to_community_mapping()
            self._load_index_to_community_mapping()

    def _girvan_newman_communities(self):
        communities_generator = girvan_newman(self.graph)
        first_level_communities = next(communities_generator)
        return sorted(map(sorted, first_level_communities))
    
    def _save_community(self,community):
        community.community_name=community.community_name.replace(" ","_")
        
        with open(f"{self.community_dir}/{community.community_name}.txt", "w") as f:
            f.write(community.community_description)

    def save_index_to_community_mapping(self,indices,community_names):
        for i,community_name in zip(indices,community_names):
            self.index_to_community_mapping[i]=community_name
        with open(f"{self.community_dir}/index_to_community_mapping.pkl", "wb") as f:
            pickle.dump(self.index_to_community_mapping, f)

    def _load_index_to_community_mapping(self):
        with open(f"{self.community_dir}/index_to_community_mapping.pkl", "rb") as f:
            index_to_community_mapping = pickle.load(f)
        self.index_to_community_mapping=index_to_community_mapping
        


    def load_community(self,community_name):
        if(isinstance(community_name,int)):
            community_name=self.index_to_community_mapping[community_name]
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
        community_names=[]
        for i,community in enumerate(first_level_communities):
            community_description = ""

            for node in community:
                community_description+= self.graph.nodes[node].get("description","")

            response=self.chain.invoke({"input_text":community_description})
            if isinstance(response,AIMessage):
                response=response.content
            response=response[response.find("{"):response.rfind("}")]+"}"
            community_class=Community.model_validate(from_json(response,allow_partial=True))

            for node in community:
                self.node_to_community_mapping[node]=(i,community_class.community_name)
            community_names.append(community_class.community_name)
            self._save_community(community_class)
        self.save_index_to_community_mapping(list(range(len(first_level_communities))),community_names)
        self._save_community_mapping()
    

class UpdateCommunities(CommunitySummary):

    def __init__(self,updated_nodes,graph,llm,community_dir,create):
        super().__init__(llm=llm,graph=graph,community_dir=community_dir,create=create)
        self.updated_nodes=set(updated_nodes)
        self.update_community_template=PromptTemplate(template=update_community_prompt,input_variables=["existing_summary","updated_sentences"])
        self.update_chain=self.update_community_template | self.llm

    def _identify_new_communities(self):
        communities_generator = girvan_newman(self.graph)
        first_level_communities = next(communities_generator)
        return sorted(map(sorted, first_level_communities))

    
    def compare_communities(self):

        self.new_communities=self._identify_new_communities()
        old_node_to_community_mapping=self.node_to_community_mapping
        self.max_indexed_community=max([val[0] for val in old_node_to_community_mapping.values()])
    
        for i,community in enumerate(self.new_communities):
            for node in community:
                if node not in old_node_to_community_mapping or old_node_to_community_mapping[node][0]!=i:
                    self.updated_nodes.add(node)
                    
  
            
    def update_communities(self):
        self.compare_communities()
        for i,community in enumerate(self.new_communities):
            if i>self.max_indexed_community: # new community
                community_description = ""

                for node in community:
                    community_description+= self.graph.nodes[node].get("description","")

                response=self.chain.invoke({"input_text":community_description})
                if isinstance(response,AIMessage):
                    response=response.content
               
                response=response[response.find("{"):response.rfind("}")]+"}"
                
                community_class=Community.model_validate(from_json(response,allow_partial=True))
                

                for node in community:
                    self.node_to_community_mapping[node]=(i,community_class.community_name)
                self.index_to_community_mapping[i]=community_class.community_name

            
            else: #modify existing community
                community_class=self.load_community(i)
                community_description = community_class.community_description.split(".")
                updated_nodes=[node for node in community if node in self.updated_nodes or self.node_to_community_mapping[node][0]!=i  ]
                updated_nodes_descriptions=""
               
                if(len(updated_nodes)==0):
                    continue

                for node in updated_nodes:
                    updated_nodes_descriptions+=self.graph.nodes[node].get("description","")

                if updated_nodes_descriptions=="": #no change
                    continue
                response=self.update_chain.invoke({"existing_summary":community_description,"updated_sentences":updated_nodes_descriptions})
                if isinstance(response,AIMessage):
                    response=response.content
          
                response=response[response.find("{"):response.rfind("}")]+"}"
                try:
                    update_class=Update_Community.model_validate(from_json(response,allow_partial=True))
                except:
                    raise Exception(f"Error in updating community {community_class.community_name} with response {response}")
                for i,sentence in enumerate(update_class.new_sentences):
                    try:
                        community_description=community_description[:i+update_class.indices[i]]+[sentence]+community_description[i+update_class.indices[i]:]
                    except:
                        break
                community_class.community_description=".".join(community_description)
            
                for node in updated_nodes:
                    self.node_to_community_mapping[node]=(i,community_class.community_name)
            self._save_community(community_class)

        self._save_community_mapping()

        

    
                    
                    
                    

                    
        
        

