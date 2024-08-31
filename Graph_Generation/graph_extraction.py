from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import langchain_core.messages.ai
from pydantic_core import from_json

from pydantic import BaseModel,Field
from typing import List, Tuple
from Graph_Generation.prompts import ENTITY_TYPE_PROMPT,ENTITY_EXTRACTION_PROMPT,RELATION_PROMPT
from tqdm import tqdm
import pickle
from langchain_groq import ChatGroq
from Community_Generation.communitySummary import UpdateCommunities
class Entity_type(BaseModel):
    entity_type :list =Field(description="The entity types found",example=["type1","type2"])

class Entity(BaseModel):
    entity_name:List[str]=Field(description="The entities found",example=["entity1","entity2"])
    entity_type:List[str]=Field(description="The entity types corresponding to the entities",example=["type1","type2"])
    entity_description:List[str]=Field(description="The description of the corresponding entities",example=["description1","description2"])

class Relation(BaseModel):
    relation:List[Tuple[str,str,str]]=Field(description="The relations found between the entities along with description",example=[("entity1","entity2","relation description"),("entity2","entity3","relation description")])
    
class GraphExtractionChain:
    def __init__(self,llm):
        self.llm=llm
        self.templates=[PromptTemplate(template=ENTITY_TYPE_PROMPT,input_variables=["task","input_text"]),PromptTemplate(template=ENTITY_EXTRACTION_PROMPT,input_variables=["entity_types","input_text","prev_entities"]),PromptTemplate(template=RELATION_PROMPT,input_variables=["entities","input_text"])]
    
    def _extract_entity_types(self,task,input_text,prev_entities="None",retries=3)->Entity_type:
        chain=self.templates[0] | self.llm
        response=chain.invoke({"task":task,"input_text":input_text,"prev_entities":prev_entities})
        if isinstance(response,langchain_core.messages.ai.AIMessage):
            response=response.content
        response=response[response.find("{"):response.rfind("}")]+"}"
       
        try:
            
            return Entity_type.model_validate(from_json(response,allow_partial=True))
        except Exception as e:
            if(retries>0):
                return self._extract_entity_types(task,input_text,prev_entities,retries-1)
            return Entity_type(entity_type=[])
    
    def _extract_entities(self,entity_types,input_text,prev_entities="None",retries=3)->Entity:
        chain=self.templates[1] | self.llm
        response=chain.invoke({"entity_types":entity_types,"input_text":input_text,"prev_entities":prev_entities})
        if isinstance(response,langchain_core.messages.ai.AIMessage):
            response=response.content
        response=response[response.find("{"):response.rfind("}")+1]

        try:
            return Entity.model_validate(from_json(response,allow_partial=True))
        except:
            if(retries>0):
                return self._extract_entities(entity_types,input_text,prev_entities,retries-1)
            return Entity(entity_name=[],entity_type=[],entity_description=[])
    
    def _extract_relations(self,entities,input_text,retries=3)->Relation:
       
        entities=",".join(entities)
        
        chain=self.templates[2] | self.llm
        response=chain.invoke({"entities":entities,"input_text":input_text})
        if isinstance(response,langchain_core.messages.ai.AIMessage) or not isinstance(response,str):
            response=response.content
        response=response[response.find("{"):response.rfind("}")+1]
        
        try:
            return Relation.model_validate(eval(response))
        except Exception as e:
            if(retries>0):
                return self._extract_relations(entities,input_text,retries-1)
            return Relation(relation=[])
    
    
    def execute(self,input_text,prev_entities):
        task="Identify classes for the objects in the given text. Classes could be person, location, organization, concept,item,living thing,technology,thought,time,activity etc.Feel free to generate new classes which are relavant to the text."
        entity_types=self._extract_entity_types(task,input_text)
        entities=self._extract_entities(entity_types,input_text,prev_entities=prev_entities)
        if(entities.entity_name==[]):
            return entities,Relation(relation=[])
        relations=self._extract_relations(entities.entity_name,input_text)
        return entities,relations
    
    
class DataLoader:
    def __init__(self,path,chunk_size,chunk_overlap):
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        self.path=path
    def load(self):
        if os.path.isdir(self.path):
            text=""
            for file in os.listdir(self.path):
                with open(os.path.join(self.path,file)) as fh:
                    text+=fh.read()+"\n\n"
            return self.text_splitter.split_text(text)
        
        with open(self.path) as fh:
            text=fh.read()
        
        return self.text_splitter.split_text(text)
    def load_text(self,text):
        return self.text_splitter.split_text(text)
    
class DataSaver:
    def __init__(self,path,chunks,update=False):
        self.path=path
        self.chunks=chunks
        self.update=update
    def save(self):
        if self.update:
            with open(self.path,"wb") as fh:
                id_to_chunk={i:chunk for i,chunk in enumerate(self.chunks)}
                pickle.dump(id_to_chunk,fh)
            start_chunk=0
        else:
            with open(self.path,"rb") as fh:
                id_to_chunk=pickle.load(fh)
            start_chunk=len(id_to_chunk)
            with open(self.path,"wb") as fh:
                for i,chunk in enumerate(self.chunks):
                    id_to_chunk[len(id_to_chunk)+i]=chunk
                pickle.dump(id_to_chunk,fh)
        return start_chunk
        
    def load_id_to_chunk(self):
        with open(self.path,"rb") as fh:
            return pickle.load(fh)

class PrepareDataForNX:
    def __init__(self):
        pass

    @staticmethod
    def transform_dict_to_nx_format(dictionary,relation=False):
        keys=list(dictionary.keys())
        vals=[{"description":dictionary[key][0],"chunk_ids":dictionary[key][1]} for key in keys]
        if relation:
            temp=list(zip(*keys))
            return list(zip(temp[0],temp[1],vals))
        return list(zip(keys,vals))
    
    @staticmethod
    def load_data_from_llm(data,chain:GraphExtractionChain,final_entities={},final_relations={},start_chunk=0):
        
        initial_entities,initial_relations=list(final_entities.keys()),list(final_relations.keys())
        with tqdm(total=len(data), desc="Processing Chunks", unit="chunk",position=0, leave=True) as pbar:
            for i in range(len(data)):
                text=data[i]
                entities,relations=chain.execute(text,",".join(list(final_entities.keys())))
                for entity,description in zip(entities.entity_name,entities.entity_description):
                    
                    if final_entities.get(entity.lower()):
                        if final_entities[entity.lower()][0] is None:
                            final_entities[entity.lower()][0]=""
                        if final_entities[entity.lower()][1] is None:
                            final_entities[entity.lower()][1]=[]
                        final_entities[entity.lower()][0]+=f"{description}"
                        final_entities[entity.lower()][1].append(i+start_chunk)
                    else:

                        final_entities[entity.lower()]=[description,[i+start_chunk]]

                for relation in relations.relation:
                    if final_relations.get((relation[0].lower(),relation[1].lower())):
                        if final_relations[(relation[0].lower(),relation[1].lower())][0] is None:
                            final_relations[(relation[0].lower(),relation[1].lower())][0]=""
                        if final_relations[(relation[0].lower(),relation[1].lower())][1] is None:
                            final_relations[(relation[0].lower(),relation[1].lower())][1]=[]
                        final_relations[(relation[0].lower(),relation[1].lower())][0]+="."+relation[2]
                        final_relations[(relation[0].lower(),relation[1].lower())][1].append(i+start_chunk)

                    else:
                        final_relations[(relation[0].lower(),relation[1].lower())]=[relation[2],[i+start_chunk]]
                pbar.update(1)
                pbar.set_postfix({"Current Chunk": i+1+start_chunk, "Remaining": len(data) - (i+1+start_chunk)})
        return {key:final_entities[key] for key in final_entities if key not in initial_entities},{key:final_relations[key] for key in final_relations if key not in initial_relations}
        
    def execute(self,data,chain):
        self.entities,self.relations=PrepareDataForNX.load_data_from_llm(data,chain)
        entities=PrepareDataForNX.transform_dict_to_nx_format(self.entities)
        relations=PrepareDataForNX.transform_dict_to_nx_format(self.relations,relation=True)
        return entities,relations
    
class UpdateGraph(PrepareDataForNX):
    def __init__(self,graph,graph_path,start_chunk):
        super().__init__()
        self.graph=graph
        self.graph_path=graph_path
        self.start_chunk=start_chunk
       
    def update_graph(self,entities,relations):
        updated_nodes=[]
        added_nodes=[]
        for entity in entities:
            if self.graph.has_node(entity[0]):
                self.graph.nodes[entity[0]]["description"]+=entity[1]["description"]
                self.graph.nodes[entity[0]]["chunk_ids"]+=entity[1]["chunk_ids"]
                updated_nodes.append(entity[0])
            else:
                self.graph.add_node(entity[0])
                added_nodes.append(entity)

        self.graph.add_edges_from(relations)
        added_edges=[(relation[0],relation[1]) for relation in relations ]
        return updated_nodes,added_nodes,added_edges
    
    def save_graph(self):
        with open(self.graph_path,"wb") as fh:
            pickle.dump(self.graph,fh)
    
    def execute(self,data,chain):
        nodes={node[0]:[node[1].get("description"),node[1].get("chunk_ids")] for node in self.graph.nodes(data=True)}
        edges={(edge[0],edge[1]):[edge[2].get("description"),edge[2].get("chunk_ids")] for edge in self.graph.edges(data=True)}

        self.entities,self.relations=PrepareDataForNX.load_data_from_llm(data,chain,nodes,edges,start_chunk=self.start_chunk)
        entities=PrepareDataForNX.transform_dict_to_nx_format(self.entities)
        relations=PrepareDataForNX.transform_dict_to_nx_format(self.relations,relation=True)
        updated_nodes,added_nodes,added_edges=self.update_graph(entities=entities,relations=relations)
        
        self.save_graph()

        return updated_nodes,added_nodes,added_edges


    
    




        
