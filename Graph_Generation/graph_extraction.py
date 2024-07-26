from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
# from prompts import *
import os
import langchain_core.messages.ai
from langchain_core.pydantic_v1 import BaseModel,Field
from typing import List, Tuple
from Graph_Generation.prompts import ENTITY_TYPE_PROMPT,ENTITY_EXTRACTION_PROMPT,RELATION_PROMPT
import tqdm
class Entity_type(BaseModel):
    entity_type :list =Field(description="The entity types found",)

class Entity(BaseModel):
    entity_name:List[str]=Field(description="The entities found")
    entity_type:List[str]=Field(description="The entity types corresponding to the entities")
    entity_description:List[str]=Field(description="The description of the corresponding entities")
class Relation(BaseModel):
    relation:List[Tuple[str,str,str]]=Field(description="The relations found between the entities along with description")
    
class GraphExtractionChain:
    def __init__(self,llm):
        self.llm=[llm.with_structured_output(Entity_type),llm.with_structured_output(Entity),llm.with_structured_output(Relation)]
        self.templates=[PromptTemplate(template=ENTITY_TYPE_PROMPT,input_variables=["task","input_text"]),PromptTemplate(template=ENTITY_EXTRACTION_PROMPT,input_variables=["entity_types","input_text","prev_entities"]),PromptTemplate(template=RELATION_PROMPT,input_variables=["entities","input_text"])]
    
    def _extract_entity_types(self,task,input_text,prev_entities="None")->Entity_type:
        chain=self.templates[0] | self.llm[0]
        response=chain.invoke({"task":task,"input_text":input_text,"prev_entities":prev_entities})
        try:
            return response
        except:
            raise ValueError(f"Error in extracting entity types.The LLM is unable to return the output in the expected format.")
    
    def _extract_entities(self,entity_types,input_text,prev_entities="None")->Entity:
        chain=self.templates[1] | self.llm[1]
        response=chain.invoke({"entity_types":entity_types,"input_text":input_text,"prev_entities":prev_entities})
        try:
            return response
        except:
            raise ValueError(f"Error in extracting entities.The LLM's response is {response}.The LLM is unable to return the output in the expected format.")
    
    def _extract_relations(self,entities,input_text)->Relation:
       
        entities=",".join(entities)
        
        chain=self.templates[2] | self.llm[2]
        response=chain.invoke({"entities":entities,"input_text":input_text})
        try:
            return response
        except:
            raise ValueError(f"Error in extracting relations. The LLM's response is {response}. The LLM is unable to return the output in the expected format.")
    
    
    def execute(self,input_text,prev_entities):
        task="Identify classes for the objects in the given text. Classes could be person, location, organization, concept, etc."
        entity_types=self._extract_entity_types(task,input_text)
        entities=self._extract_entities(entity_types,input_text,prev_entities=prev_entities)
        relations=self._extract_relations(entities.entity_name,input_text)
        return entities,relations
    
    
class DataLoader:
    def __init__(self,path,chunk_size=512):
        self.text_splitter=CharacterTextSplitter(chunk_size=chunk_size)
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

class PrepareDataForNX:
    def __init__(self):
        pass

    def transform_dict_to_nx_format(self,dictionary,relation=False):
        keys=list(dictionary.keys())
        vals=[{"description":dictionary[key]} for key in keys]
        if relation:
            temp=list(zip(*keys))
            return list(zip(temp[0],temp[1],vals))
        return list(zip(keys,vals))
    
    def load_data_from_llm(self,data,chain:GraphExtractionChain):
        final_entities={}
        final_relations={}
        for i in tqdm(range(len(data)),desc="Chunk"):
            text=data[i]
            entities,relations=chain.execute(text,",".join(list(final_entities.keys())))
            for entity,description in zip(entities.entity_name,entities.entity_description):
                
                if final_entities.get(entity):
                    final_entities[entity.lower()]+=description
                else:
                    final_entities[entity.lower()]=description

            for relation in relations.relation:
                if final_relations.get((relation[0].lower(),relation[1].lower())):
                    final_relations[(relation[0].lower(),relation[1].lower())]+="."+relation[2]
                else:
                    final_relations[(relation[0].lower(),relation[1].lower())]=relation[2]
        return final_entities,final_relations
        
    def execute(self,data,chain):
        self.entities,self.relations=self.load_data_from_llm(data,chain)
        entities=self.transform_dict_to_nx_format(self.entities)
        relations=self.transform_dict_to_nx_format(self.relations,relation=True)
        return entities,relations






        
