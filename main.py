import yaml
from Graph_Generation.graph_extraction import *
import networkx as nx
import pickle
from Static_Graph.netx import GraphNodestoEmbeddings
from Static_Graph.query import Query
import warnings
warnings.filterwarnings("ignore")

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__=="__main__":

    config = read_config('config.yaml')

    if config.get("server").lower()=="openai" and config.get("api_key"):
        from langchain_openai.chat_models import ChatOpenAI
        llm = ChatOpenAI(api_key=config["api_key"], model=config["model"], temperature=config["temperature"])

    elif config.get("server").lower()=="ollama" :
        from langchain_experimental.llms.ollama_functions import OllamaFunctions
        llm = OllamaFunctions(model=config["model"], temperature=config["temperature"])
        

    elif config.get("server").lower()=="groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
    temperature=config.get("temperature"),
    model=config.get("model"),
    api_key=config.get("api_key") # Optional if not set as an environment variable
)
    if config.get("create_graph"):

        chain=GraphExtractionChain(llm=llm)
        data=DataLoader(path=config["data_path"],chunk_size=config["chunk_size"]).load()
        NxData=PrepareDataForNX().execute(data,chain)
        graph=nx.Graph()
        graph.add_nodes_from(NxData[0])
        graph.add_edges_from(NxData[1])
       
        with open(config.get("graph_file_path"), "wb") as f:
            pickle.dump(graph, f)

    
    if os.path.exists(config.get("graph_file_path")):
        with open(config.get("graph_file_path"), "rb") as f:
            graph=pickle.load(f)

        from Static_Graph.netx import GraphNodestoEmbeddings
        obj=GraphNodestoEmbeddings(graph,config["create_graph"],config["node2vec_model_path"],config["sentence_model_path"],config["node2vec_embeddings_path"],config["sentence_embeddings_path"],config["node_names_path"],config["faiss_model_path"],config["sentence_model_name"])
        
        query=input("Enter the query:")
        context=obj.get_context(query=query)

        print(Query(context, query, llm).execute_query())
    else:
        raise FileNotFoundError("Graph file not found. Either create the graph or provide the correct path in the config file.")

        
            
            

