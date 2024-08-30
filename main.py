import yaml
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
import os
import warnings
from checkDataUpdates.checkFileUpdates import SyncData,create_temp_folder
from Graph_Generation.graph_extraction import *
from Community_Generation.communitySummary import UpdateCommunities
from langchain_community.embeddings import OllamaEmbeddings
from Graph_Retrieval.vector_retreival import VectorStore

warnings.filterwarnings("ignore")
console = Console()

def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        return pickle.load(f)

def save_graph(graph, graph_path):
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)

def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_llm(config):
    if config.get("server").lower() == "openai" and config.get("api_key"):
        from langchain_openai.chat_models import ChatOpenAI
        return ChatOpenAI(api_key=config["api_key"], model=config["model"], temperature=config["temperature"])
    elif config.get("server").lower() == "ollama":
        from langchain_experimental.llms.ollama_functions import OllamaFunctions
        return OllamaFunctions(model=config["model"], temperature=config["temperature"])
    elif config.get("server").lower() == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(temperature=config["temperature"], model=config["model"], groq_api_key=config.get("api_key"))
    else:
        raise ValueError("Invalid server configuration")
    
def initialize_embedding_model(config):
    embedding_server = config.get("embedding_server").lower()
    if embedding_server=="ollama":
        from langchain_community import embeddings
        return embeddings.OllamaEmbeddings(model=config.get("embedding_model",""))
    elif embedding_server=="huggingface":
        from langchain_community import embeddings
        return embeddings.HuggingFaceEmbeddings(model=config.get("embedding_model",""))
    elif embedding_server=="openai":
        from langchain_community import embeddings
        return embeddings.OpenAIEmbeddings(model=config.get("embedding_model",""),api_key=config.get("api_key"))
    elif embedding_server=="local":
        return None

# Load configuration
config = read_config('config.yaml')

# Initialize LLM
llm = initialize_llm(config)

# Other variables
graph_file_path = config.get("graph_file_path")
node2vec_model_path = config.get("node2vec_model_path")
node_data_dir = config.get("node_data_dir")
community_data_dir = config.get("community_data_dir")
node_vectorstore_path = config.get("node_vectorstore_path")
collection_name = config.get("collection_name")
chunk_size = config.get("chunk_size")
chunk_overlap = config.get("chunk_overlap")
sentence_model_path = config.get("sentence_model_path")
node2vec_embeddings_path = config.get("node2vec_embeddings_path")
sentence_embeddings_path = config.get("sentence_embeddings_path")
node_names_path = config.get("node_names_path")
faiss_model_path = config.get("faiss_model_path")
sentence_model_name = config.get("sentence_model_name")
use_sentence_embeddings = config.get("use_sentence_embeddings")
chunk_path=config.get("chunk_path")



from Graph_Generation.graph_extraction import *
import networkx as nx
import pickle
from Graph_Retrieval.sentence_graph_retrieval import SentenceGraphRetrieval
from Graph_Retrieval.context_based_node_retrieval import ContextBasedNodeRetrieval
from Graph_Retrieval.query import Query

def main(create_graph):
    console.print(Panel.fit("[bold magenta]GraphRAG![/bold magenta]"))
    
    if create_graph:
        with console.status("[bold green]Creating graph..."):
            chain = GraphExtractionChain(llm=llm)
            data = DataLoader(path=config["data_path"], chunk_size=chunk_size, chunk_overlap=chunk_overlap).load()
            NxData = PrepareDataForNX().execute(data, chain)
            graph = nx.Graph()
            vectorstore=VectorStore(embedding=initialize_embedding_model(config),persist_dir=node_vectorstore_path,collection_name=collection_name,create=True,documents=data,metadata=[{"chunk_id":i} for i in range(len(data))])
            graph.add_nodes_from(NxData[0])
            graph.add_edges_from(NxData[1])
            
            with open(graph_file_path, "wb") as f:
                pickle.dump(graph, f)
    else:
        with console.status("[bold green]Loading graph..."):
            if os.path.exists(graph_file_path):
                with open(graph_file_path, "rb") as f:
                    graph = pickle.load(f)
            else:
                console.print("[bold red]Graph file not found. Please create the graph first.[/bold red]")
                return
    
    with console.status("[bold green]Initializing embeddings and creating communities..."):
        if not use_sentence_embeddings:
            obj = ContextBasedNodeRetrieval(llm, graph, node2vec_model_path, node_data_dir, community_data_dir, create_graph)
            obj.setup()
        else:
            obj = SentenceGraphRetrieval(
                graph, create_graph, node2vec_model_path,
                sentence_model_path, node2vec_embeddings_path,
                sentence_embeddings_path, node_names_path,
                faiss_model_path, sentence_model_name
            )
    
    console.print(Panel.fit("[bold green]Chatbot initialized. Type 'exit' to end the conversation.[/bold green]"))
    
    while True:
        query = Prompt.ask("[bold cyan]You")
        if query.lower() == 'exit':
            console.print(Panel("[bold green]Chatbot: Goodbye![/bold green]", expand=True))
            break
        
        with console.status("[bold green]Thinking..."):
            context = obj.get_context(query=query)
            
            if isinstance(obj, ContextBasedNodeRetrieval):
                response = Query(context, query, llm, chat_history=obj.chat_history).execute_query()
                obj.chat_history.append({"question": query, "response": response})
            else:
                response = Query(context, query, llm, chat_history=[]).execute_query()
                
        console.print(Panel(Markdown(f"**Chatbot:** {response}"), expand=True))

if __name__ == "__main__":
    create_graph=input("Create a new graph? (y/n): ")
    if create_graph.lower() == 'y':
        create_graph = True
        create_temp_folder(config["data_path"],community_data_dir)


    else:
        create_graph = False
        
        with console.status("[bold green]Checking for update..."):
            sync=SyncData(folder=config["data_path"],temp_folder="./.temp")
            updates=sync.compareFolders()
            if updates:
                console.print("[bold red]Data update found.[/bold red]")

                updates="\n".join(updates)
                updates=DataLoader(path=None,chunk_overlap=chunk_overlap,chunk_size=chunk_size).load_text(updates)

                start_chunk=len(VectorStore(embedding=initialize_embedding_model(config),persist_dir=node_vectorstore_path,collection_name=collection_name,create=False,update=False).get_vectorstore().get()["documents"])

        
                vectorstore=VectorStore(embedding=initialize_embedding_model(config),persist_dir=node_vectorstore_path,collection_name=collection_name,create=False,update=True,documents=updates,metadata=[{"chunk_id":i} for i in range(start_chunk,start_chunk+len(updates))])
                
                sync.syncTempFolder()
                chain=GraphExtractionChain(llm=llm)
                graph=load_graph(graph_file_path)
                start_chunk=0
                updated_nodes,added_nodes,added_edges=UpdateGraph(graph=graph,graph_path=graph_file_path,start_chunk=start_chunk).execute(updates,chain)

                
                updated_nodes=[node[0] for node in updated_nodes+added_nodes]
                UpdateCommunities(graph=graph,llm=llm,community_dir=community_data_dir,create=False,updated_nodes=updated_nodes).update_communities()
                

    main(create_graph)
