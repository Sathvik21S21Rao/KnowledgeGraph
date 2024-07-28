import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
import os

console = Console()

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
        return ChatGroq(temperature=config["temperature"], model=config["model"], api_key=config.get("api_key"))
    else:
        raise ValueError("Invalid server configuration")

# Load configuration
config = read_config('config.yaml')

# Initialize LLM
llm = initialize_llm(config)

# Other variables
create_graph = config.get("create_graph", False)
graph_file_path = config.get("graph_file_path")
use_node_vectorstore = config.get("use_node_vectorstore", False)
node2vec_model_path = config.get("node2vec_model_path")
node_data_dir = config.get("node_data_dir")
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

from Graph_Generation.graph_extraction import *
import networkx as nx
import pickle
from Graph_Retrieval.sentence_graph_retrieval import SentenceGraphRetrieval
from Graph_Retrieval.vector_based_node_retrieval import VectorBasedNodeRetrieval
from Graph_Retrieval.query import Query

def main():
    console.print(Panel.fit("[bold magenta]GraphRAG![/bold magenta]"))
    
    if create_graph:
        with console.status("[bold green]Creating graph..."):
            chain = GraphExtractionChain(llm=llm)
            data = DataLoader(path=config["data_path"], chunk_size=chunk_size,chunk_overlap=chunk_overlap).load()
            NxData = PrepareDataForNX().execute(data, chain)
            graph = nx.Graph()
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
    
    with console.status("[bold green]Initializing embeddings..."):
        if use_node_vectorstore:
            obj = VectorBasedNodeRetrieval(llm, graph, node2vec_model_path, node_data_dir, node_vectorstore_path, collection_name, create_graph)
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
            console.print(Panel("[bold green]Chatbot: Goodbye![/bold green]", expand=False))
            break
        
        with console.status("[bold green]Thinking..."):
            context = obj.get_context(query=query)
            
            if isinstance(obj, VectorBasedNodeRetrieval):
                console.print(Panel(Markdown(f"**Context:** {context}"), expand=False))
                response = Query(context, query, llm,chat_history=obj.chat_history).execute_query()
                obj.chat_history.append({"question": query, "response": response})
            else:
                
                response = Query(context, query, llm, chat_history=[]).execute_query()
                
        console.print(Panel(Markdown(f"**Chatbot:** {response}"), expand=False))

if __name__ == "__main__":
    main()