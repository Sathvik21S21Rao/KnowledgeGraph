import yaml
from Graph_Generation.graph_extraction import *
import networkx as nx
import pickle
from Graph_Retrieval.sentence_graph_retrieval import GraphNodestoEmbeddings
from Graph_Retrieval.query import Query
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.progress import Progress
from langchain_core.messages import HumanMessage

warnings.filterwarnings("ignore")

console = Console()

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

def create_graph(config, llm):
    
    chain = GraphExtractionChain(llm=llm)
    data = DataLoader(path=config["data_path"], chunk_size=config["chunk_size"]).load()
    NxData = PrepareDataForNX().execute(data, chain)
    graph = nx.Graph()
    graph.add_nodes_from(NxData[0])
    graph.add_edges_from(NxData[1])
        
    with open(config.get("graph_file_path"), "wb") as f:
        pickle.dump(graph, f)
        
    return graph

def load_graph(config):
    
    if os.path.exists(config.get("graph_file_path")):
        with open(config.get("graph_file_path"), "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError("Graph file not found. Either create the graph or provide the correct path in the config file.")

def chatbot(config, llm):
    if config.get("create_graph"):
        with console.status("[bold green]Creating graph..."):
            graph=create_graph(config, llm)
    else:
        with console.status("[bold green]Loading graph..."):
            graph=load_graph(config)
    
    
    with console.status("[bold green]Initializing embeddings..."):
        obj = GraphNodestoEmbeddings(
            graph, config["create_graph"], config["node2vec_model_path"],
            config["sentence_model_path"], config["node2vec_embeddings_path"],
            config["sentence_embeddings_path"], config["node_names_path"],
            config["faiss_model_path"], config["sentence_model_name"]
        )
    
    console.print(Panel.fit("[bold green]Chatbot initialized. Type 'exit' to end the conversation.[/bold green]"))
    
    while True:
        query = Prompt.ask("[bold cyan]You")
        if query.lower() == 'exit':
            console.print(Panel("[bold green]Chatbot: Goodbye![/bold green]", expand=False))
            break
        
        with console.status("[bold green]Thinking..."):
            context = obj.get_context(query=query)
            response = Query(context, query, llm).execute_query()
        
        console.print(Panel(Markdown(f"**Chatbot:** {response}"), expand=False))

if __name__ == "__main__":
    console.print(Panel.fit("[bold magenta]GraphRAG![/bold magenta]"))
    
    with console.status("[bold green]Loading configuration..."):
        config = read_config('config.yaml')
    with console.status("[bold green]Initializing LLM..."):
        llm = initialize_llm(config)
    chatbot(config, llm)