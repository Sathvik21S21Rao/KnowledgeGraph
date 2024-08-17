import warnings
warnings.filterwarnings("ignore")

import unittest
import pickle
import os
import time
import networkx as nx
from checkDataUpdates.checkFileUpdates import SyncData, create_temp_folder
from Graph_Generation.graph_extraction import GraphExtractionChain, DataLoader, PrepareDataForNX, UpdateGraph
from Community_Generation.communitySummary import UpdateCommunities
from Graph_Retrieval.context_based_node_retrieval import ContextBasedNodeRetrieval
from Graph_Retrieval.query import Query
from testing_data import intitial_data, update_data


# Function definitions
def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        return pickle.load(f)

def save_graph(graph, graph_path):
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)

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
    
def initialize_embedding_model(config):
    embedding_server = config.get("embedding_server").lower()

    if embedding_server == "ollama":
        from langchain_community import embeddings
        return embeddings.OllamaEmbeddings(model=config.get("embedding_model", ""))
    elif embedding_server == "huggingface":
        from langchain_community import embeddings
        return embeddings.HuggingFaceEmbeddings(model=config.get("embedding_model", ""))
    elif embedding_server == "openai":
        from langchain_community import embeddings
        return embeddings.OpenAIEmbeddings(model=config.get("embedding_model", ""), api_key=config.get("api_key"))
    elif embedding_server == "local":
        return None


api_key = os.environ.get("API_KEY")

config = {
    "data_path": "data_test",
    "chunk_size": 512,
    "chunk_overlap": 128,
    "api_key": api_key,
    "server": "Groq",
    "model": "llama-3.1-8b-instant",
    "temperature": 0.5,
    "use_sentence_embeddings": False,
    "node2vec_model_path": "./model_test/node2vec.model",
    "sentence_model_path": "./model_test/sentence.model",
    "node2vec_embeddings_path": "./embeddings_test/node2vec_embeddings.npy",
    "graph_file_path": "./graph_test/graph.pkl",
    "collection_name": "node_data_test",
    "node_data_dir": "./node_data_test",
    "community_data_dir": "./community_data_test",
    "sentence_embeddings_path": "embeddings_test/sentence_embeddings.npy",
    "node_names_path": "./embeddings_test/node_names.npy",
    "sentence_model_name": "all-MiniLM-L6-v2",
    "faiss_model_path": "./model_test/faiss.index"
}

graph_file_path = config.get("graph_file_path")
node2vec_model_path = config.get("node2vec_model_path")
node_data_dir = config.get("node_data_dir")
community_data_dir = config.get("community_data_dir")
chunk_size = config.get("chunk_size")
chunk_overlap = config.get("chunk_overlap")
sentence_model_path = config.get("sentence_model_path")
node2vec_embeddings_path = config.get("node2vec_embeddings_path")
sentence_embeddings_path = config.get("sentence_embeddings_path")
node_names_path = config.get("node_names_path")
faiss_model_path = config.get("faiss_model_path")
sentence_model_name = config.get("sentence_model_name")
use_sentence_embeddings = config.get("use_sentence_embeddings")
llm = initialize_llm(config)


# Unit test class for graph creation
class TestGraphCreation(unittest.TestCase):

    def setUp(self):
        """Setup the initial graph and other resources"""
        os.mkdir(config["data_path"])
        with open(config["data_path"] + "/data", "w") as f:
            f.write(intitial_data)
        create_temp_folder(config["data_path"],community_data_dir)

    def test_graph_creation(self):
        """Test graph creation from data"""
        chain = GraphExtractionChain(llm=llm)
        data = DataLoader(path=config["data_path"], chunk_size=chunk_size, chunk_overlap=chunk_overlap).load()
        NxData = PrepareDataForNX().execute(data, chain)
        graph = nx.Graph()
        graph.add_nodes_from(NxData[0])
        graph.add_edges_from(NxData[1])
        save_graph(graph, graph_file_path)
        obj = ContextBasedNodeRetrieval(llm, graph, node2vec_model_path, node_data_dir, community_data_dir, True)
        obj.setup()
        self.assertTrue(os.path.exists(graph_file_path))
        loaded_graph = load_graph(graph_file_path)
        self.assertIsInstance(loaded_graph, nx.Graph)
        self.assertEqual(len(graph.nodes), len(loaded_graph.nodes))
        self.assertEqual(len(graph.edges), len(loaded_graph.edges))

class TestGraphUpdate(unittest.TestCase):
    def setUp(self):
        """Setup the initial graph and other resources"""
        # Create an initial graph
        self.graph=load_graph(graph_file_path)
        time.sleep(10)

    def test_graph_updation(self):
        """Test graph updation after detecting changes in the data"""
        
        with open(config["data_path"] + "/data", "a") as f:
            f.write(update_data)
        sync = SyncData(folder=config["data_path"], temp_folder="./.temp")
        updates = sync.compareFolders()
        
        if updates:
            updates = "\n".join(updates)
            updates = DataLoader(path=None, chunk_overlap=chunk_overlap, 
            chunk_size=chunk_size).load_text(updates)
            sync.syncTempFolder()
            chain = GraphExtractionChain(llm=llm)
            updated_graph = load_graph(graph_file_path)
            updated_nodes, added_nodes, added_edges = UpdateGraph(
                graph=updated_graph, graph_path=graph_file_path
            ).execute(updates, chain)
            
            updated_nodes = [node[0] for node in updated_nodes + added_nodes]
            
            # Update communities based on the new graph structure
            UpdateCommunities(
                graph=updated_graph, llm=llm, community_dir=community_data_dir,
                create=False, updated_nodes=updated_nodes
            ).update_communities()
            
            save_graph(updated_graph, graph_file_path)
        
        # Load the updated graph and perform assertions
        loaded_graph = load_graph(graph_file_path)
        self.assertIsInstance(loaded_graph, nx.Graph)
        self.assertGreaterEqual(len(loaded_graph.nodes), len(self.graph.nodes))
        self.assertGreaterEqual(len(loaded_graph.edges), len(self.graph.edges))
    
class TestQuery(unittest.TestCase):
    def setUp(self):
        """Setup the initial graph and other resources"""
        # Create an initial graph
        self.graph = load_graph(graph_file_path)
        self.obj = ContextBasedNodeRetrieval(llm, self.graph, node2vec_model_path,
        node_data_dir, community_data_dir, False)
        self.obj.setup()
        
    def test_query(self):
        """Test query execution"""
        query = "What is the story of the lost expedition?"
        context = self.obj.get_context(query=query)
        response = Query(context, query, llm, chat_history=[]).execute_query()
        self.assertIsInstance(response, str)
        self.assertTrue(response)


