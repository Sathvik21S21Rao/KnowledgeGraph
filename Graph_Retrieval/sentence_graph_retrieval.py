import networkx as nx
from sentence_transformers import SentenceTransformer
from node2vec import Node2Vec
import numpy as np
from gensim.models import Word2Vec
import faiss
import pickle

def softmax(x):
    e_x = np.exp(-x)
    return e_x / e_x.sum()

class SentenceGraphRetrieval:
    def __init__(self, graph, create, node2vec_model_path,sentence_model_path,node2vec_embeddings_path,sentence_embeddings_path,node_names_path,faiss_index_path,sentence_model_name='all-MiniLM-L6-v2',):

        self.graph = graph
        self.is_created = False
        self.node2vec_model_path = node2vec_model_path
        self.sentence_model_name = sentence_model_name
        self.node_names_path=node_names_path
        self.node2vec_embeddings_path=node2vec_embeddings_path
        self.sentence_embeddings_path=sentence_embeddings_path
        self.sentence_model_path=sentence_model_path
        self.sentence_model = None
        self.node2vec_model = None
        self.sentence_embeddings = None
        self.node_names = None
        self.faiss_index = None
        self.faiss_index_path=faiss_index_path

        if self.is_created:
            self._load_models()
        else:
            self._create_models()

    def _load_models(self):
        
        self.sentence_model = SentenceTransformer.load(self.sentence_model_path)

      
        if self.node2vec_model_path:
            self.node2vec_model = Word2Vec.load(self.node2vec_model_path)

        self.sentence_embeddings = np.load(self.sentence_embeddings_path)
        self.node_names = np.load(self.node_names_path)

        with open(self.faiss_index_path,"rb") as f:
            self.faiss_index=pickle.load(f)
        
        self.is_created = True

    def _create_models(self):
        self.sentence_model = SentenceTransformer(self.sentence_model_name)

        self.node2vec_model = self._create_node2vec_model()

        self.node_names = list(self.graph.nodes())
        self.sentence_embeddings = np.array([self._get_sentence_embedding(name) for name in self.node_names])
        
        self.faiss_index = faiss.IndexFlatL2(self.sentence_embeddings.shape[1])
        self.faiss_index.add(self.sentence_embeddings)

        with open(self.faiss_index_path,"wb") as f:
            pickle.dump(self.faiss_index,f)

        self.sentence_model.save(self.sentence_model_path)
        self.node2vec_model.save(self.node2vec_model_path)
        np.save(self.sentence_embeddings_path, self.sentence_embeddings)
        np.save(self.node_names_path, np.array(self.node_names))

    def _create_node2vec_model(self):
        # Initialize and train Node2Vec model
        node2vec = Node2Vec(self.graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        return model

    def _get_sentence_embedding(self, string):
        return self.sentence_model.encode(string)

    def find_most_similar(self, node_name, p=0.6,score_thresh=0.5,top_k=5):
        target_sentence_embedding = self._get_sentence_embedding(node_name).reshape(1, -1)
        
        distances, indices = self.faiss_index.search(target_sentence_embedding, len(self.node_names))  # Query all nodes

        sorted_indices = indices[0]
        distances[0]=(distances[0]-np.mean(distances[0]))/np.std(distances[0])
        sorted_distances = softmax(distances[0])
        sorted_node_names = np.array(self.node_names)[sorted_indices]
        cumulative_prob = np.cumsum(sorted_distances / np.sum(sorted_distances))
        top_p_indices = np.where(cumulative_prob <= p)[0]
        
        similar_nodes_by_sentence = [sorted_node_names[i] for i in top_p_indices]
        
        similar_nodes_by_node2vec = [self.node2vec_model.wv.most_similar(node_name) for node_name in similar_nodes_by_sentence]
        
        similar_nodes=set()
        for i, nodes in enumerate(similar_nodes_by_node2vec):
            similar_nodes.update([node[0] for node in nodes if node[1]>score_thresh])

        similar_edges=[]
        
        for node1 in similar_nodes_by_sentence:
            for node2 in similar_nodes:
                if self.graph.has_edge(node1,node2):
                    similar_edges.append((node1,node2))
            

        return similar_nodes,similar_edges
    
        
    def _get_node_descriptions(self, nodes)->dict:
        return {node:self.graph.nodes[node].get("description") for node in nodes}
    
    def _get_edge_descriptions(self,edges)->dict:
        return {edge:self.graph.to_undirected()[edge[0]][edge[1]].get("description") for edge in edges}
    def get_context(self, query, p=0.9, score_thresh=0.8)->str:
        similar_nodes,similar_edges = self.find_most_similar(query, p, score_thresh)
       
        node_descriptions = self._get_node_descriptions(similar_nodes)
        edge_descriptions = self._get_edge_descriptions(similar_edges)
        context = "Nodes and their descriptions:\n"
        for node, description in node_descriptions.items():
            context += f"{node}: {description}\n"

        context += "\nEdges and their descriptions:\n"
        for edge,description in edge_descriptions.items():
            context += f"{edge[0]} <-> {edge[1]}: {description}\n"
        return context
