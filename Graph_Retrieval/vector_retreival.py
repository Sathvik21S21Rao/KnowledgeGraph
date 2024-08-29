from langchain_community.vectorstores import Chroma


class VectorStore:
    def __init__(self, embedding, persist_dir, collection_name, documents=None, metadata=None,create=False,update=False):
        self.embedding = embedding
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.collection_name = collection_name
        self.documents = documents
        self.metadata = metadata
        if create:
            self._create_vectorstore()
        else:
            self._load_vectorstore()
            if update:
                self._update_vectorstore()

    def _create_vectorstore(self):
 
        if self.documents and self.metadata and len(self.documents) == len(self.metadata):
            self.vectorstore = Chroma.from_texts(
                texts=self.documents,
                metadatas=self.metadata,
                collection_name=self.collection_name,
                embedding=self.embedding,
                persist_directory=self.persist_dir
            )
        else:
            raise ValueError("Documents and metadata must be provided and have the same length.")
        
        return self.vectorstore
    
    def _load_vectorstore(self):
        self.vectorstore = Chroma(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            embedding_function=self.embedding,
        )
    
    def _update_vectorstore(self):
        if self.vectorstore:
            # Ensure documents and metadata are lists of the same length
            if self.documents and self.metadata and len(self.documents) == len(self.metadata):
                self.vectorstore.add_texts(texts=self.documents, metadatas=self.metadata)
            else:
                raise ValueError("Documents and metadata must be provided and have the same length.")
        else:
            raise ValueError("Vector store is not initialized.")
    
    def get_vectorstore(self):
        if self.vectorstore is None:
            self.create_vectorstore()
        return self.vectorstore
    
    def get_closest_documents(self, query, k=5):
        return self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 10}).invoke(query)

