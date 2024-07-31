# GraphRAG

GraphRAG is a **Python project** that uses **graph-based methods** for information retrieval. It uses **language models** and **embeddings** to create and interact with a **graph of data**.

## Project Structure

The project is structured as follows:

- `main.py`: The **main entry point** of the application.
- `Graph_Generation/`: Contains the code for **generating the graph**.
- `Graph_Retrieval/`: Contains the code for **retrieving information** from the graph.
- `config.yaml`: **Configuration file** for the application.
- `requirements.txt`: Lists the **Python dependencies** required for the project.

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/Sathvik21S21Rao/KnowledgeGraph.git
```

2. **Update** the `config.yaml` file, with relevant **API keys** and models to be used.

3. **Build the docker image** in the working directory
```bash
docker build -t graphrag .
```

4. **Run the docker container** in interactive mode
```bash
docker run -it graphrag
```

5. In case you are running an **Ollama model locally**
```bash
docker run -it --network host graphrag
```

## Configuration

The application's behavior can be configured by modifying `config.yaml`. This includes the paths to the graph, node data, and community data, as well as the settings for the **language model** and **embeddings**.

## Language Model

The application supports different language models and embeddings, which can be configured in `config.yaml`. The supported language models are **OpenAI, Ollama, and Groq**. For OpenAI and Groq fill in the `api_key` field.

## Graph Generation and Retrieval

**Graph Generation** is done with the help of the **LLM**, by identifying entities and relationships. The prompts for generating graphs have been taken from [Microsoft's graphrag](https://github.com/microsoft/graphrag) with modifications. The graphs are then analyzed using **node2vec**, which produces graph embeddings. The graph, the corresponding node names, and the embedding models are stored. **Girvan Newman technique** has been used to identify first-level communities, which are summarized.

**Graph Retrieval** is used during query. The **LLM** is used twice: once for extracting relevant nodes from the graph and second for answering the query. After extracting relevant nodes, the **node2vec model** is used to enrich the context by retrieving nodes that are similar to the present nodes (using graph embeddings). If the nodes belong to more than 40% of the communities, then the context is switched to the **community summaries**. Otherwise, the context format is as follows: **Nodes and their Description** | **Edges and their Description**.

This is done to ensure the **size of the context is optimized** and relevant information stays intact.

## Interacting with the Application

When running `main.py`, you will be asked whether you want to create a **new graph**. If you choose not to create a new graph, the program will attempt to **load a graph** from the specified path.

Once the application is running, you can interact with it by typing **queries** into the console. The application will respond with information retrieved from the graph based on your query. To end the conversation, type **'exit'**.

## Further Improvements

- Integrating graphs with **vector-based retrieval**.
- Enriching answers by retrieval from the internet using **DuckDuckGo**.