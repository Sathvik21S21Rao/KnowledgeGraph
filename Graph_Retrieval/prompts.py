retrieve_nodes_prompt = """
-Goal-
To retrieve nodes related to a query based on the given context.
-Instructions-
1.Retrive the related nodes based only on the following context and chat history.
2. The nodes must be retrieved must be relavant to the query
2.Return the output as shown in the example. Do not use the nodes in the example as context.
3.Do not hallucinate new nodes. If the query is not related to any node, return an empty list.
4.If the query requests for a summary of the entire doc, return all the nodes.
        
Example output:
{{"node_names": ["node1", "node2", "node3"]}}
        
Nodes: {context}
Chat History: {chat_history}
Find all the nodes related to the following query based on the context.
        
Query: {query}
        

Output:
"""