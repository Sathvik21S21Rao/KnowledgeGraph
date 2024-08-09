retrieve_nodes_prompt = """
-Goal-
The goal is to retrieve nodes in a graph so as to answer the query.
-Instructions-
1.Retrieve the nodes that are required to answer the query. You can refer the chat history for corefoerence resolution.
2. The nodes must be retrieved must be relavant to the query
2.Return the output as shown in the example. Do not use the nodes in the example as context.
3.Do not hallucinate new nodes. If the query is not related to any node, return an empty list.
4. In case the query asks for a summary return all the nodes
Example output:
{{"node_names": ["node1", "node2", "node3"]}}
        
Nodes: {context}
Chat History: {chat_history}
Find all the nodes related to the following query based on the context.
        
Query: {query}
        

Output:
"""

query_template = """Based on the following context answer the query and the following chat history\n\nContext:{context}\n\nChat history:{chat_history}\n\n Query: {query}. Do not give reasoning of the answer. Just answer the query without missing out on any detail."""