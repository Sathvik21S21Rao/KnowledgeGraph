community_prompt = """
-Goal-
Given a community with nodes along with their description, Generate a summary of the community.

-Instructions-
1. Look at the nodes in the community, along with their descriptions.Identify key aspects and relationships between nodes.
2. Generate a summary of the community based on the key aspects and relationships.
3. The summary should be comprehensive and capture the essence of the community.
4. Do not miss out on any information.
5. Return the Output as shown in the example.Do not return reasoning of your output.

-Example Output-
{{"community_name": "India a diverse country", "community_description": "India is a diverse country with a rich cultural heritage. It is known for its diverse landscapes, languages, and traditions. The country has a long history and has made significant contributions to various fields such as science, art, and literature."}}

-Input-
{input_text}

Output:
"""

update_community_prompt = """
-Goal-
Given a community summary and updated nodes, integrate the changes from the updated nodes into the existing summary.

-Instructions-  
1. Compare the existing summary with the updated nodes to identify changes.
2. Do not provide explanations or reasoning for the output.
3. Ensure that information is not repeated between the updated nodes and the existing summary. Avoid redundancy.
4. Return only the new sentences, specifying the exact indices where they should be inserted within the existing summary.
5. Do not repeat or rewrite the entire summary; only return the new sentences and their corresponding indices.

-Example Output-  

{
  "new_sentences": ["India is known for its diverse landscapes, languages, and traditions."],
  "indices": [2]
}


**Input:**  
Existing Summary: {existing_summary}  
Updated Nodes: {updated_nodes}

"""

