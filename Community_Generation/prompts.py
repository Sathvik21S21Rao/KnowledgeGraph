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

