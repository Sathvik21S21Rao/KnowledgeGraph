ENTITY_TYPE_PROMPT = """
The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
The user's task is to {task}.
As part of the analysis, you want to identify the entity types present in the following text.
The entity types must be relevant to the user task.
Avoid general entity types such as "other" or "unknown".
This is VERY IMPORTANT: Do not generate redundant or overlapping entity types. For example, if the text contains "company" and "organization" entity types, you should return only one of them.
Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.
And remember, it is ENTITY TYPES what we need.
Return the entity types in as a list of comma sepparated of strings.
=====================================================================
EXAMPLE SECTION: The following section includes example output. These examples **must be excluded from your answer**.

EXAMPLE 1
Task: Determine the connections and organizational hierarchy within the specified community.
Text: Example_Org_A is a company in Sweden. Example_Org_A's director is Example_Individual_B.
RESPONSE:
organization, person
END OF EXAMPLE 1

EXAMPLE 2
Task: Identify the key concepts, principles, and arguments shared among different philosophical schools of thought, and trace the historical or ideological influences they have on each other.
Text: Rationalism, epitomized by thinkers such as René Descartes, holds that reason is the primary source of knowledge. Key concepts within this school include the emphasis on the deductive method of reasoning.
RESPONSE:
concept, person, school of thought
END OF EXAMPLE 2

EXAMPLE 3
Task: Identify the full range of basic forces, factors, and trends that would indirectly shape an issue.
Text: Industry leaders such as Panasonic are vying for supremacy in the battery production sector. They are investing heavily in research and development and are exploring new technologies to gain a competitive edge.
RESPONSE:
organization, technology, sectors, investment strategies
END OF EXAMPLE 3

======================================================================

======================================================================
REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate Entity Types only.
Task: {task}
Text: {input_text}
RESPONSE:
{{<entity_types>}}
"""

ENTITY_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]. Do not hallucinate entity types.
- entity_description: Comprehensive description of the entity's attributes and activities
2. Do not have pronouns as entities (for example, he/she, him/her, they/them, etc.); use the previously recognized entities as a reference. Do not create variation names for the same entity; use the previous entities as a reference. Previous entities: {prev_entities}
3. If an entity is mentioned multiple times in the text, provide a single description that captures all relevant information.
######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.
Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”
The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.
It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths

#############
Output:


  "entity_name": ["Alex", "Taylor", "Jordan", "Cruz"],
  "entity_type": ["person", "person", "person", "person"],
  "entity_description": [
    "Alex is a member of a team committed to discovery, who noted a transformation in Taylor's behavior.",
    "Taylor showed an unexpected reverence for a device, indicating its potential significance.",
    "Jordan shared a commitment to discovery with Alex and had a moment of mutual understanding with Taylor.",
    "Cruz has a vision of control and order that contrasts with Alex and Jordan's commitment to discovery."
  ]



###############
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

RELATION_PROMPT="""
-Goal-
With the given text document and given entities generate relations between the entities.
From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.

-Steps-

1. For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other


2. If the text contains multiple relationships between the same entities, provide a single description that captures all relevant information.

3. Do not generate variations of entity names, use the one's which are provided.

4. Do not miss out any possible relationship between entities.

 ######################
-Examples-
######################
Example 1:

Entities: Alex,Taylor,Jordan,Cruz,The Device
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:

("Alex","Taylor","Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."),("Alex","Jordan","Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."),("Alex","Jordan","Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."),("Taylor","Jordan","Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."),("Taylor","The Device","Taylor shows reverence towards the device, indicating its importance and potential impact."),("Jordan","Cruz","Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."),("Taylor","The Device","Taylor shows reverence towards the device, indicating its importance and potential impact.")




#############################

-Real Data-
######################
Entities:{entities}
Text: {input_text}
######################

"""