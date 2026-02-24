# Variable DEFAULT_ARTIFACTS_EXTRACTOR_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 1-37)

DEFAULT_ARTIFACTS_EXTRACTOR_PROMPT = """
**-Goal-**  
A text document and a list of entity types are given. 
It is necessary to identify all entities of the specified types in the text, as well as all relationships between the identified entities.  

**-Steps-**  
1. **Identify all entities.**  
    For each detected entity, extract the following information:  
    - **entity_name**: The normalized name of the entity, starting with a capital letter.  
        Normalization means reducing the word to its base form.  
        Example: рождеству → Рождество, кошек → Кошки, Павла → Павел.  
    - **entity_type**: The type of the entity.  
    {% if entity_types -%}
        The entity type must be one of the following: {{ entity_types }}
    {% endif %}    
    - **description**: A detailed description of the entity according to the given text. The description must be precise and as complete as possible.  

2. **Determine relationships between entities.**  
    Based on the entities identified in step 1, determine all pairs (**source_entity**, **target_entity**) that are *explicitly connected* to each other.  
    For each such pair, extract the following information:  
    - **source_entity**: The name of the source entity (as defined in step 1).  
    - **target_entity**: The name of the target entity (as defined in step 1).  
    - **relation_type**: The type of the relation.  
    {% if relation_types -%}
        The relation type must be one of the following: {{ relation_types }}
    {% endif %}  
    - **description**: A description of the relationship between the two entities.  
    - **relationship_strength**: A numeric value representing the strength of the relationship between the entities, ranging from 0 to 5, 
    where 0 = weak connection and 5 = strong connection.  

Text:  
{{ context }}  

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""