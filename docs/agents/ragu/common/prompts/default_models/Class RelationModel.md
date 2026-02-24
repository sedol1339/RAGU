# Class RelationModel (defined in ragu/common/prompts/default_models.py at lines 55-63)

class RelationModel(pydantic.BaseModel):
...

    source_entity: str = Field(..., description="Name of the source entity (matches an Entity.entity_name)")

    target_entity: str = Field(..., description="Name of the target entity (matches an Entity.entity_name)")

    relation_type: str = Field(..., description="Type of relation")

    description: str = Field(..., description="Description of the relationship")

    relationship_strength: pydantic.conint(ge=0, le=5) = Field(
        ..., description="Relationship strength 0–5 (0 = weak, 5 = strong)"
    )