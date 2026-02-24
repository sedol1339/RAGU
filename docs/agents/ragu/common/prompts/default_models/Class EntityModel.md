# Class EntityModel (defined in ragu/common/prompts/default_models.py at lines 49-53)

class EntityModel(pydantic.BaseModel):
...

    entity_name: str = Field(..., description="Normalized entity name, capitalized")

    entity_type: str = Field(..., description="Entity type")

    description: str = Field(..., description="Detailed description of the entity from the text")