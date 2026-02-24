# Class EntityDescriptionModel (defined in ragu/common/prompts/default_models.py at lines 135-138)

class EntityDescriptionModel(pydantic.BaseModel):
...

    entity_name: str = Field(description="Entity name")

    description: str = Field(description="Summarized description")