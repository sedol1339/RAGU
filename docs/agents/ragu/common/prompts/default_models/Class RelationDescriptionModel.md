# Class RelationDescriptionModel (defined in ragu/common/prompts/default_models.py at lines 140-144)

class RelationDescriptionModel(pydantic.BaseModel):
...

    subject_name: str = Field(description="Subject entity name")

    object_name: str = Field(description="Object entity name")

    description: str = Field(description="Summarized description of the relationship between the entities")