# Class RelationsExtractionModel (defined in ragu/common/prompts/default_models.py at lines 97-102)

class RelationsExtractionModel(pydantic.BaseModel):
...

    relations: typing.List[ragu.common.prompts.default_models.RelationModel] = Field(
        default_factory=list,
        description="List of relationships between provided entities"
    )