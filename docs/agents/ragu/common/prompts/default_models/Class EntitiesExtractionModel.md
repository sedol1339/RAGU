# Class EntitiesExtractionModel (defined in ragu/common/prompts/default_models.py at lines 90-95)

class EntitiesExtractionModel(pydantic.BaseModel):
...

    entities: typing.List[ragu.common.prompts.default_models.EntityModel] = Field(
        default_factory=list,
        description="List of entities extracted from text"
    )