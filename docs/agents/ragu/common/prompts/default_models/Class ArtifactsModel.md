# Class ArtifactsModel (defined in ragu/common/prompts/default_models.py at lines 65-88)

class ArtifactsModel(pydantic.BaseModel):
...

    entities: typing.List[ragu.common.prompts.default_models.EntityModel] = Field(
        default_factory=list,
        description="List of extracted entities"
    )

    relations: typing.List[ragu.common.prompts.default_models.RelationModel] = Field(
        default_factory=list,
        description="List of extracted relationships"
    )