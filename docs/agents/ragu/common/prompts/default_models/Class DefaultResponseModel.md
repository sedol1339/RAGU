# Class DefaultResponseModel (defined in ragu/common/prompts/default_models.py at lines 131-133)

class DefaultResponseModel(pydantic.BaseModel):
...

    response: str = Field(..., description="Answer based on the provided context; if unknown, explicitly state so")