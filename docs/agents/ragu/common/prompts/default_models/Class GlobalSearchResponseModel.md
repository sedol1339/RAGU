# Class GlobalSearchResponseModel (defined in ragu/common/prompts/default_models.py at lines 126-129)

class GlobalSearchResponseModel(pydantic.BaseModel):
...

    reasoning: str = Field(..., description="Reasoning about context relevance and the final answer")

    response: str = Field(..., description="Final answer")