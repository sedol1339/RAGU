# Class GlobalSearchContextModel (defined in ragu/common/prompts/default_models.py at lines 120-124)

class GlobalSearchContextModel(pydantic.BaseModel):
...

    reasoning: str = Field(..., description="Reasoning about the relevance of the context")

    response: str = Field(..., description="Answer to the query")

    rating: pydantic.confloat(ge=0, le=10) = Field(..., description="Relevance rating of the context 0–10")