# Class CommunityFindingModel (defined in ragu/common/prompts/default_models.py at lines 104-107)

class CommunityFindingModel(pydantic.BaseModel):
...

    summary: str = Field(..., description="Short description of the finding")

    explanation: str = Field(..., description="Detailed explanation (several paragraphs based on the data)")