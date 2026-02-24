# Class CommunityReportModel (defined in ragu/common/prompts/default_models.py at lines 109-118)

class CommunityReportModel(pydantic.BaseModel):
...

    title: str = Field(..., description="Report title")

    summary: str = Field(..., description="Short summary of the community")

    rating: pydantic.confloat(ge=0, le=10) = Field(..., description="Impact rating 0–10")

    rating_explanation: str = Field(..., description="Explanation of the rating")

    findings: typing.List[ragu.common.prompts.default_models.CommunityFindingModel] = Field(
        default_factory=list,
        description="List of 5–10 key findings"
    )