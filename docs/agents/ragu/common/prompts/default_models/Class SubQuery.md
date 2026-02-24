# Class SubQuery (defined in ragu/common/prompts/default_models.py at lines 149-168)

class SubQuery(pydantic.BaseModel):
...

    id: str = Field(..., description="Unique identifier of the subquery, e.g. 'q1', 'q2'")

    query: str = Field(..., description="Natural language formulation of the atomic subquery")

    depends_on: typing.List[str] = Field(
        default_factory=list,
        description="List of subquery IDs that must be resolved before this one"
    )

    intent: typing.Optional[typing.Literal[
        "lookup",
        "definition",
        "comparison",
        "aggregation",
        "reasoning",
        "filtering",
        "other"
    ]] = Field(
        default=None,
        description="Optional classification of the subquery intent"
    )