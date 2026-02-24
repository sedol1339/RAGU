# Class RewriteQuery (defined in ragu/common/prompts/default_models.py at lines 172-174)

class RewriteQuery(pydantic.BaseModel):
...

    query: str = Field(..., description="Rewritten query that is self-contained and explicit")