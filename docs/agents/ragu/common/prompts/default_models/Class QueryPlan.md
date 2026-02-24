# Class QueryPlan (defined in ragu/common/prompts/default_models.py at lines 169-171)

class QueryPlan(pydantic.BaseModel):
...

    subqueries: typing.List[ragu.common.prompts.default_models.SubQuery] = Field(..., description="List of decomposed subqueries forming a DAG")