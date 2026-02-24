# Class ClusterSummarizationModel (defined in ragu/common/prompts/default_models.py at lines 146-148)

class ClusterSummarizationModel(pydantic.BaseModel):
...

    content: str = Field(description="Summarized content of the cluster")