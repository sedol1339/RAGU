# Class CommunitySummarizer (defined in ragu/graph/community_summarizer.py at lines 14-103)

class CommunitySummarizer(ragu.common.base.RaguGenerativeModule):
    """
    Generates textual summaries for detected graph communities using an LLM.

    The summarization process converts a group of entities or
    relations belonging to the same community into a human-readable report.

    :param client: LLM client used for generating community reports.
    :param language: Language of generated summaries. Defaults to ``Settings.language``.
    """
    ...

    async def summarize(self, communities: typing.List[ragu.graph.types.Community]) -> typing.List[ragu.graph.types.CommunitySummary]:
        """
        Generate structured summaries for a list of graph communities.

        :param communities: Communities to summarize.
        :return: Community summaries aligned with input communities.
        """
        ...

    @staticmethod
    def combine_report_text(report: ragu.common.prompts.default_models.CommunityReportModel) -> str:
        """
        Merge structured sections of a community report into a readable text block.

        :param report: Structured community report.
        :return: Rendered report text.
        """
        ...