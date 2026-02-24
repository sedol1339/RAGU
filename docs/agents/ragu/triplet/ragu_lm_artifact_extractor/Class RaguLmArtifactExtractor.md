# Class RaguLmArtifactExtractor (defined in ragu/triplet/ragu_lm_artifact_extractor.py at lines 37-501)

class RaguLmArtifactExtractor(ragu.triplet.base_artifact_extractor.BaseArtifactExtractor):
    """
    RAGU-LM artifact extractor with stage-by-stage batch processing.
    """
    ...

    async def extract(
        self,
        chunks: typing.List[ragu.chunker.types.Chunk],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Tuple[typing.List[ragu.graph.types.Entity], typing.List[ragu.graph.types.Relation]]:
        """
        Run optimized knowledge extraction pipeline via RAGU-LM.

        Uses stage-by-stage batch processing for better vLLM utilization:
        1. Extract entities from chunks
        2. Normalize entities across chunks
        3. Generate descriptions for entities
        4. Extract relations for inner product of entities from every chunk

        :param chunks: Text chunks to process.
        :return: Tuple of (entities, relations) extracted from all chunks.
        """
        ...

    @staticmethod
    def filter_relations(
            relations: typing.List[ragu.graph.types.Relation],
            negative_pattern: typing.Optional[typing.Union[str, re.Pattern[str]]] = None,
    ) -> typing.List[ragu.graph.types.Relation]:
        """
        Filter out empty, irrelevant, or negated relations.
        """
        ...