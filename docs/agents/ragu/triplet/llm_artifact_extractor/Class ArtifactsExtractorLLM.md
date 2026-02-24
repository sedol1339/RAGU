# Class ArtifactsExtractorLLM (defined in ragu/triplet/llm_artifact_extractor.py at lines 15-162)

class ArtifactsExtractorLLM(ragu.triplet.base_artifact_extractor.BaseArtifactExtractor):
    """
    Extracts entities and relations from text chunks using LLM.

    Pipeline:
      1. Render the `artifact_extraction` instruction in batch mode over chunk texts.
      2. Call the LLM to produce structured artifacts for each chunk.
      3. Optionally render and run `artifact_validation` to refine extracted artifacts.
      4. Convert model outputs into Entity/Relation objects, preserving source chunk ids.
    """
    ...

    async def extract(self, chunks: typing.List[ragu.chunker.types.Chunk], *args, **kwargs) -> typing.Tuple[typing.List[ragu.graph.types.Entity], typing.List[ragu.graph.types.Relation]]:
        """
        Extract entities and relations from a collection of chunks.

        Steps:
          1) Batch-render the extraction prompt with `context=<chunk_texts>`,
          2) Generate structured artifacts per chunk,
          3) Optionally validate artifacts against the original context,
          4) Convert artifacts into Entity/Relation objects.

        :param chunks: Iterable of Chunk objects.
        :return: (entities, relations) extracted from all chunks.
        """
        ...