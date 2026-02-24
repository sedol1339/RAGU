# Class InMemoryGraphBuilder (defined in ragu/graph/graph_builder_pipeline.py at lines 86-299)

class InMemoryGraphBuilder:
    """
    High-level orchestrator for extracting and summarizing entities and relations
    directly in memory using an LLM client and supporting components.

    The pipeline consists of:
      1. **Chunking** input documents.
      2. **Entity & relation extraction** using a triplet-based artifact extractor.
      3. **Artifact summarization** for merging and deduplicating similar entities.
      4. (Optional) **Additional modules** for graph enrichment.
      5. **Community summarization** (aggregated graph-level summaries).

    When `build_parameters.build_only_vector_context=True`, steps 2-5 are skipped,
    and only chunking is performed. This is useful for naive vector RAG where only
    chunk embeddings are needed without knowledge graph construction.

    :param client: LLM client used for understanding and summarization tasks.
    :param chunker: Module responsible for splitting documents into chunks.
    :param artifact_extractor: Extractor for entities and relations from chunks.
    :param build_parameters: Graph-building settings controlling summarization,
        clustering, and optimization behavior.
    :param embedder: Embedding model used for vectorization and clustering.
    :param llm_cache_flush_every: Number of LLM calls between cache flushes.
    :param embedder_cache_flush_every: Number of embedder calls between cache flushes.
    :param additional_pipeline: Optional post-processing modules executed after
        extraction/summarization.
    :param language: Working language for prompts and generation.
    """
    ...

    async def extract_graph(
            self, chunks: typing.List[ragu.chunker.types.Chunk]
    ) -> typing.Tuple[typing.List[ragu.graph.types.Entity], typing.List[ragu.graph.types.Relation], typing.List[ragu.graph.types.CommunitySummary], typing.List[ragu.graph.types.Community], typing.List[ragu.chunker.types.Chunk]]:
        """
        Run the full extraction pipeline and produce entities, relations,
        community summaries, and communities.

        Pipeline:
          1. Extract entities/relations via :class:`BaseArtifactExtractor`
             (skipped if ``build_only_vector_context=True``).
          2. Summarize or merge similar artifacts.
          3. Detect communities and generate summaries (optional).

        :param chunks: list of input text documents.
        :return:
            A tuple ``(entities, relations, summaries, communities)`` where
              - **entities** (:class:`list[Entity]`) — extracted and summarized entities (empty if build_only_vector_context=True).
              - **relations** (:class:`list[Relation]`) — extracted and summarized relations (empty if build_only_vector_context=True).
              - **summaries** (:class:`list[CommunitySummary]`) — generated summaries for detected communities.
              - **communities** (:class:`list[Community]`) — graph communities detected via Leiden clustering.
              - **chunks** (:class:`list[Chunk]`) - list of chunks extracted from input documents.
        """
        ...

    async def cluster_graph(
        self,
        entities: typing.List[ragu.graph.types.Entity],
        relations: typing.List[ragu.graph.types.Relation],
    ) -> typing.List[ragu.graph.types.Community]:
        """
        Detect graph communities with hierarchical Leiden clustering.

        Builds an undirected graph from entities/relations and do clusterization.

        :param entities: Entities.
        :param relations: Relations.
        :return: Detected communities.
        """
        ...