# Class BuilderArguments (defined in ragu/graph/graph_builder_pipeline.py at lines 20-52)

@dataclasses.dataclass
class BuilderArguments:
    """
    Configuration settings for the knowledge graph building pipeline.

    This dataclass controls various aspects of graph construction including
    summarization strategies, clustering behavior, and optimization modes.

    :param use_llm_summarization: Enable LLM-based summarization for merging and
        deduplicating similar entity and relation descriptions.
    :param use_clustering: Apply clustering to group similar entities before
        summarization.
    :param build_only_vector_context: Skip entity/relation extraction and build
        only vector context for naive RAG.
    :param make_community_summary: Generate high-level summaries for detected
        graph communities.
    :param remove_isolated_nodes: Remove entities that have no relations.
    :param vectorize_chunks: Generate and store embeddings for text chunks.
    :param cluster_only_if_more_than: Minimum number of entities required before
        clustering is applied.
    :param max_cluster_size: Maximum number of entities per cluster.
    :param random_seed: Random seed for reproducible clustering/community detection.
    """
    ...

    use_llm_summarization: bool = True

    use_clustering: bool = False

    build_only_vector_context: bool = False

    make_community_summary: bool = True

    remove_isolated_nodes: bool = True

    vectorize_chunks: bool = False

    cluster_only_if_more_than: int = 10000

    max_cluster_size: int = 128

    random_seed: int = 42