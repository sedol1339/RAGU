# Class KnowledgeGraph (defined in ragu/graph/knowledge_graph.py at lines 23-380)

class KnowledgeGraph:
    """
    High-level facade for building, storing, and querying a knowledge graph.

    :param client: LLM client used by extraction and summarization modules.
    :param embedder: Embedder used for vector storage and clustering/similarity steps.
    :param chunker: Optional chunker used to split input documents.
    :param artifact_extractor: Optional extractor used to generate entities/relations from chunks.
    :param builder_settings: Graph-building behavior configuration. Defaults are used if omitted.
    :param storage_settings: Storage backend configuration. Defaults are used if omitted.
    :param additional_modules: Optional post-processing modules for extracted graph items.
    :param language: Optional language override. Defaults to ``Settings.language``.
    """
    ...

    async def build_from_docs(self, docs: typing.List[str]) -> "KnowledgeGraph":
        """
        Build graph and vector context from a list of input documents.

        :param docs: Input documents to process.
        :return: Self for method chaining.
        """
        ...

    async def insert_entities(self, entities: ragu.graph.types.Entity | typing.List[ragu.graph.types.Entity]) -> "KnowledgeGraph":
        """
        Add one or more entities to the knowledge graph.

        Entities with duplicate (name, type) will be automatically merged.

        :param entities: Single entity or list of entities to add.
        :return: Self for method chaining.
        """
        ...

    async def update_entities(self, entities: ragu.graph.types.Entity | typing.List[ragu.graph.types.Entity]) -> "KnowledgeGraph":
        """
        Replace one or more existing entities by ID.

        :param entities: Single entity or list of entities to replace.
        :return: Self for method chaining.
        """
        ...

    async def add_entity(self, entities: ragu.graph.types.Entity | typing.List[ragu.graph.types.Entity]) -> "KnowledgeGraph":
        """
        Backward-compatible alias for :meth:`insert_entities`.

        :param entities: Single entity or list of entities to add.
        :return: Self for method chaining.
        """
        ...

    async def get_entity(self, entity_id) -> ragu.graph.types.Entity | None:
        """
        Retrieve one entity by ID.

        :param entity_id: Entity identifier.
        :return: Entity if found, otherwise ``None``.
        """
        ...

    async def delete_entity(self, entity_id: str) -> "KnowledgeGraph":
        """
        Delete an entity from the knowledge graph.

        :param entity_id: ID of the entity to delete.
        :return: Self for method chaining.
        """
        ...

    async def update_entity(self, entity_id: str, new_entity: ragu.graph.types.Entity) -> "KnowledgeGraph":
        """
        Replace an entity's data while keeping its ID and graph connections.

        :param entity_id: ID of the entity to update.
        :param new_entity: Entity with updated fields.
        :return: Self for method chaining.
        :raises ValueError: If the entity does not exist.
        """
        ...

    async def insert_relations(self, relation: ragu.graph.types.Relation | typing.List[ragu.graph.types.Relation]) -> "KnowledgeGraph":
        """
        Add one or more relations to the knowledge graph.

        Relations with duplicate IDs will be automatically merged.
        Validates that referenced entities exist.

        :param relation: Single relation or list of relations to add.
        :return: Self for method chaining.
        """
        ...

    async def update_relations(self, relation: ragu.graph.types.Relation | typing.List[ragu.graph.types.Relation]) -> "KnowledgeGraph":
        """
        Replace one or more existing relations by ID.

        :param relation: Single relation or list of relations to replace.
        :return: Self for method chaining.
        """
        ...

    async def delete_relation(
        self,
        subject_id: str,
        object_id: str,
        relation_id: str | None = None,
    ) -> "KnowledgeGraph":
        """
        Delete a relation from the knowledge graph.

        :param subject_id: Subject entity ID.
        :param object_id: Object entity ID.
        :param relation_id: Optional relation ID for precise delete.
        :return: Self for method chaining.
        """
        ...

    async def edges_degrees(self, edge_specs: typing.List[ragu.storage.base_storage.EdgeSpec]) -> typing.List[int]:
        """
        Get degrees for multiple edges.

        Each returned value is ``degree(source) + degree(target)`` for the
        corresponding edge spec, or ``0`` when relation/endpoints are missing.

        :param edge_specs: Edge specifications ``(subject_id, object_id, relation_id)``.
        :return: Degree sums in the same order as input specs.
        """
        ...

    async def add_summary(self, summary: ragu.graph.types.CommunitySummary | typing.List[ragu.graph.types.CommunitySummary]) -> "KnowledgeGraph":
        """
        Add one or more community summaries.

        :param summary: Single summary or list of summaries to add.
        :return: Self for method chaining.
        """
        ...

    async def get_summary(self, summary_id: str) -> ragu.graph.types.CommunitySummary | None:
        """
        Retrieve a community summary by ID.

        :param summary_id: ID of the summary to retrieve.
        :return: The summary, or ``None`` if not found.
        """
        ...

    async def delete_summary(self, summary_id: str) -> "KnowledgeGraph":
        """
        Delete a community summary.

        :param summary_id: ID of the summary to delete.
        :return: Self for method chaining.
        """
        ...

    async def update_summary(self, summary_id: str, new_summary: ragu.graph.types.CommunitySummary) -> "KnowledgeGraph":
        """
        Replace a community summary's content.

        :param summary_id: ID of the summary to update.
        :param new_summary: Summary with updated content.
        :return: Self for method chaining.
        """
        ...

    async def find_similar_entities(self, entity: ragu.graph.types.Entity, top_k: int = 10) -> typing.List[ragu.graph.types.Entity]:
        """
        Find entities semantically similar to the given entity.

        :param entity: Reference entity to search against.
        :param top_k: Maximum number of results.
        :return: Similar entities ordered by relevance.
        """
        ...

    async def find_similar_relations(self, relation: ragu.graph.types.Relation, top_k: int = 10) -> typing.List[ragu.graph.types.Relation]:
        """
        Find relations semantically similar to the given relation.

        :param relation: Reference relation to search against.
        :param top_k: Maximum number of results.
        :return: Similar relations ordered by relevance.
        """
        ...

    async def find_similar_entity_by_query(self, query: str, top_k: int = 10) -> typing.List[ragu.graph.types.Entity]:
        """
        Find entities matching a free-text query.

        :param query: Search query text.
        :param top_k: Maximum number of results.
        :return: Matching entities ordered by relevance.
        """
        ...

    async def find_similar_relation_by_query(self, query: str, top_k: int = 10) -> typing.List[ragu.graph.types.Relation]:
        """
        Find relations matching a free-text query.

        :param query: Search query text.
        :param top_k: Maximum number of results.
        :return: Matching relations ordered by relevance.
        """
        ...