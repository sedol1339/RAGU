# Class Index (defined in ragu/storage/index.py at lines 65-1147)

class Index:
    """
    Manages all storage operations for a knowledge graph.

    Coordinates three storage backends (graph, vector DB, KV) and provides
    batch CRUD operations with cascading deletes and duplicate merging.
    """
    ...

    async def insert_entities(self, entities: typing.List[ragu.graph.types.Entity]) -> "Index":
        """
        Insert entities into graph and vector DB.

        Duplicate IDs in the incoming batch are merged. If an entity with the
        same ID already exists, incoming and existing values are merged.

        :param entities: Entities to insert.
        :return: Self for method chaining.
        """
        ...

    async def update_entities(self, entities: typing.List[ragu.graph.types.Entity]) -> "Index":
        """
        Update entities by ID using replace semantics.

        Existing entities are replaced by incoming payloads. No merge with
        previous values is performed.

        :param entities: Entities to update.
        :return: Self for method chaining.
        :raises ValueError: If entity IDs are missing/duplicated in request or absent in storage.
        """
        ...

    async def insert_relations(self, relations: typing.List[ragu.graph.types.Relation]) -> "Index":
        """
        Insert relations into graph and vector DB.

        Duplicate IDs in the incoming batch are merged. If a relation with the
        same ID already exists, incoming and existing values are merged.

        :param relations: Relations to insert.
        :return: Self for method chaining.
        :raises ValueError: If referenced entities don't exist.
        """
        ...

    async def update_relations(self, relations: typing.List[ragu.graph.types.Relation]) -> "Index":
        """
        Update relations by ID using replace semantics.

        Existing relations are replaced by incoming payloads. No merge with
        previous values is performed.

        :param relations: Relations to update.
        :return: Self for method chaining.
        :raises ValueError: If relation IDs are missing/duplicated in request,
            IDs are absent in storage, or referenced entities don't exist.
        """
        ...

    async def upsert_chunks(self, chunks: typing.List[ragu.chunker.types.Chunk]) -> "Index":
        """
        Insert or update chunks into KV storage (and optionally vector DB).

        :param chunks: Chunks to upsert.
        :return: Self for method chaining.
        """
        ...

    async def reindex_cluster_ids(
        self,
        entities: typing.List[ragu.graph.types.Entity],
        communities: typing.List[ragu.graph.types.Community],
        summaries: typing.Optional[typing.List[ragu.graph.types.CommunitySummary]] = None,
    ) -> tuple[typing.List[ragu.graph.types.Entity], typing.List[ragu.graph.types.Community], typing.List[ragu.graph.types.CommunitySummary]]:
        """
        Remap cluster IDs to be globally unique per level across indexing runs.

        Levels are preserved to keep level-based filtering intact.

        :param entities: Entities whose cluster memberships should be remapped.
        :param communities: Newly generated communities with local cluster IDs.
        :param summaries: Optional summaries linked to community IDs.
        :return: Tuple with remapped entities, remapped communities, and remapped summaries.
        """
        ...

    async def upsert_communities(self, communities: typing.List[ragu.graph.types.Community]) -> "Index":
        """
        Insert or update communities into KV storage.

        :param communities: Communities to upsert.
        :return: Self for method chaining.
        """
        ...

    async def upsert_summaries(self, summaries: typing.List[ragu.graph.types.CommunitySummary]) -> "Index":
        """
        Insert or update community summaries into KV storage.

        :param summaries: Summaries to upsert.
        :return: Self for method chaining.
        """
        ...

    async def delete_entities(self, entity_ids: typing.List[str]) -> "Index":
        """
        Delete entities from graph and vector DB.

        All relations connected to the deleted
        entities are also removed from the relation vector DB.

        :param entity_ids: IDs of entities to delete.
        :return: Self for method chaining.
        """
        ...

    async def delete_relations(self, edge_specs: typing.List[ragu.storage.base_storage.EdgeSpec]) -> "Index":
        """
        Delete relations from graph and vector DB.

        :param edge_specs: List of edge specs ``(subject_id, object_id, relation_id)``.
        :return: Self for method chaining.
        """
        ...

    async def delete_chunks(self, chunk_ids: typing.List[str]) -> "Index":
        """
        Delete chunks from KV and vector storage.

        :param chunk_ids: IDs of chunks to delete.
        :return: Self for method chaining.
        """
        ...

    async def delete_communities(self, community_ids: typing.List[str]) -> "Index":
        """
        Delete communities and their summaries from KV storage.

        :param community_ids: IDs of communities to delete.
        :return: Self for method chaining.
        """
        ...

    async def get_entities(self, entity_ids: typing.List[str]) -> typing.List[typing.Optional[ragu.graph.types.Entity]]:
        """
        Retrieve entities by their IDs.

        :param entity_ids: Entity IDs to fetch.
        :return: List of entities (``None`` for missing).
        """
        ...

    async def get_relations(self, edge_specs: typing.List[ragu.storage.base_storage.EdgeSpec]) -> typing.List[typing.Optional[ragu.graph.types.Relation]]:
        """
        Retrieve relations by edge specs.

        :param edge_specs: List of edge specs ``(subject_id, object_id, relation_id)``.
        :return: List of relations (``None`` for missing).
        """
        ...

    async def get_chunks(self, chunk_ids: typing.List[str]) -> typing.List[typing.Optional[ragu.chunker.types.Chunk]]:
        """
        Retrieve chunks by their IDs.

        :param chunk_ids: Chunk IDs to fetch.
        :return: List of chunks (``None`` for missing).
        """
        ...

    async def get_communities(self, community_ids: typing.List[str]) -> typing.List[typing.Optional[ragu.graph.types.Community]]:
        """
        Retrieve communities by their IDs, reconstructing from stored metadata.

        :param community_ids: Community IDs to fetch.
        :return: List of communities (``None`` for missing).
        """
        ...

    async def query_entities(self, query: str, top_k: int = 20) -> typing.List[ragu.graph.types.Entity]:
        """
        Search for entities using semantic similarity.

        :param query: Search query text.
        :param top_k: Number of results to return.
        :return: Matching entities.
        """
        ...

    async def query_relations(self, query: str, top_k: int = 20) -> typing.List[ragu.graph.types.Relation]:
        """
        Search for relations using semantic similarity.

        :param query: Search query text.
        :param top_k: Number of results to return.
        :return: Matching relations.
        """
        ...