# Class NetworkXStorage (defined in ragu/storage/graph_storage_adapters/networkx_adapter.py at lines 29-293)

class NetworkXStorage(ragu.storage.base_storage.BaseGraphStorage):
    """
    NetworkX-based implementation of BaseGraphStorage.
    """
    ...

    async def index_done_callback(self) -> None:
        """
        Persist the current graph state to disk in GML format.
        """
        ...

    async def query_done_callback(self) -> None:
        """
        Callback executed after a query is completed.
        Reserved for potential post-processing hooks.
        """
        ...

    async def index_start_callback(self) -> None:
        """
        Callback executed before indexing starts.
        Reserved for potential setup hooks.
        """
        ...

    async def get_node_edges(self, source_node_id: str) -> typing.List[ragu.graph.types.Relation]:
        """
        Retrieve all edges connected to a given node.

        Each returned :class:`Relation` includes associated metadata
        and node display names when available. Missing nodes are tolerated.

        :param source_node_id: ID of the node whose edges to fetch.
        :return: List of relations connected to the node.
        """
        ...

    async def edges_degrees(self, edge_specs: typing.List[ragu.storage.base_storage.EdgeSpec]) -> typing.List[int]:
        """
        Retrieve degree values for multiple edges.

        For each edge spec, returns ``degree(subject_id) + degree(object_id)``.
        Returns ``0`` when the relation or either endpoint is missing.

        :param edge_specs: Edge specifications to evaluate.
        :return: Degree sums aligned with ``edge_specs``.
        """
        ...

    async def upsert_nodes(self, nodes: typing.Iterable[ragu.graph.types.Entity]) -> None:
        """
        Insert or update multiple nodes in the graph.

        :param nodes: Iterable of entities to process.
        """
        ...

    async def get_nodes(self, node_ids: typing.List[str]) -> typing.List[typing.Optional[ragu.graph.types.Entity]]:
        """
        Retrieve multiple nodes by their IDs.

        :param node_ids: List of node identifiers to fetch.
        :return: List of entities (``None`` for missing nodes).
        """
        ...

    async def delete_nodes(self, node_ids: typing.List[str]) -> None:
        """
        Delete multiple nodes from the graph.

        Cascade removes all connected edges.

        :param node_ids: List of node identifiers to remove.
        """
        ...

    async def get_edges(self, edge_specs: typing.List[ragu.storage.base_storage.EdgeSpec]) -> typing.List[typing.Optional[ragu.graph.types.Relation]]:
        """
        Retrieve multiple edges by specs.

        :param edge_specs: List of edge specs ``(subject_id, object_id, relation_id)``.
        :return: List of relations (``None`` for missing edges).
        """
        ...

    async def upsert_edges(self, edges: typing.List[ragu.graph.types.Relation]) -> None:
        """
        Insert or update multiple edges in the graph.

        :param edges: List of relations to upsert.
        """
        ...

    async def delete_edges(self, edge_specs: typing.List[ragu.storage.base_storage.EdgeSpec]) -> None:
        """
        Delete multiple edges from the graph.

        :param edge_specs: List of edge specs (subject_id, object_id, relation_id).
        """
        ...

    async def get_all_edges_for_nodes(self, node_ids: typing.List[str]) -> typing.List[typing.List[ragu.graph.types.Relation]]:
        """
        Retrieve edges for each given node.

        Returns one relation list per input node ID. No cross-node deduplication
        is performed.

        :param node_ids: List of node identifiers.
        :return: Grouped relations for each node.
        """
        ...

    async def get_all_nodes(self) -> typing.List[ragu.graph.types.Entity]:
        """
        Retrieve all nodes in the graph.

        :return: List of all entities.
        """
        ...

    async def get_all_edges(self) -> typing.List[ragu.graph.types.Relation]:
        """
        Retrieve all edges in the graph.

        :return: List of all relations.
        """
        ...