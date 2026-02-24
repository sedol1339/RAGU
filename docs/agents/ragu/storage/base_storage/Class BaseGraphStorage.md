# Class BaseGraphStorage (defined in ragu/storage/base_storage.py at lines 149-249)

@dataclasses.dataclass
class BaseGraphStorage(ragu.storage.base_storage.BaseStorage, abc.ABC):
    """
    Abstract interface for multigraph storage backends.
    """
    ...

    @abc.abstractmethod
    async def edges_degrees(self, edge_specs: typing.List[ragu.storage.base_storage.EdgeSpec]) -> typing.List[int]:
        """
        Compute degree sums of node's degree for provided edge specifications.

        :param edge_specs: Tuples ``(subject_id, object_id, relation_id)``.
        :return: Degree values aligned with the input order.
        """
        ...

    @abc.abstractmethod
    async def get_nodes(self, node_ids: typing.List[str]) -> typing.List[typing.Optional[ragu.graph.types.Entity]]:
        """
        Fetch nodes by IDs.

        :param node_ids: Node IDs to retrieve.
        :return: Entities aligned with input IDs; missing IDs mapped to ``None``.
        """
        ...

    @abc.abstractmethod
    async def upsert_nodes(self, nodes: typing.List[ragu.graph.types.Entity]) -> None:
        """
        Insert or update nodes.

        :param nodes: Entity nodes to upsert.
        """
        ...

    @abc.abstractmethod
    async def delete_nodes(self, node_ids: typing.List[str]) -> None:
        """
        Delete nodes by IDs.

        :param node_ids: Node IDs to remove.
        """
        ...

    @abc.abstractmethod
    async def get_edges(self, edge_specs: typing.List[ragu.storage.base_storage.EdgeSpec]) -> typing.List[typing.Optional[ragu.graph.types.Relation]]:
        """
        Fetch edges by specifications.

        :param edge_specs: Tuples ``(subject_id, object_id, relation_id)``.
        :return: Relations aligned with input specs; missing specs mapped to ``None``.
        """
        ...

    @abc.abstractmethod
    async def upsert_edges(self, edges: typing.List[ragu.graph.types.Relation]) -> None:
        """
        Insert or update edges.

        :param edges: Relations to upsert.
        """
        ...

    @abc.abstractmethod
    async def delete_edges(self, edge_specs: typing.List[ragu.storage.base_storage.EdgeSpec]) -> None:
        """
        Delete edges by specifications.

        :param edge_specs: Tuples ``(subject_id, object_id, relation_id)`` to delete.
        """
        ...

    @abc.abstractmethod
    async def get_all_edges_for_nodes(self, node_ids: typing.List[str]) -> typing.List[typing.List[ragu.graph.types.Relation]]:
        """
        Fetch all incident edges for each provided node.

        :param node_ids: Node IDs to inspect.
        :return: Edge lists aligned with input node IDs.
        """
        ...

    @abc.abstractmethod
    async def get_all_nodes(self) -> typing.List[ragu.graph.types.Entity]:
        """
        Fetch all nodes stored in the backend.

        :return: List of entities.
        """
        ...

    @abc.abstractmethod
    async def get_all_edges(self) -> typing.List[ragu.graph.types.Relation]:
        """
        Fetch all edges stored in the backend.

        :return: List of relations.
        """
        ...