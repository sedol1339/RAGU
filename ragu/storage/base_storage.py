from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, Generic, TypeVar, List, Set, Optional, Tuple

import numpy as np

from ragu.graph.types import Entity, Relation

EdgeSpec = Tuple[str, str, Optional[str]]


@dataclass
class BaseStorage(ABC):
    @abstractmethod
    async def index_start_callback(self):
        pass

    @abstractmethod
    async def index_done_callback(self):
        pass

    @abstractmethod
    async def query_done_callback(self):
        pass


@dataclass
class BaseVectorStorage(BaseStorage, ABC):
    @abstractmethod
    async def query(self, query: str, top_k: int) -> list[dict]:
        ...

    @abstractmethod
    async def upsert(self, data: dict[str, np.ndarray]):
        ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        ...


T = TypeVar("T")

@dataclass
class BaseKVStorage(Generic[T], BaseStorage, ABC):
    @abstractmethod
    async def all_keys(self) -> List[str]:
        ...

    @abstractmethod
    async def get_by_id(self, id: str) -> Union[T, None]:
        ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str], fields: Union[set[str], None] = None) -> List[Union[T, None]]:
        ...

    @abstractmethod
    async def filter_keys(self, data: List[str]) -> Set[str]:
        ...

    @abstractmethod
    async def upsert(self, data: dict[str, T]):
        ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        ...

    @abstractmethod
    async def drop(self):
        ...


@dataclass
class BaseGraphStorage(BaseStorage, ABC):
    @abstractmethod
    async def edges_degrees(self, edge_specs: List[EdgeSpec]) -> List[int]:
        ...

    @abstractmethod
    async def index_start_callback(self):
        ...

    @abstractmethod
    async def index_done_callback(self):
        ...

    @abstractmethod
    async def get_nodes(self, node_ids: List[str]) -> List[Optional[Entity]]:
        ...

    @abstractmethod
    async def upsert_nodes(self, nodes: List[Entity]) -> None:
        ...

    @abstractmethod
    async def delete_nodes(self, node_ids: List[str]) -> None:
        ...

    @abstractmethod
    async def get_edges(self, edge_specs: List[EdgeSpec]) -> List[Optional[Relation]]:
        ...

    @abstractmethod
    async def upsert_edges(self, edges: List[Relation]) -> None:
        ...

    @abstractmethod
    async def delete_edges(self, edge_specs: List[EdgeSpec]) -> None:
        ...

    @abstractmethod
    async def get_all_edges_for_nodes(self, node_ids: List[str]) -> List[List[Relation]]:
        ...

    @abstractmethod
    async def get_all_nodes(self) -> List[Entity]:
        ...

    @abstractmethod
    async def get_all_edges(self) -> List[Relation]:
        ...
