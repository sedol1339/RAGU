import asyncio
import os
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type
)

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import DEFAULT_FILENAMES
from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.graph.types import (
    Entity,
    Relation,
    Community,
    CommunitySummary
)
from ragu.storage.base_storage import (
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage
)
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage
from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage
from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage


@dataclass
class StorageArguments:
    """
    Configuration for Index storage backends.

    This dataclass consolidates all storage-related configuration in one place,
    making it easy to customize storage backends and their initialization parameters
    without understanding Index internals.

    Attributes
    ----------
    graph_backend_storage : Type[BaseGraphStorage], default=NetworkXStorage
        Storage backend class for graph structure (nodes/edges).
        Must implement BaseGraphStorage interface.
    kv_storage_type : Type[BaseKVStorage], default=JsonKVStorage
        Storage backend class for key-value data (chunks, communities, summaries).
        Must implement BaseKVStorage interface.
    vdb_storage_type : Type[BaseVectorStorage], default=NanoVectorDBStorage
        Storage backend class for vector embeddings (entities, relations, chunks).
        Must implement BaseVectorStorage interface.
    chunks_kv_storage_kwargs : Dict, default={}
        Additional kwargs passed to KV storage for text chunks.
        Common keys: 'filename' (auto-set if omitted), custom serialization options.
    summary_kv_storage_kwargs : Dict, default={}
        Additional kwargs passed to KV storage for community summaries.
    communities_kv_storage_kwargs : Dict, default={}
        Additional kwargs passed to KV storage for community metadata.
        Used to store community structure (level, cluster_id, entity/relation IDs).
    vdb_storage_kwargs : Dict, default={}
        Additional kwargs passed to vector database instances.
        Shared across entity, relation, and chunk vector stores.
    graph_storage_kwargs : Dict, default={}
        Additional kwargs passed to graph backend storage.
    max_community_size : int, default=128
        Maximum number of nodes per community during clustering.
        Controls granularity of community detection. Smaller values create smaller communities.
    random_seed : int, default=42
        Random seed for reproducible community detection.
        Ensures consistent clustering results across runs.
    """
    graph_backend_storage: Type[BaseGraphStorage] = NetworkXStorage
    kv_storage_type: Type[BaseKVStorage] = JsonKVStorage
    vdb_storage_type: Type[BaseVectorStorage] = NanoVectorDBStorage

    chunks_kv_storage_kwargs: Dict = field(default_factory=dict)
    summary_kv_storage_kwargs: Dict = field(default_factory=dict)
    communities_kv_storage_kwargs: Dict = field(default_factory=dict)
    vdb_storage_kwargs: Dict = field(default_factory=dict)
    graph_storage_kwargs: Dict = field(default_factory=dict)

    max_community_size: int = 128
    random_seed: int = 42


class Index:
    """
    Index class that manages all storage operations for a knowledge graph.

    Provides CRUD operations for entities, relations, chunks, communities,
    and community summaries with proper cascading deletes and multi-storage
    consistency.
    """

    def __init__(
            self,
            embedder: BaseEmbedder,
            arguments: StorageArguments,
    ):
        """
        Initializes the Index.
        """

        Settings.init_storage_folder()
        storage_folder: str = Settings.storage_folder

        self.embedder = embedder
        self.summary_kv_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["community_summary_kv_storage_name"],
            arguments.summary_kv_storage_kwargs,
        )
        self.community_kv_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["community_kv_storage_name"],
            arguments.communities_kv_storage_kwargs,
        )
        self.chunks_kv_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["chunks_kv_storage_name"],
            arguments.chunks_kv_storage_kwargs,
        )
        self.vdb_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["entity_vdb_name"],
            arguments.vdb_storage_kwargs,
        )
        relation_vdb_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["relation_vdb_name"],
            arguments.vdb_storage_kwargs,
        )
        chunk_vdb_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["chunk_vdb_name"],
            arguments.vdb_storage_kwargs,
        )
        self.graph_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["knowledge_graph_storage_name"],
            arguments.graph_storage_kwargs,
        )

        # Key-value storages
        self.chunks_kv_storage = arguments.kv_storage_type(**self.chunks_kv_storage_kwargs)

        self.community_summary_kv_storage = arguments.kv_storage_type(**self.summary_kv_storage_kwargs)
        self.communities_kv_storage = self.community_summary_kv_storage

        self.community_kv_storage = arguments.kv_storage_type(**self.community_kv_storage_kwargs)

        # Vector storage
        self.entity_vector_db = arguments.vdb_storage_type(embedder=embedder, **self.vdb_storage_kwargs)
        self.relation_vector_db = arguments.vdb_storage_type(embedder=embedder, **relation_vdb_storage_kwargs)
        self.chunk_vector_db = arguments.vdb_storage_type(embedder=embedder, **chunk_vdb_storage_kwargs)

        # Graph storage
        self.graph_backend = arguments.graph_backend_storage(
            max_cluster_size=arguments.max_community_size,
            random_seed=arguments.random_seed,
            **self.graph_storage_kwargs
        )

    async def make_index(
            self,
            entities: List[Entity] = None,
            relations: List[Relation] = None,
            communities: List[Community] = None,
            summaries: List[CommunitySummary] = None,
    ) -> None:
        """
        Creates an index for the given knowledge graph items.
        """
        tasks = []

        if entities:
            tasks.extend(
                [
                    self.insert_entities_to_graph(entities),
                    self.insert_entities_to_vdb(entities),
                ]
            )
        if relations:
            tasks.extend(
                [
                    self.insert_relations_to_graph(relations),
                    self.insert_relations_to_vdb(relations),
                ]
            )
        if communities:
            tasks.append(self.insert_communities(communities))
        if summaries:
            tasks.append(self.insert_summaries(summaries))

        if tasks:
            await asyncio.gather(*tasks)

    async def insert_entities_to_graph(self, entities: List[Entity]) -> None:
        if not entities:
            return
        backend = self.graph_backend
        if self.graph_backend is None:
            logger.warning("Graph storage is not initialized.")
            return
        await self.graph_bulk_upsert(backend, entities, backend.upsert_node, "entities")

    async def insert_relations_to_graph(self, relations: List[Relation]) -> None:
        if not relations:
            return
        backend = self.graph_backend
        if backend is None:
            logger.warning("Graph storage is not initialized.")
            return
        await self.graph_bulk_upsert(backend, relations, backend.upsert_edge, "relations")

    async def insert_entities_to_vdb(self, entities: List[Entity]) -> None:
        if not entities:
            return

        data_for_vdb = {
            entity.id: {
                "entity_name": entity.entity_name,
                "content": entity.entity_name + " - " + entity.description,
            }
            for entity in entities
        }
        await self._vdb_upsert(self.entity_vector_db, data_for_vdb, "entities")

    async def insert_relations_to_vdb(self, relations: List[Relation]) -> None:
        if not relations:
            return

        data_for_vdb = {
            relation.id: {
                "subject": relation.subject_id,
                "object": relation.object_id,
                "content": relation.description
            }
            for relation in relations
        }
        await self._vdb_upsert(self.relation_vector_db, data_for_vdb, "relations")

    async def insert_chunks(self, chunks: List[Chunk], vectorize: bool = False) -> None:
        """
        Stores raw chunks in a KV storage.
        """
        tasks = []

        if self.chunks_kv_storage is not None:
            data_for_kv = {}
            for chunk in chunks:
                chunk_dict = asdict(chunk)
                chunk_id = chunk_dict.pop("id")
                data_for_kv[chunk_id] = chunk_dict

            async def insert_to_kv():
                await self.chunks_kv_storage.upsert(data_for_kv)
                await self.chunks_kv_storage.index_done_callback()

            tasks.append(insert_to_kv())

        if vectorize:
            tasks.append(self.insert_chunks_to_vdb(chunks))

        if tasks:
            await asyncio.gather(*tasks)

    async def insert_chunks_to_vdb(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        data_for_vdb = {
            chunk.id: {
                "content": chunk.content,
                "doc_id": chunk.doc_id,
            }
            for chunk in chunks
        }
        await self._vdb_upsert(self.chunk_vector_db, data_for_vdb, "chunks")

    async def insert_communities(self, communities: List[Community]) -> None:
        if self.community_kv_storage is None:
            logger.warning("Community KV storage is not initialized.")
            return

        try:
            data_for_kv: Dict[str, Dict] = {}
            for c in communities:
                data_for_kv[c.id] = {
                    "level": c.level,
                    "cluster_id": c.cluster_id,
                    "entity_ids": self._unique_ids(c.entities),
                    "relation_ids": self._unique_ids(c.relations),
                }

            await self.community_kv_storage.upsert(data_for_kv)
            await self.community_kv_storage.index_done_callback()

        except Exception as e:
            logger.error(f"Failed to insert communities into KV storage: {e}")

    async def insert_summaries(self, summaries: List[CommunitySummary]) -> None:
        if self.community_summary_kv_storage is None:
            logger.warning("Community summary KV storage is not initialized.")
            return

        try:
            data_for_kv = {s.id: s.summary for s in summaries}
            await self.community_summary_kv_storage.upsert(data_for_kv)
            await self.community_summary_kv_storage.index_done_callback()

        except Exception as e:
            logger.error(f"Failed to insert community summaries into KV storage: {e}")

    async def graph_bulk_upsert(
            self,
            backend: BaseGraphStorage,
            items: Iterable[Any],
            upsert_fn: Callable[[Any], Awaitable[None]],
            artifact_label: str,
    ) -> None:
        if not items:
            return

        try:
            await backend.index_start_callback()
            await asyncio.gather(*(upsert_fn(item) for item in items))
        except Exception as e:
            logger.error(f"Failed to insert {artifact_label} into graph: {e}")
        finally:
            await backend.index_done_callback()

    async def _vdb_upsert(
            self,
            storage: Optional[BaseVectorStorage],
            payload: Dict[str, Dict],
            artifact_label: str,
    ) -> None:
        if storage is None:
            logger.warning("Vector database storage is not initialized.")
            return
        if not payload:
            return

        try:
            await storage.upsert(payload)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Failed to insert {artifact_label} into vector DB: {e}")
        finally:
            await storage.index_done_callback()  # type: ignore[func-returns-value]

    @staticmethod
    def _build_storage_kwargs(
            storage_folder: str,
            filename: str,
            provided_kwargs: Optional[Dict] = None,
    ) -> Dict:
        kwargs = dict(provided_kwargs or {})
        kwargs.setdefault(
            "filename",
            os.path.abspath(os.path.join(storage_folder, filename)),
        )
        return kwargs

    @staticmethod
    def _unique_ids(items: Iterable[Any]) -> List[str]:
        ids: List[str] = []
        seen = set()
        for item in items:
            item_id = getattr(item, "id", str(item))
            if item_id not in seen:
                ids.append(item_id)
                seen.add(item_id)
        return ids
