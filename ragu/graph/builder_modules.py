from typing import List, Tuple

from ragu.common.logger import logger
from ragu.graph.graph_builder_pipeline import GraphBuilderModule
from ragu.graph.types import Entity, Relation


class RemoveIsolatedNodes(GraphBuilderModule):
    def __init__(self):
        super().__init__()

    async def run(
            self,
            entities: List[Entity],
            relations: List[Relation],
            **kwargs
    ) -> Tuple[List[Entity], List[Relation]]:

        entity_ids = {e.id for e in entities if e.id}
        relations = [
            r for r in relations
            if r.subject_id in entity_ids and r.object_id in entity_ids
        ]

        connected_ids = {r.subject_id for r in relations} | {r.object_id for r in relations}
        updated_entities = [e for e in entities if e.id in connected_ids]

        if len(entities) != len(updated_entities):
            logger.info(f"Take only {len(updated_entities)}/{len(entities)} entities after removing isolated nodes")

        return updated_entities, relations