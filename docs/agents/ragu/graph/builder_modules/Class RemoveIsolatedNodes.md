# Class RemoveIsolatedNodes (defined in ragu/graph/builder_modules.py at lines 8-46)

class RemoveIsolatedNodes(ragu.graph.graph_builder_pipeline.GraphBuilderModule):
    """
    Graph-builder module that removes isolated entities.

    Keeps only relations whose endpoints exist in the entity set, then removes
    entities not connected by any remaining relation.
    """
    ...

    async def run(
            self,
            entities: typing.List[ragu.graph.types.Entity],
            relations: typing.List[ragu.graph.types.Relation],
            **kwargs
    ) -> typing.Tuple[typing.List[ragu.graph.types.Entity], typing.List[ragu.graph.types.Relation]]:
        """
        Remove dangling relations and entities without edges.

        :param entities: Candidate entities.
        :param relations: Candidate relations.
        :return: Filtered entities and relations.
        """
        ...