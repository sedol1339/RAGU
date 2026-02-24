# Class GraphBuilderModule (defined in ragu/graph/graph_builder_pipeline.py at lines 54-84)

class GraphBuilderModule:
    """
    Abstract interface for modules that extend the graph-building pipeline.

    Each module receives entities and relations
    and can modify, enrich, or filter them before insertion into the graph.

    Typically used for:
      - normalization of entity names
      - filtering noisy relations
      - post-processing after extraction

    Subclasses should override `run` to apply module-specific logic.
    """
    ...

    async def run(
            self,
            entities: typing.List[ragu.graph.types.Entity],
            relations: typing.List[ragu.graph.types.Relation],
            **kwargs
    ) -> typing.Tuple[typing.List[ragu.graph.types.Entity], typing.List[ragu.graph.types.Relation]]:
        """
        Process or update multiple nodes and edges during graph construction.

        :param entities: list of :class:`Entity` objects to insert or modify.
        :param relations: list of :class:`Relation` objects to insert or modify.
        :param kwargs: optional additional parameters specific to the module.
        :return: updated or enriched entities/relations.
        """
        ...