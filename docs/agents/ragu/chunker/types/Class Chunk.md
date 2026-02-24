# Class Chunk (defined in ragu/chunker/types.py at lines 6-16)

@dataclasses.dataclass(slots=True)
class Chunk:
...

    id: str=field(init=False)

    content: str

    chunk_order_idx: int

    doc_id: str

    num_tokens: int | None = None