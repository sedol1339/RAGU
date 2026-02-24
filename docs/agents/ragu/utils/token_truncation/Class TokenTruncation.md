# Class TokenTruncation (defined in ragu/utils/token_truncation.py at lines 4-93)

class TokenTruncation:
    """
    A universal text truncator class that limits input text
    to a maximum number of tokens using either `tiktoken` or HuggingFace's `AutoTokenizer`.

    :param model_id: The model name or identifier (e.g., "gpt-4o", "bert-base-uncased")
    :param tokenizer_type: Tokenizer type - either "tiktoken" or "local"
    :param max_tokens: Maximum number of tokens to retain
    :param safe_decode: Whether to use safe UTF-8 decoding for truncated output
    """
    ...