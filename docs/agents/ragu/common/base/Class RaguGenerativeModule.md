# Class RaguGenerativeModule (defined in ragu/common/base.py at lines 21-82)

class RaguGenerativeModule:
    """
    Base class for generative components in the RAGU framework.

    Provides consistent handling of prompt templates across modules.
    The class can load default templates by name or accept custom
    :class:`PromptTemplate` instances directly.
    """
    ...

    def get_prompts(self) -> dict:
        """
        Retrieve all prompt templates registered in the module.

        :return: Dictionary mapping prompt names to :class:`ChatTemplate` objects.
        """
        ...

    def get_prompt(self, prompt_name: str) -> ragu.common.prompts.prompt_storage.RAGUInstruction:
        """
        Retrieve a specific prompt template by name.

        :param prompt_name: The name of the prompt to retrieve.
        :return: The corresponding :class:`ChatTemplate` instance.
        :raises ValueError: If the prompt name is not found.
        """
        ...

    def update_prompt(self, prompt_name: str, prompt: ragu.common.prompts.prompt_storage.RAGUInstruction) -> None:
        """
        Replace or add a prompt template in the module.

        :param prompt_name: The key name under which to store the prompt.
        :param prompt: The :class:`PromptTemplate` object to register.
        """
        ...