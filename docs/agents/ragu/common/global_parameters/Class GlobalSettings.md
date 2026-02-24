# Class GlobalSettings (defined in ragu/common/global_parameters.py at lines 20-46)

class GlobalSettings:
...

    language: str = "english"

    @property
    def storage_folder(self):
    ...

    @storage_folder.setter
    def storage_folder(self, path):
    ...

    def init_storage_folder(self):
    ...