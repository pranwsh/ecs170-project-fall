"""Save Stage 2 prediction results (same idea as the Stage 1 Result_Saver)."""

import pickle
from pathlib import Path

from local_code.base_class.result import result


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...') # Remember to check file permissions if this fails
        folder = Path(self.result_destination_folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        filepath = folder / self.result_destination_file_name
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self):
        filepath = Path(self.result_destination_folder_path) / self.result_destination_file_name
        with open(filepath, 'rb') as f:
            self.data = pickle.load(f)
        return self.data
