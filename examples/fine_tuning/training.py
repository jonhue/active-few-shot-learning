import torch
import datasets
from transformers.utils.import_utils import is_datasets_available
from torch.utils.data import DataLoader
from trl import SFTTrainer



import afsl
from afsl.acquisition_functions import AcquisitionFunction, M
from afsl.data import InputDataset



class LlamaTrainer(SFTTrainer):
    def __init__(self, acquisition_function: AcquisitionFunction, query_batch_size: int, *args, **kwargs):
        super(LlamaTrainer, self).__init__(*args, **kwargs)
        self.acquisition_function = acquisition_function
        self.query_batch_size = query_batch_size

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        train_inputs = InputDataset(train_dataset)

        return self.accelerator.prepare(
            afsl.ActiveDataLoader(
                dataset=train_dataset, 
                batch_size=self.query_batch_size,
                acquisition_function=self.acquisition_function,
                model=self.model,
                collate_fn=self.data_collator
            )
        )