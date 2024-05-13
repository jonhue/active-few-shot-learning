from dataclasses import dataclass
from typing import Dict, Optional
import torch
import datasets
import numpy as np
from transformers import AutoTokenizer, TrainingArguments
from transformers.utils.import_utils import is_datasets_available
from accelerate.state import PartialState
from torch.utils.data import DataLoader
from trl import SFTTrainer
from trl.extras.dataset_formatting import get_formatting_func_from_dataset


import afsl
from afsl.acquisition_functions import AcquisitionFunction, M
from afsl.data import InputDataset, LlamaDataset
from examples.acquisition_functions import get_acquisition_function


@dataclass
class ITLConfig:
    alg: str
    noise_std: float
    mini_batch_size: int
    num_workers: int
    subsample_acquisition: bool
    subsampled_target_frac: float
    max_target_size: int | None


@dataclass
class SFTConfig(TrainingArguments):
    dataset_text_field: Optional[str] = None
    packing: Optional[bool] = False
    max_seq_length: Optional[int] = None
    dataset_num_proc: Optional[int] = None
    dataset_batch_size: int = 1000
    neftune_noise_alpha: Optional[float] = None
    model_init_kwargs: Optional[Dict] = None
    dataset_kwargs: Optional[Dict] = None
    eval_packing: Optional[bool] = None
    num_of_sequences: Optional[int] = 1024
    chars_per_token: Optional[float] = 3.6


class LlamaTrainer(SFTTrainer):
    def __init__(self, itl_config: ITLConfig, query_batch_size: int, *args, **kwargs):
        super(LlamaTrainer, self).__init__(*args, **kwargs)
        self.itl_config = itl_config
        self.query_batch_size = query_batch_size

    #
    #   Introduce the ActiveDataLoader
    #

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
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        acquisition_function = get_acquisition_function(
            alg=self.itl_config.alg,
            target=self.get_test_data(),  # TODO give test_set
            noise_std=self.itl_config.noise_std,
            mini_batch_size=self.itl_config.mini_batch_size,
            num_workers=self.itl_config.num_workers,
            subsample_acquisition=self.itl_config.subsample_acquisition,
            subsampled_target_frac=self.itl_config.subsampled_target_frac,
            max_target_size=self.itl_config.max_target_size,
        )

        train_dataset = LlamaDataset(train_dataset)

        active_sampler = afsl.ActiveDataLoader(
            dataset=train_dataset,
            batch_size=self.query_batch_size,
            acquisition_function=acquisition_function,
            model=self.model,
        )

        return self.accelerator.prepare(
            DataLoader(
                dataset=train_dataset,
                batch_size=self.query_batch_size,
                sampler=active_sampler,
                num_workers=self.itl_config.num_workers,
            )   
        )
    
    #
    #   Create tensor from test data for the target space
    #

    def get_test_data(self) -> torch.Tensor:
        if self.eval_dataset is None:
            return torch.Tensor()

        main_input_name = getattr(self.model, "main_input_name", "input_ids")

        input_ids_list = []
        for target in self.eval_dataset:
            input_ids = torch.tensor(target[main_input_name])
            input_ids_list.append(input_ids)

        return torch.stack(input_ids_list)

    #
    #   Pad to max_length when tokenizing
    #

    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset
