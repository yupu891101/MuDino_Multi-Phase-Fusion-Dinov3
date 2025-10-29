import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from dynodino.utils.data_reader import msd_base_data_reader
from dynodino.utils.data_utils import extract_data_file, folder_validate
from dynodino.utils.utils import file_downloader

type TaskNames = Literal[
    "Task01_BrainTumour",
    "Task02_Heart",
    "Task03_Liver",
    "Task04_Hippocampus",
    "Task05_Prostate",
    "Task06_Lung",
    "Task07_Pancreas",
    "Task08_HepaticVessel",
    "Task09_Spleen",
    "Task10_Colon",
]

TASK_DATA_URLS: dict[TaskNames, str] = {
    "Task01_BrainTumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
    "Task02_Heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
    "Task03_Liver": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
    "Task04_Hippocampus": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
    "Task05_Prostate": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
    "Task06_Lung": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
    "Task07_Pancreas": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
    "Task08_HepaticVessel": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
    "Task09_Spleen": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
    "Task10_Colon": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
}


class MSDBaseDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        task_name: TaskNames,
        num_workers: int = 4,
        modality_num: int | None = None,
        input_transform: Callable[..., torch.Tensor] | None = None,
        label_transform: Callable[..., torch.Tensor] | None = None,
    ):
        self.data_root = Path(data_root)
        folder_validate(self.data_root)

        if self.data_root.joinpath(task_name).exists():
            # valid task folder exists
            self.data_root = self.data_root.joinpath(task_name)

        elif self.data_root.joinpath(f"{task_name}.tar").exists():
            # valid tar file exists, extract it
            extracted_folder = extract_data_file(self.data_root.joinpath(f"{task_name}.tar").as_posix())
            if len(extracted_folder) > 1:
                raise ValueError(
                    f"Expected only one folder to be extracted for task {task_name}, "
                    f"but found multiple: {extracted_folder}"
                )
            self.data_root = self.data_root.joinpath(extracted_folder[0])
        else:
            # download and extract the dataset
            task_url = TASK_DATA_URLS.get(task_name)
            if not task_url:
                raise ValueError(f"Invalid task name: {task_name}")

            tar_save_path = self.data_root.joinpath(f"{task_name}.tar").as_posix()
            file_downloader(task_url, tar_save_path, log_error=False)
            extracted_folder = extract_data_file(tar_save_path)
            if len(extracted_folder) > 1:
                raise ValueError(
                    f"Expected only one folder to be extracted for task {task_name}, "
                    f"but found multiple: {extracted_folder}"
                )
            self.data_root = self.data_root.joinpath(extracted_folder[0])

        # prepare npy data if not exists
        if not self.data_root.joinpath("npy_imagesTr").exists() or not self.data_root.joinpath("npy_labelsTr").exists():
            msd_base_data_reader(self.data_root.as_posix(), num_workers=num_workers)

        self.modality_num = self.get_modality_num(modality_num)
        self.file_list = self.gen_data_list()
        self.input_transform = input_transform
        self.label_transform = label_transform

    def get_modality_num(self, input_modality_num: int | None) -> int:
        dataset_json_path = self.data_root.joinpath("dataset.json")
        if not dataset_json_path.exists():
            if input_modality_num is not None:
                return input_modality_num
            else:
                raise ValueError("modality_num is not set and dataset.json is missing.")

        with open(dataset_json_path, encoding="utf-8") as f:
            dataset_info = json.load(f)

        modalities = dataset_info.get("modality")
        if modalities is None:
            if input_modality_num is not None:
                return input_modality_num
            else:
                raise ValueError("modality information is missing in dataset.json and modality_num is not set.")

        return len(modalities)

    def gen_data_list(self) -> list[dict[str, list[str] | str]]:
        data_list = []
        npy_images_tr_path = self.data_root.joinpath("npy_imagesTr")
        npy_labels_tr_path = self.data_root.joinpath("npy_labelsTr")
        if self.modality_num == 1:
            file_pairs = zip(
                sorted(npy_images_tr_path.glob("*.npy")), sorted(npy_labels_tr_path.glob("*.npy")), strict=False
            )
            entry_generator = mono_modal_generator
        else:
            file_pairs = zip(
                sorted(npy_images_tr_path.glob("*_00_*.npy")),
                sorted(npy_labels_tr_path.glob("*_00_*.npy")),
                strict=False,
            )
            entry_generator = multi_modal_generator

        for image_file, label_file in file_pairs:
            data_entry = entry_generator(image_file, label_file, self.modality_num)
            data_list.append(data_entry)

        return data_list

    def __len__(self):
        return len(self.file_list)

    def load_image(self, image_path: str | list[str]) -> torch.Tensor:
        if isinstance(image_path, str):
            image_tensor = torch.from_numpy(np.load(image_path)).unsqueeze(0)
        else:
            image_slices = [torch.from_numpy(np.load(modality_path)) for modality_path in image_path]
            image_tensor = torch.stack(image_slices, dim=0)

        if self.input_transform:
            image_tensor = self.input_transform(image_tensor)

        return image_tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_entry = self.file_list[idx]
        image_tensor = self.load_image(file_entry["image"]).float()
        label_tensor = self.load_image(file_entry["label"]).long()
        if self.input_transform:
            image_tensor = self.input_transform(image_tensor)
        if self.label_transform:
            label_tensor = self.label_transform(label_tensor)
        return image_tensor, label_tensor


def mono_modal_generator(image_path: Path, label_path: Path, modality_num: int) -> dict[str, str]:
    case_id = image_path.stem.split("_")[0]
    return {
        "image": image_path.as_posix(),
        "label": label_path.as_posix(),
        "case_id": case_id,
    }


def multi_modal_generator(image_path: Path, label_path: Path, modality_num: int) -> dict[str, list[str] | str]:
    case_id = image_path.stem.split("_")[0]
    return {
        "image": [
            image_path.as_posix().replace("_00_", f"_{modality_idx:02d}_") for modality_idx in range(modality_num)
        ],
        "label": [
            label_path.as_posix().replace("_00_", f"_{modality_idx:02d}_") for modality_idx in range(modality_num)
        ],
        "case_id": case_id,
    }
