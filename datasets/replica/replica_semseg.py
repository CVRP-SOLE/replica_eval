import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
from random import randrange

import datasets.replica_util as replica_util
import datasets.replica_util_3d_original as util_3d

import numpy
import torch
from datasets.random_cuboid import RandomCuboid

import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml

# from yaml import CLoader as Loader
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)

# ---------- Label info ---------- #
DATASET_NAME = "replica"
CLASS_LABELS = ["basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle", "box", "bowl", "camera", "cabinet", "candle", "chair", "clock",
"cloth", "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant", "lamp", "monitor", "nightstand",
"panel", "picture", "pillar", "pillow", "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa", "stool", "switch", "table",
"tablet", "tissue-paper", "tv-screen", "tv-stand", "vase", "vent", "wall-plug", "window", "rug"]

VALID_CLASS_IDS = np.asarray([3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 34, 35, 37, 44, 47, 52, 54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 79, 80, 82, 83, 87, 88, 91, 92, 95, 97, 98])
ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
SKIPPED_CLASSES = ["undefined", "floor", "ceiling", "wall"]

class ReplicaSemanticSegmentationDataset(Dataset):
    """Docstring for SemanticSegmentationDataset."""

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling=0,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        add_unlabeled_pc=False,
        task="instance_segmentation",
        cropping=False,
        cropping_args=None,
        is_tta=False,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        area=-1,
        on_crops=False,
        eval_inner_core=-1,
        filter_out_classes=[],
        label_offset=0,
        add_clip=False,
        is_elastic_distortion=True,
        color_drop=0.0
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "unknown task"

        color_mean_std = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        )

        self.dataset_name = dataset_name
        self.label_offset = label_offset

        self.reps_per_epoch = reps_per_epoch
        self.on_crops = on_crops

        self.version1 = cropping_v1

        self.mode = mode
        self.data_dir = data_dir
        self.add_unlabeled_pc = add_unlabeled_pc
        if add_unlabeled_pc:
            self.other_database = self._load_yaml(
                Path(data_dir).parent / "matterport" / "train_database.yaml"
            )
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.instance_oversampling = instance_oversampling
        self.place_around_existing = place_around_existing
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if not (database_path / f"{mode}_database.yaml").exists():
                print(
                    f"generate {database_path}/{mode}_database.yaml first"
                )
                exit()
            self._data.extend(
                self._load_yaml(database_path / f"{mode}_database.yaml")
            )

        if data_percent < 1.0:
            self._data = sample(
                self._data, int(len(self._data) * data_percent)
            )

        labels = self._load_yaml(Path(label_db_filepath))
        self.color_map = {}
        for k, v in labels.items():
            self.color_map[k] = v['color']
        # if working only on classes for validation - discard others
        self._labels = self._select_correct_labels(labels, num_labels)

        if instance_oversampling > 0:
            self.instance_data = self._load_yaml(
                Path(label_db_filepath).parent / "instance_database.yaml"
            )

        color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
        color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)

    def __len__(self):
        return self.reps_per_epoch * len(self.data)

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)

        points = torch.load(self.data[idx]["filepath"])
        gt_ids= util_3d.load_ids(self.data[idx]["instance_gt_filepath"])
        gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)

        labels_semantic = np.ones((gt_ids.shape[0], 1), dtype=np.int32) * 255
        labels_instance = gt_ids.copy()[:, np.newaxis]
        labels = np.concatenate([labels_semantic, labels_instance], axis=-1)

        for class_info_list in gt_instances.values():
            if len(class_info_list) == 0:
                continue

            for info in class_info_list:
                label_idx = gt_ids == info['instance_id']
                label_id = info['label_id']
                labels[label_idx, 0] = label_id

        coordinates, color = points[0], points[1]

        raw_coordinates = coordinates.copy()

        if not self.add_colors:
            color = np.ones((len(color), 3))

        # normalize color information
        color = np.clip(((color + 1) / 2) * 255, 0, 255)
        raw_color = color.copy()
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        # prepare labels and map from 0 to 20(40)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            if not self.add_instance:
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()[..., None]

        features = color
        
        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))
        
        return (
            coordinates,
            features,
            labels,
            self.data[idx]["filepath"].split("/")[-1],
            raw_color,
            raw_coordinates,
            idx,
        )

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f)
        return file

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
            k,
            v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                k,
                v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _remap_from_zero(self, labels):
        labels[
            ~np.isin(labels, list(self.label_info.keys()))
        ] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def flip_in_center(coordinates):
    # moving coordinates to center
    coordinates -= coordinates.mean(0)
    aug = V.Compose(
        [
            V.Flip3d(axis=(0, 1, 0), always_apply=True),
            V.Flip3d(axis=(1, 0, 0), always_apply=True),
        ]
    )

    first_crop = coordinates[:, 0] > 0
    first_crop &= coordinates[:, 1] > 0
    # x -y
    second_crop = coordinates[:, 0] > 0
    second_crop &= coordinates[:, 1] < 0
    # -x y
    third_crop = coordinates[:, 0] < 0
    third_crop &= coordinates[:, 1] > 0
    # -x -y
    fourth_crop = coordinates[:, 0] < 0
    fourth_crop &= coordinates[:, 1] < 0

    if first_crop.size > 1:
        coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        minimum = coordinates[second_crop].min(0)
        minimum[2] = 0
        minimum[0] = 0
        coordinates[second_crop] = aug(points=coordinates[second_crop])[
            "points"
        ]
        coordinates[second_crop] += minimum
    if third_crop.size > 1:
        minimum = coordinates[third_crop].min(0)
        minimum[2] = 0
        minimum[1] = 0
        coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
        coordinates[third_crop] += minimum
    if fourth_crop.size > 1:
        minimum = coordinates[fourth_crop].min(0)
        minimum[2] = 0
        coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])[
            "points"
        ]
        coordinates[fourth_crop] += minimum

    return coordinates


def random_around_points(
    coordinates,
    color,
    normals,
    labels,
    rate=0.2,
    noise_rate=0,
    ignore_label=255,
):
    coord_indexes = sample(
        list(range(len(coordinates))), k=int(len(coordinates) * rate)
    )
    noisy_coordinates = deepcopy(coordinates[coord_indexes])
    noisy_coordinates += np.random.uniform(
        -0.2 - noise_rate, 0.2 + noise_rate, size=noisy_coordinates.shape
    )

    if noise_rate > 0:
        noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
        noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
        noisy_labels = np.full(labels[coord_indexes].shape, ignore_label)

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))
    else:
        noisy_color = deepcopy(color[coord_indexes])
        noisy_normals = deepcopy(normals[coord_indexes])
        noisy_labels = deepcopy(labels[coord_indexes])

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))

    return coordinates, color, normals, labels


def random_points(
    coordinates, color, normals, labels, noise_rate=0.6, ignore_label=255
):
    max_boundary = coordinates.max(0) + 0.1
    min_boundary = coordinates.min(0) - 0.1

    noisy_coordinates = int(
        (max(max_boundary) - min(min_boundary)) / noise_rate
    )

    noisy_coordinates = np.array(
        list(
            product(
                np.linspace(
                    min_boundary[0], max_boundary[0], noisy_coordinates
                ),
                np.linspace(
                    min_boundary[1], max_boundary[1], noisy_coordinates
                ),
                np.linspace(
                    min_boundary[2], max_boundary[2], noisy_coordinates
                ),
            )
        )
    )
    noisy_coordinates += np.random.uniform(
        -noise_rate, noise_rate, size=noisy_coordinates.shape
    )

    noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
    noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
    noisy_labels = np.full(
        (noisy_coordinates.shape[0], labels.shape[1]), ignore_label
    )

    coordinates = np.vstack((coordinates, noisy_coordinates))
    color = np.vstack((color, noisy_color))
    normals = np.vstack((normals, noisy_normals))
    labels = np.vstack((labels, noisy_labels))
    return coordinates, color, normals, labels
