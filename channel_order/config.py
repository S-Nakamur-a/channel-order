from pathlib import Path

from serde import serialize, deserialize
from dataclasses import dataclass


@deserialize
@serialize
@dataclass
class Dataset:
    train: Path
    val: Path
    test: Path


@deserialize
@serialize
@dataclass
class Config:
    save_dir: Path
    experiment_name: str
    version: int
    model: str
    dataset: Dataset
    epochs: int
