from __future__ import annotations

import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..config import ECGDATA_DIR, MAPPINGS_DIR, WINDOW_SAMPLES, ExperimentConfig, FileRecord, LabelMapping, WindowRecord
from ..utils.runtime import TeeLogger


@lru_cache(maxsize=4096)
def read_signal_cached(file_path: str) -> np.ndarray:
    values = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
    if values.ndim == 0:
        values = np.asarray([float(values)], dtype=np.float32)
    return values.astype(np.float32, copy=False)


def load_mapping_preset(name: str) -> LabelMapping:
    mapping_path = MAPPINGS_DIR / f"{name}.json"
    if not mapping_path.exists():
        available = sorted(p.stem for p in MAPPINGS_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"Mapping preset '{name}' not found in {MAPPINGS_DIR}. Available: {', '.join(available)}"
        )
    payload = json.loads(mapping_path.read_text())
    return LabelMapping(
        name=name,
        description=payload["description"],
        class_order=list(payload["class_order"]),
        class_map=dict(payload["class_map"]),
    )


def map_label(raw_type: str, class_map: dict[str, str | None]) -> str | None:
    return class_map.get(raw_type)


def list_candidate_files(days: Iterable[str], class_map: dict[str, str | None]) -> dict[str, list[Path]]:
    files_by_class: dict[str, list[Path]] = defaultdict(list)
    for day in days:
        day_dir = ECGDATA_DIR / day
        if not day_dir.exists():
            continue
        for type_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
            mapped = map_label(type_dir.name, class_map)
            if mapped is None:
                continue
            files_by_class[mapped].extend(sorted(type_dir.glob("*.csv")))
    return files_by_class


def build_split_file_records(
    config: ExperimentConfig,
    mapping: LabelMapping,
    logger: TeeLogger | None = None,
) -> tuple[list[FileRecord], list[WindowRecord]]:
    rng = random.Random(config.seed)
    files_by_class = list_candidate_files(config.days, mapping.class_map)
    file_records: list[FileRecord] = []
    window_records: list[WindowRecord] = []
    class_to_index = {name: idx for idx, name in enumerate(mapping.class_order)}

    for mapped_label in mapping.class_order:
        candidates = list(files_by_class.get(mapped_label, []))
        if not candidates:
            if logger:
                logger.log(f"[manifest] class={mapped_label} candidates=0 skipped")
            continue
        rng.shuffle(candidates)
        selected = candidates if config.max_files_per_class <= 0 else candidates[: config.max_files_per_class]
        n_selected = len(selected)
        if n_selected < 3:
            if logger:
                logger.log(f"[manifest] class={mapped_label} candidates={n_selected} skipped (<3)")
            continue
        val_count = max(1, round(n_selected * config.val_fraction))
        test_count = max(1, round(n_selected * config.test_fraction))
        if val_count + test_count >= n_selected:
            overflow = val_count + test_count - (n_selected - 1)
            if overflow > 0:
                if test_count >= val_count and test_count > 1:
                    test_count -= overflow
                else:
                    val_count = max(1, val_count - overflow)
        val_files = {str(p) for p in selected[:val_count]}
        test_files = {str(p) for p in selected[val_count : val_count + test_count]}
        if logger:
            logger.log(
                f"[manifest] class={mapped_label} selected={n_selected} "
                f"train={n_selected - val_count - test_count} val={val_count} test={test_count}"
            )

        for path in selected:
            signal = read_signal_cached(str(path))
            n_windows = len(signal) // WINDOW_SAMPLES
            if n_windows <= 0:
                continue
            if str(path) in val_files:
                split = "val"
            elif str(path) in test_files:
                split = "test"
            else:
                split = "train"
            raw_type = path.parent.name
            file_record = FileRecord(
                split=split,
                date=path.parent.parent.name,
                raw_type=raw_type,
                mapped_label=mapped_label,
                file_path=str(path),
                file_name=path.name,
                n_samples=int(len(signal)),
                n_windows=int(n_windows),
            )
            file_records.append(file_record)
            class_index = class_to_index[mapped_label]
            for window_index in range(n_windows):
                start = window_index * WINDOW_SAMPLES
                end = start + WINDOW_SAMPLES
                window_records.append(
                    WindowRecord(
                        split=split,
                        date=path.parent.parent.name,
                        raw_type=raw_type,
                        mapped_label=mapped_label,
                        class_index=class_index,
                        file_path=str(path),
                        file_name=path.name,
                        window_index=window_index,
                        start_sample=start,
                        end_sample=end,
                    )
                )
    if logger:
        logger.log(f"[manifest] total_files={len(file_records)} total_windows={len(window_records)}")
    return file_records, window_records


class WindowDataset(Dataset):
    def __init__(self, records: list[WindowRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[index]
        signal = read_signal_cached(rec.file_path)
        window = signal[rec.start_sample : rec.end_sample]
        return (
            torch.from_numpy(window.copy()).float(),
            torch.tensor(rec.class_index, dtype=torch.long),
        )


class FileWindowBatchSampler(Sampler[list[int]]):
    """Shuffle by file first so adjacent samples reuse the cached CSV signal."""

    def __init__(
        self,
        records: list[WindowRecord],
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        groups: dict[str, list[int]] = defaultdict(list)
        for index, rec in enumerate(records):
            groups[rec.file_path].append(index)
        self.groups = list(groups.values())
        self.num_records = len(records)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_records // self.batch_size
        return (self.num_records + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = [list(group) for group in self.groups]
        rng.shuffle(groups)
        batch: list[int] = []
        for group in groups:
            rng.shuffle(group)
            for index in group:
                batch.append(index)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch and not self.drop_last:
            yield batch
