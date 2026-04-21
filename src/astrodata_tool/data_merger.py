from __future__ import annotations

import pandas as pd

from .exceptions import DataMergeError
from .models import DatasetArtifact, MergeConfiguration, OperationRecord


class DataMerger:
    SUPPORTED_JOIN_TYPES = {"inner", "left", "right", "outer"}

    def analyse_merge_risk(
        self,
        left_dataset: DatasetArtifact,
        right_dataset: DatasetArtifact,
        *,
        left_keys: list[str],
        right_keys: list[str],
    ) -> list[str]:
        self._validate_keys(left_dataset, right_dataset, left_keys, right_keys)
        warnings: list[str] = []
        if left_dataset.dataframe[left_keys].duplicated().any():
            warnings.append("Left dataset contains duplicate join keys; the merge may expand row counts.")
        if right_dataset.dataframe[right_keys].duplicated().any():
            warnings.append("Right dataset contains duplicate join keys; the merge may expand row counts.")

        left_non_matches = self._estimate_unmatched(left_dataset.dataframe, right_dataset.dataframe, left_keys, right_keys)
        right_non_matches = self._estimate_unmatched(right_dataset.dataframe, left_dataset.dataframe, right_keys, left_keys)
        if left_non_matches:
            warnings.append(f"{left_non_matches} left-side rows have no matching join key in the right dataset.")
        if right_non_matches:
            warnings.append(f"{right_non_matches} right-side rows have no matching join key in the left dataset.")
        return warnings

    def merge(
        self,
        left_dataset: DatasetArtifact,
        right_dataset: DatasetArtifact,
        configuration: MergeConfiguration,
    ) -> tuple[DatasetArtifact, list[str]]:
        if configuration.join_type not in self.SUPPORTED_JOIN_TYPES:
            raise DataMergeError(f"Unsupported join type: {configuration.join_type}")

        warnings = self.analyse_merge_risk(
            left_dataset,
            right_dataset,
            left_keys=configuration.left_keys,
            right_keys=configuration.right_keys,
        )
        left_frame = left_dataset.dataframe.copy(deep=True)
        right_frame = right_dataset.dataframe.copy(deep=True)

        if configuration.align_key_types:
            for left_key, right_key in zip(configuration.left_keys, configuration.right_keys):
                if str(left_frame[left_key].dtype) != str(right_frame[right_key].dtype):
                    left_frame[left_key] = left_frame[left_key].astype("string")
                    right_frame[right_key] = right_frame[right_key].astype("string")
                    warnings.append(f"Aligned key types by converting '{left_key}' and '{right_key}' to string.")

        merged = left_frame.merge(
            right_frame,
            how=configuration.join_type,
            left_on=configuration.left_keys,
            right_on=configuration.right_keys,
            suffixes=configuration.suffixes,
            copy=True,
            indicator=True,
        )
        merge_summary = merged["_merge"].value_counts().to_dict()
        merged = merged.drop(columns=["_merge"]).copy(deep=True)
        result = left_dataset.clone(
            dataframe=merged,
            name=f"{left_dataset.name}_{configuration.join_type}_merged_{right_dataset.name}",
        )
        result.metadata["merge_partner"] = right_dataset.name
        result.operation_history.append(
            OperationRecord(
                operation_name="merge_datasets",
                parameters={
                    "left_keys": configuration.left_keys,
                    "right_keys": configuration.right_keys,
                    "join_type": configuration.join_type,
                    "suffixes": configuration.suffixes,
                },
                summary=f"Merged datasets with result shape {merged.shape} and merge indicator counts {merge_summary}.",
                dataset_before=left_dataset.name,
                dataset_after=result.name,
            )
        )
        return result, warnings

    @staticmethod
    def _validate_keys(
        left_dataset: DatasetArtifact,
        right_dataset: DatasetArtifact,
        left_keys: list[str],
        right_keys: list[str],
    ) -> None:
        if len(left_keys) != len(right_keys) or not left_keys:
            raise DataMergeError("Merge requires matching non-empty key lists.")
        missing_left = sorted(set(left_keys) - set(left_dataset.dataframe.columns))
        missing_right = sorted(set(right_keys) - set(right_dataset.dataframe.columns))
        if missing_left:
            raise DataMergeError(f"Left merge keys are missing: {', '.join(missing_left)}")
        if missing_right:
            raise DataMergeError(f"Right merge keys are missing: {', '.join(missing_right)}")

    @staticmethod
    def _estimate_unmatched(left_frame: pd.DataFrame, right_frame: pd.DataFrame, left_keys: list[str], right_keys: list[str]) -> int:
        left_keys_frame = left_frame[left_keys].drop_duplicates().reset_index(drop=True)
        right_keys_frame = right_frame[right_keys].drop_duplicates().reset_index(drop=True)
        merged_keys = left_keys_frame.merge(
            right_keys_frame,
            how="left",
            left_on=left_keys,
            right_on=right_keys,
            indicator=True,
        )
        return int((merged_keys["_merge"] == "left_only").sum())
