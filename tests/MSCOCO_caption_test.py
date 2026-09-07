import os

import datasets as ds
import pytest


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames=(
        "dataset_year",
        "coco_task",
        "expected_num_train",
        "expected_num_validation",
    ),
    argvalues=(
        (2014, "captions", 82783, 40504),
        (2017, "captions", 118287, 5000),
    ),
)
def test_load_caption_dataset(
    dataset_path: str,
    dataset_year: int,
    coco_task: str,
    expected_num_train: int,
    expected_num_validation: int,
    repo_id: str,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        year=dataset_year,
        coco_task=coco_task,
    )
    assert isinstance(dataset, ds.DatasetDict)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_validation

    dataset.push_to_hub(
        repo_id=repo_id,
        config_name=f"year={dataset_year}_task={coco_task}",
    )
