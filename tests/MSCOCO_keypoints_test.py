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
    argnames="decode_rle,",
    argvalues=(
        True,
        False,
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
        (2014, "person-keypoints", 45174, 21634),
        (2017, "person-keypoints", 64115, 2693),
    ),
)
def test_load_keypoints_dataset(
    dataset_path: str,
    dataset_year: int,
    coco_task: str,
    decode_rle: bool,
    expected_num_train: int,
    expected_num_validation: int,
    repo_id: str,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        year=dataset_year,
        coco_task=coco_task,
        decode_rle=decode_rle,
    )
    assert isinstance(dataset, ds.DatasetDict)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_validation

    dataset.push_to_hub(
        repo_id=repo_id,
        config_name=f"year={dataset_year}_task={coco_task}_decode-rle={decode_rle}",
    )
