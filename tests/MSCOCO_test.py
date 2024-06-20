import os

import datasets as ds
import pytest

from const import CATEGORIES, SUPER_CATEGORIES


@pytest.fixture
def dataset_path() -> str:
    return "./MSCOCO.py"


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
        (2014, "captions", 82783, 40504),
        (2017, "captions", 118287, 5000),
        (2014, "instances", 82081, 40137),
        (2017, "instances", 117266, 4952),
        (2014, "person_keypoints", 45174, 21634),
        (2017, "person_keypoints", 64115, 2693),
    ),
)
def test_load_dataset(
    dataset_path: str,
    dataset_year: int,
    coco_task: str,
    decode_rle: bool,
    expected_num_train: int,
    expected_num_validation: int,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        year=dataset_year,
        coco_task=coco_task,
        decode_rle=decode_rle,
        trust_remote_code=True,
    )
    assert isinstance(dataset, ds.DatasetDict)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_validation


def test_consts():
    assert len(CATEGORIES) == 80
    assert len(SUPER_CATEGORIES) == 12
