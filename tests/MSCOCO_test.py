import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "MSCOCO.py"


@pytest.mark.parametrize(
    argnames="dataset_year",
    argvalues=(
        2014,
        # 2017,
    ),
)
@pytest.mark.parametrize(
    argnames="coco_task",
    argvalues=(
        # "captions",
        "instances",
        # "person_keypoints",
    ),
)
def test_load_dataset(dataset_path: str, dataset_year: int, coco_task: str):
    all_dataset = ds.load_dataset(
        path=dataset_path, year=dataset_year, coco_task=coco_task
    )
    for split in all_dataset.keys():
        dataset = all_dataset[split]
        for _ in dataset:
            pass
