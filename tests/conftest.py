import pytest


@pytest.fixture
def org_name() -> str:
    return "shunk031"


@pytest.fixture
def dataset_name() -> str:
    return "MSCOCO"


@pytest.fixture
def dataset_path(dataset_name: str) -> str:
    return f"{dataset_name}.py"


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"
