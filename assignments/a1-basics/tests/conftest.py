from pathlib import Path
import pytest

FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures"

@pytest.fixture
def snapshot():
    class Snapshot:
        def assert_match(self, actual, *args, **kwargs):
            raise NotImplementedError("Snapshot comparison requires fetched reference snapshots")
    return Snapshot()

@pytest.fixture
def numpy_snapshot(snapshot):
    return snapshot
