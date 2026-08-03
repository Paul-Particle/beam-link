"""Test-wide setup: keep the array cache out of the developer's real one.

`physics.cache` is deliberately persistable, so a suite that used the default location would both
pollute it and be unable to tell a cold path from a warm one.
"""

import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_cache():
    with tempfile.TemporaryDirectory(prefix="beamlink-test-cache-") as directory:
        previous = os.environ.get("BEAMLINK_CACHE_DIR")
        os.environ["BEAMLINK_CACHE_DIR"] = directory
        yield directory
        if previous is None:
            os.environ.pop("BEAMLINK_CACHE_DIR", None)
        else:
            os.environ["BEAMLINK_CACHE_DIR"] = previous
