import pytest

from objectmodel import synthetic as s


@pytest.fixture(scope="session")
def tracks():
    """One noisy capture per catalogue object, shared across tests."""
    return {name: (f(), s.generate_tracks(f(), seed=11)) for name, f in s.CATALOGUE.items()}
