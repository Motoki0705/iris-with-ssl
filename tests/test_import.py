"""Smoke tests for verifying the ml package skeleton is importable."""


def test_import_ml() -> None:
    """Ensure that the ml package can be imported."""

    import ml  # noqa: F401
