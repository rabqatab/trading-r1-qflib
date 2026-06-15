import compare_lab  # noqa: F401  (triggers path bootstrap)


def test_alpha_lab_importable():
    import alpha_lab.core as core
    assert hasattr(core, "load_context")
    assert hasattr(core, "select_universe")


def test_prices_path_points_at_submodule_data():
    import alpha_lab.core as core
    assert core.PRICES_PATH.name == "prices.parquet"
    assert "qf-lib-harness" in str(core.PRICES_PATH)
