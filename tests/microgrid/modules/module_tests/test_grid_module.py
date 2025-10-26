import numpy as np
import pytest

from src.pymgrid.modules import GridModule


def test_grid_module_converts_three_column_series():
    time_series = np.array([
        [0.2, 0.1, 0.0],
        [0.3, 0.15, 0.02],
    ])

    module = GridModule(
        max_import=10.0,
        max_export=5.0,
        time_series=time_series,
    )

    assert module.time_series.shape == (2, 4)
    np.testing.assert_allclose(module.time_series[:, :3], time_series)
    np.testing.assert_allclose(module.time_series[:, -1], np.ones(2))


def test_grid_module_online_bootstrap_defaults():
    module = GridModule(
        max_import=10.0,
        max_export=5.0,
        time_series=None,
        online=True,
    )

    assert module.online_mode
    assert module.time_series.shape == (1, 4)
    np.testing.assert_allclose(module.time_series[0], np.array([0.0, 0.0, 0.0, 1.0]))
    assert module.final_step == len(module)


def test_grid_module_online_ingestion_updates_series():
    module = GridModule(
        max_import=10.0,
        max_export=5.0,
        time_series=None,
        online=True,
        initial_time_series_value=[0.1, 0.05, 0.0, 1.0],
    )

    first = np.array([0.12, 0.06, 0.0, 1.0])
    second = np.array([0.15, 0.07, 0.01, 1.0])

    module.ingest_online_data(first)
    module.ingest_online_data(second, step=1)

    np.testing.assert_allclose(module.time_series[0], first)
    np.testing.assert_allclose(module.time_series[1], second)
    assert module.final_step == len(module)


def test_grid_module_rejects_non_binary_status_initial_value():
    with pytest.raises(ValueError, match=r"Last column \(grid status\) must contain binary values."):
        GridModule(
            max_import=10.0,
            max_export=5.0,
            time_series=None,
            online=True,
            initial_time_series_value=[0.1, 0.05, 0.0, 0.5],
        )
