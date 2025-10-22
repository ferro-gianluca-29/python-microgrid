import numpy as np
import pytest

from src.pymgrid import Microgrid
from src.pymgrid.modules import GridModule, LoadModule, RenewableModule


def test_online_real_time_simulation():
    rng = np.random.default_rng(42)
    steps = 6

    horizon = steps + 5
    load_template = LoadModule(
        time_series=None,
        online=True,
        initial_time_series_value=0.0,
        final_step=horizon,
    )
    pv_template = RenewableModule(
        time_series=None,
        online=True,
        initial_time_series_value=0.0,
        final_step=horizon,
    )

    grid_row = np.array([0.15, 0.05, 0.0, 1.0])
    grid_series = np.tile(grid_row, (horizon, 1))
    grid_module = GridModule(max_import=50.0, max_export=50.0, time_series=grid_series)

    microgrid = Microgrid(modules=[('load', load_template), ('pv', pv_template), grid_module])
    load_module = microgrid.modules['load'][0]
    pv_module = microgrid.modules['pv'][0]
    microgrid.reset()

    for step in range(steps):
        load_value = rng.uniform(5.0, 15.0)
        pv_value = rng.uniform(0.0, 10.0)

        microgrid.ingest_real_time_data({'load': load_value, 'pv': pv_value})

        measurements = microgrid.fetch_real_time_data(['load', 'pv'])
        assert pytest.approx(load_value) == measurements['load'][0]
        assert pytest.approx(pv_value) == measurements['pv'][0]
        assert pytest.approx(-load_value) == load_module.time_series[step, 0]

        observations, reward, done, info = microgrid.step({'grid': [0.0]})
        assert 'load' in observations
        assert 'pv' in observations

    assert microgrid.current_step == load_module.initial_step + steps
    assert len(load_module.time_series) >= steps
    assert len(pv_module.time_series) >= steps
