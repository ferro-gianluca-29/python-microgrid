#!/usr/bin/env python3
"""Run a microgrid with streaming measurements in real time.

The script instantiates a microgrid with a load, renewable generation, grid,
and battery.  At every iteration it ingests fresh measurements for the load,
PV, and grid tariffs, writes them into the respective time-series buffers, and
performs a single control step.  The loop sleeps briefly between steps to mimic
real-time execution.

Usage
-----
Install the package in editable mode or point ``PYTHONPATH`` to ``src`` and run::

    PYTHONPATH=src python examples/online_realtime_microgrid.py

The example uses a deterministic pseudo-random generator so the numbers are
repeatable while still resembling realistic profiles.
"""
from __future__ import annotations

import itertools
import time
from typing import Generator, Tuple

import numpy as np

from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, GridModule, LoadModule, RenewableModule

# ---------------------------------------------------------------------------
# Configuration knobs for the demo.
# ---------------------------------------------------------------------------
STREAM_SLEEP_SECONDS = 0.3  # Pause between iterations to emulate real sensors.
STREAM_STEPS = 48           # Number of iterations before the demo stops.
INITIAL_BUFFER = 4          # Initial number of time steps allocated per module.


def build_microgrid() -> Tuple[Microgrid, LoadModule, RenewableModule, GridModule, BatteryModule]:
    """Create the microgrid and return the in-grid module instances.

    The :class:`Microgrid` constructor deep-copies incoming modules.  The
    returned modules are the copies living inside the microgrid so callers can
    safely mutate their internal time-series buffers.
    """
    load_template = LoadModule(np.zeros(INITIAL_BUFFER))
    pv_template = RenewableModule(np.zeros(INITIAL_BUFFER))

    tariff_template = np.tile(np.array([[0.23, 0.11, 0.45, 1.0]]), (INITIAL_BUFFER, 1))
    grid_template = GridModule(max_import=40.0, max_export=25.0, time_series=tariff_template)

    battery_template = BatteryModule(
        min_capacity=0.0,
        max_capacity=80.0,
        max_charge=15.0,
        max_discharge=20.0,
        efficiency=0.92,
        init_soc=0.55,
    )

    microgrid = Microgrid(
        modules=[
            ("load", load_template),
            ("pv", pv_template),
            ("grid", grid_template),
            ("battery", battery_template),
        ]
    )

    return (
        microgrid,
        microgrid.modules["load"].item(),
        microgrid.modules["pv"].item(),
        microgrid.modules["grid"].item(),
        microgrid.modules["battery"].item(),
    )


def ensure_future_capacity(module: LoadModule | RenewableModule | GridModule, min_free_rows: int = 1) -> None:
    """Extend a module time series if the buffer is about to run out.

    ``BaseTimeSeriesMicrogridModule`` instances mark ``done`` when the current
    step reaches ``final_step - 1``.  To keep stepping indefinitely we append a
    few empty rows whenever the remaining horizon falls below ``min_free_rows``.
    """
    rows_remaining = module.final_step - module.current_step - 1
    if rows_remaining >= min_free_rows:
        return

    add_rows = max(min_free_rows - rows_remaining, min_free_rows)
    padding = np.zeros((add_rows, module.time_series.shape[1]), dtype=float)
    module.time_series = np.vstack([module.time_series, padding])
    module.final_step = -1  # Re-synchronise the cached ``final_step`` with the new length.


def inject_scalar(module: LoadModule | RenewableModule, value: float, *, is_sink: bool) -> None:
    """Write the newest measurement into a single-column time series."""
    ensure_future_capacity(module)
    signed = -abs(value) if is_sink else abs(value)
    updated = module.time_series.copy()
    updated[module.current_step, 0] = signed
    module.time_series = updated


def inject_grid_prices(module: GridModule, price_row: np.ndarray) -> None:
    """Update the grid tariffs for the current step."""
    ensure_future_capacity(module)
    row = np.asarray(price_row, dtype=float).reshape(-1)
    if row.shape[0] not in (3, 4):
        raise ValueError("Grid price row must have three (price/co2) or four (price/co2/status) entries.")
    if row.shape[0] == 3:
        row = np.concatenate([row, np.ones(1, dtype=float)])
    updated = module.time_series.copy()
    updated[module.current_step, :] = row
    module.time_series = updated


def measurement_stream(seed: int = 7) -> Generator[Tuple[float, float, np.ndarray], None, None]:
    """Yield synthetic but realistic-looking measurements indefinitely."""
    rng = np.random.default_rng(seed)
    day_period = 96  # 96 quarter-hour samples in a day.

    for step in itertools.count():
        phase = (step % day_period) / day_period
        # Base load follows a daily sinusoid plus Gaussian jitter.
        base_load = 18.0 + 6.0 * np.sin(2.0 * np.pi * phase - np.pi / 3.0)
        load_kw = max(base_load + rng.normal(scale=1.5), 5.0)

        # PV generation peaks at midday with smaller noise and zero at night.
        solar_envelope = max(np.sin(np.pi * phase), 0.0)
        pv_kw = max(12.0 * solar_envelope + rng.normal(scale=0.8), 0.0)

        # Import/export prices and CO2 intensity oscillate during the day.
        import_price = max(0.22 + 0.05 * np.sin(2.0 * np.pi * phase + 0.2), 0.08)
        export_price = max(0.12 + 0.04 * np.cos(2.0 * np.pi * phase - 0.5), 0.04)
        co2_intensity = max(0.40 + 0.08 * np.sin(2.0 * np.pi * phase - 0.9), 0.05)
        grid_row = np.array([import_price, export_price, co2_intensity, 1.0], dtype=float)

        yield load_kw, pv_kw, grid_row


def dispatch_controls(net_demand: float, battery: BatteryModule, grid: GridModule) -> Tuple[float, float]:
    """Choose battery and grid actions that balance the residual demand."""
    battery_action = 0.0
    if net_demand > 0.0:
        discharge_cap = battery.max_production
        battery_action = min(net_demand, discharge_cap)
    elif net_demand < 0.0:
        charge_cap = battery.max_consumption
        battery_action = -min(-net_demand, charge_cap)

    residual = net_demand - battery_action
    grid_import_cap = grid.max_import
    grid_export_cap = grid.max_export
    grid_action = float(np.clip(residual, -grid_export_cap, grid_import_cap))
    return battery_action, grid_action


def main() -> None:
    microgrid, load_module, pv_module, grid_module, battery_module = build_microgrid()
    microgrid.reset()

    for step, (load_kw, pv_kw, grid_row) in zip(range(STREAM_STEPS), measurement_stream()):
        inject_scalar(load_module, load_kw, is_sink=True)
        inject_scalar(pv_module, pv_kw, is_sink=False)
        inject_grid_prices(grid_module, grid_row)

        net_demand = load_kw - pv_kw
        battery_action, grid_action = dispatch_controls(net_demand, battery_module, grid_module)

        obs, reward, done, info = microgrid.step(
            {"grid": [grid_action], "battery": [battery_action]},
            normalized=False,
        )

        print(
            f"step={step:02d} load={load_kw:5.2f} kW pv={pv_kw:5.2f} kW "
            f"battery={battery_action:+5.2f} kW grid={grid_action:+5.2f} kW "
            f"soc={battery_module.soc:4.2f} reward={reward:7.3f}"
        )

        if done:
            raise RuntimeError("Microgrid signalled termination; increase the buffer padding.")

        time.sleep(STREAM_SLEEP_SECONDS)

    print("Finished streaming demo.")


if __name__ == "__main__":
    main()
