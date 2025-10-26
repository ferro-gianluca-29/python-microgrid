"""Simulazione di controllo rule-based su una microgrid con dati in tempo reale.

Il file crea una microgrid con moduli di carico (load), produzione fotovoltaica (pv),
una batteria e l'accesso alla rete elettrica. Per 10 step di simulazione vengono
generati in tempo reale valori casuali di domanda e produzione rinnovabile, mentre
i prezzi della rete restano costanti. Il controller RuleBasedControl elabora le
azioni ottimali e viene calcolato il costo complessivo dell'operazione in euro.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np

from src.pymgrid import Microgrid
from src.pymgrid.algos import RuleBasedControl
from src.pymgrid.modules import BatteryModule, GridModule, LoadModule, RenewableModule


@dataclass
class SimulationConfig:
    """Configurazione della simulazione RBC."""

    num_steps: int = 10
    random_seed: int = 42
    grid_import_price: float = 0.20  # €/kWh
    grid_export_price: float = 0.08  # €/kWh
    grid_co2_intensity: float = 0.0  # kg/kWh (non influisce sul costo)


class RBCSimulation:
    """Esegue una simulazione step-by-step con RuleBasedControl."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)

        self.microgrid = self._build_microgrid(config)
        self.controller = RuleBasedControl(self.microgrid)

    @staticmethod
    def _build_microgrid(config: SimulationConfig) -> Microgrid:
        """Istanzia i moduli della microgrid con configurazione online."""

        load_module = LoadModule(
            time_series=None,
            online=True,
            initial_time_series_value=0.0,
            final_step=config.num_steps,
        )
        pv_module = RenewableModule(
            time_series=None,
            online=True,
            initial_time_series_value=0.0,
            final_step=config.num_steps,
        )

        grid_prices = np.array([
            [
                config.grid_import_price,
                config.grid_export_price,
                config.grid_co2_intensity,
                1.0,  # grid status sempre disponibile
            ]
        ])
        grid_module = GridModule(
            max_import=60.0,
            max_export=30.0,
            time_series=np.repeat(grid_prices, config.num_steps, axis=0),
            final_step=config.num_steps,
        )

        battery_module = BatteryModule(
            min_capacity=0.0,
            max_capacity=40.0,
            max_charge=10.0,
            max_discharge=10.0,
            efficiency=0.92,
            init_soc=0.5,
        )

        modules = [
            ("load", load_module),
            ("pv", pv_module),
            grid_module,
            battery_module,
        ]

        return Microgrid(modules=modules)

    def _generate_real_time_measurements(self) -> Mapping[str, float]:
        """Restituisce un dizionario con i valori casuali per pv e load."""

        load_value = self._rng.uniform(8.0, 15.0)
        pv_value = self._rng.uniform(0.0, 10.0)
        return {"load": load_value, "pv": pv_value}

    @staticmethod
    def _format_action(action: Mapping[str, Iterable[float]]) -> Dict[str, list[float]]:
        """Converte l'azione RBC in una forma stampabile."""

        formatted: Dict[str, list[float]] = {}
        for name, values in action.items():
            if isinstance(values, np.ndarray):
                formatted[name] = values.astype(float).tolist()
            elif isinstance(values, (list, tuple)):
                formatted[name] = [float(v) for v in values]
            else:
                formatted[name] = [float(values)]
        return formatted

    def run(self) -> float:
        """Esegue la simulazione restituendo il costo totale in euro."""

        self.microgrid.reset()
        self.controller.reset()
        total_cost_eur = 0.0

        for step in range(self.config.num_steps):
            measurements = self._generate_real_time_measurements()
            self.microgrid.ingest_real_time_data(measurements, step=step)

            action = self.controller.get_action()
            _, reward, done, _ = self.microgrid.step(action, normalized=False)

            step_cost = -reward
            total_cost_eur += step_cost

            action_display = self._format_action(action)

            print(f"Step {step + 1}/{self.config.num_steps}")
            print(f"  Load [kW]: {measurements['load']:.2f}")
            print(f"  PV   [kW]: {measurements['pv']:.2f}")
            print(f"  Azioni RBC: {action_display}")
            print(f"  Costo step [EUR]: {step_cost:.2f}\n")

            if done:
                break

        print("Simulazione completata.\n")
        print(f"Costo totale operativo: {total_cost_eur:.2f} EUR")

        return total_cost_eur


def main() -> None:
    config = SimulationConfig()
    simulation = RBCSimulation(config)
    simulation.run()


if __name__ == "__main__":
    main()
