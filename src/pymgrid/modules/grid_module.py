import numpy as np
import yaml

from src.pymgrid.microgrid import DEFAULT_HORIZON
from src.pymgrid.modules.base import BaseTimeSeriesMicrogridModule


class GridModule(BaseTimeSeriesMicrogridModule):
    """
    An electrical grid module.

    By default, ``GridModule`` is a *fixed* module; it can be transformed to a flex module with ``GridModule.as_flex``.
    ``GridModule`` is the only built-in module that can be both a fixed and flex module.

    Parameters
    ----------
    max_import : float
        Maximum import at any time step.

    max_export : float
        Maximum export at any time step.

    time_series : array-like or None, shape (n_features, n_steps), n_features = {3, 4}
        If n_features=3, time series of ``(import_price, export_price, co2_per_kwH)`` in each column, respectively.
        Grid is assumed to have no outages.
        If n_features=4, time series of ``(import_price, export_price, co2_per_kwH, grid_status)``
        in each column, respectively. ``time_series[:, -1]`` -- the grid status -- must be binary.

    forecaster : callable, float, "oracle", or None, default None.
        Function that gives a forecast n-steps ahead.

        * If ``callable``, must take as arguments ``(val_c: float, val_{c+n}: float, n: int)``, where

          * ``val_c`` is the current value in the time series: ``self.time_series[self.current_step]``

          * ``val_{c+n}`` is the value in the time series n steps in the future

          * n is the number of steps in the future at which we are forecasting.

          The output ``forecast = forecaster(val_c, val_{c+n}, n)`` must have the same sign
          as the inputs ``val_c`` and ``val_{c+n}``.

        * If ``float``, serves as a standard deviation for a mean-zero gaussian noise function
          that is added to the true value.

        * If ``"oracle"``, gives a perfect forecast.

        * If ``None``, no forecast.

    forecast_horizon : int.
        Number of steps in the future to forecast. If forecaster is None, ignored and 0 is returned.

    forecaster_increase_uncertainty : bool, default False
        Whether to increase uncertainty for farther-out dates if using a GaussianNoiseForecaster. Ignored otherwise.

    cost_per_unit_co2 : float, default 0.0
        Marginal cost of grid co2 production.

    normalized_action_bounds : tuple of int or float, default (0, 1).
        Bounds of normalized actions.
        Change to (-1, 1) for e.g. an RL policy with a Tanh output activation.

    raise_errors : bool, default False
        Whether to raise errors if bounds are exceeded in an action.
        If False, actions are clipped to the limit possible.

    online : bool, default False
        If True, the module expects measurements to be provided incrementally using
        :meth:`.ingest_online_data` instead of being initialized with a complete time
        series.

    initial_time_series_value : Sequence[float] or None, default None
        Bootstrap value for the internal time series when ``online`` is True and
        ``time_series`` is None. The sequence must contain three or four non-negative
        entries corresponding to ``(import_price, export_price, co2_per_kwH, grid_status)``.
        If the status component is omitted it defaults to 1 (grid available).

    """

    module_type = ('grid', 'controllable')

    yaml_tag = u"!GridModule"
    yaml_loader = yaml.SafeLoader
    yaml_dumper = yaml.SafeDumper

    state_components = np.array(['import_price', 'export_price', 'co2_per_kwh', 'grid_status'], dtype=object)

    def __init__(self,
                 max_import,
                 max_export,
                 time_series,
                 forecaster=None,
                 forecast_horizon=DEFAULT_HORIZON,
                 forecaster_increase_uncertainty=False,
                 forecaster_relative_noise=False,
                 initial_step=0,
                 final_step=-1,
                 cost_per_unit_co2=0.0,
                 normalized_action_bounds=(0, 1),
                 raise_errors=False,
                 online=False,
                 initial_time_series_value=None):

        time_series, initial_time_series_value = self._check_params(
            max_import,
            max_export,
            time_series,
            online,
            initial_time_series_value,
        )
        self.max_import, self.max_export = max_import, max_export
        self.cost_per_unit_co2 = cost_per_unit_co2

        super().__init__(
            time_series,
            raise_errors,
            forecaster=forecaster,
            forecast_horizon=forecast_horizon,
            forecaster_increase_uncertainty=forecaster_increase_uncertainty,
            forecaster_relative_noise=forecaster_relative_noise,
            initial_step=initial_step,
            final_step=final_step,
            normalized_action_bounds=normalized_action_bounds,
            provided_energy_name='grid_import',
            absorbed_energy_name='grid_export',
            online=online,
            initial_time_series_value=initial_time_series_value
        )

    def _check_params(self,
                      max_import,
                      max_export,
                      time_series,
                      online,
                      initial_time_series_value):
        if max_import < 0:
            raise ValueError('parameter max_import must be non-negative.')
        if max_export < 0:
            raise ValueError('parameter max_export must be non-negative.')
        validated_time_series = None

        def _format_timeseries(data, label):
            arr = np.array(data, dtype=float)

            if arr.ndim == 1:
                arr = arr.reshape((1, arr.shape[0]))
            elif arr.ndim != 2:
                raise ValueError('Time series must be two dimensional with three or four columns.'
                                 'See docstring for details.')

            if arr.shape[1] not in (3, 4):
                raise ValueError('Time series must be two dimensional with three or four columns.'
                                 'See docstring for details.')

            if arr.shape[1] == 4:
                status = arr[:, -1]
                if not np.isin(status, (0, 1)).all():
                    raise ValueError("Last column (grid status) must contain binary values.")
            else:
                expanded = np.ones((arr.shape[0], 4))
                expanded[:, :3] = arr
                arr = expanded

            if (arr < 0).any():
                if label == 'initial_time_series_value':
                    raise ValueError('initial_time_series_value must be non-negative.')
                raise ValueError('Time series must be non-negative.')

            return arr

        if time_series is not None:
            validated_time_series = _format_timeseries(time_series, 'time_series')
        elif not online:
            raise ValueError('time_series cannot be None when online mode is disabled.')

        validated_initial_value = None

        if initial_time_series_value is not None:
            initial_arr = _format_timeseries(initial_time_series_value, 'initial_time_series_value')
            if initial_arr.shape[0] != 1:
                if online:
                    raise ValueError('initial_time_series_value must define a single step when using online mode.')
                raise ValueError('initial_time_series_value must define a single step.')
            validated_initial_value = initial_arr.reshape(-1)
        elif online and validated_time_series is None:
            validated_initial_value = np.array([0.0, 0.0, 0.0, 1.0])

        return validated_time_series, validated_initial_value

    def _get_bounds(self):
        min_obs = self._time_series.min(axis=0)
        max_obs = self._time_series.max(axis=0)
        assert len(min_obs) in (3, 4)

        min_act, max_act = -1 * self.max_export, self.max_import

        return min_obs, max_obs, min_act, max_act

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_source + as_sink == 1, 'Must act as either source or sink but not both or neither.'
        reward = self.get_cost(external_energy_change, as_source, as_sink)
        info_key = 'provided_energy' if as_source else 'absorbed_energy'
        info = {info_key: external_energy_change,
                'co2_production': self.get_co2_production(external_energy_change, as_source, as_sink)}

        return reward, self._done(), info

    def get_cost(self, import_export, as_source, as_sink):
        """
        Current cost of the grid's usage.

        Includes both the cost of importing/exporting as well as the cost of carbon dioxide production.
        Note that the "cost" of exporting may be negative as the module may receive revenue in exchange for
        energy export.
        If the module is exporting to the grid, co2 production cost will be zero.

        Parameters
        ----------
        import_export : float
            Amount of energy that is imported or exported.
        as_source : bool
            Whether the grid is acting as a source.
        as_sink : bool
            Whether the grid is acting as a sink.

        Returns
        -------
        cost : float
            Cost of using the grid.

        """
        if as_source:  # Import
            import_cost = self._time_series[self.current_step, 0]
            return -1 * import_cost*import_export + self.get_co2_cost(import_export, as_source, as_sink)
        elif as_sink:  # Export
            export_cost = self._time_series[self.current_step, 1]
            return export_cost * import_export + self.get_co2_cost(import_export, as_source, as_sink)
        else:
            raise RuntimeError

    def get_co2_cost(self, import_export, as_source, as_sink):
        """
        Current cost of the carbon dioxide production of the grid's usage.

        If the module is exporting to the grid, co2 production cost will be zero.

        Parameters
        ----------
        import_export : float
            Amount of energy that is imported or exported.
        as_source : bool
            Whether the grid is acting as a source.
        as_sink : bool
            Whether the grid is acting as a sink.

        Returns
        -------
        co2_cost : float
            Cost of carbon dioxide production.

        """
        return -1.0 * self.cost_per_unit_co2*self.get_co2_production(import_export, as_source, as_sink)

    def get_co2_production(self, import_export, as_source, as_sink):
        """

        Current carbon dioxide production of the grid's usage.

        If the module is exporting to the grid, co2 production will be zero.

        Parameters
        ----------
        import_export : float
            Amount of energy that is imported or exported.
        as_source : bool
            Whether the grid is acting as a source.
        as_sink : bool
            Whether the grid is acting as a sink.

        Returns
        -------
        co2_production : float
            Carbon dioxide production.

        """
        if as_source:  # Import
            co2_prod_per_kWh = self._time_series[self.current_step, 2]
            co2 = import_export*co2_prod_per_kWh
            return co2
        elif as_sink:  # Export
            return 0.0
        else:
            raise RuntimeError

    def as_flex(self):
        """
        Convert the module to a flex module.

        Flex modules do not require a control to be passed, and are deployed as necessary to balance load and demand.

        """
        self.__class__.module_type = (self.__class__.module_type[0], 'flex')

    def as_fixed(self):
        """
        Convert the module to a fixed module.

        Flex modules require a control to be passed.

        """
        self.__class__.module_type = (self.__class__.module_type[0], 'fixed')

    @property
    def import_price(self):
        """
        Current and forecasted import prices.

        Returns
        -------
        prices : np.ndarray, shape (forecast_horizon, )
            prices[0] gives the current import price while prices[1:] gives forecasted import prices.

        """
        return self.state[::4]

    @property
    def export_price(self):
        """
        Current and forecasted export prices.

        Returns
        -------
        prices : np.ndarray, shape (forecast_horizon, )
            prices[0] gives the current export price while prices[1:] gives forecasted export prices.

        """
        return self.state[1::4]

    @property
    def co2_per_kwh(self):
        """
        Current and forecasted carbon dioxide production per kWh.

        Returns
        -------
        marginal_production : np.ndarray, shape (forecast_horizon, )
            marginal_production[0] gives the current production per kWh while
            marginal_production[1:] gives forecasted production per kWh.

        """
        return self.state[2::4]

    @property
    def grid_status(self):
        """
        Current and forecasted grid status.

        Returns
        -------
        status : np.ndarray, shape (forecast_horizon, )
            status[0] gives the current status of the grid while  status[1:] gives forecasted status.

        """
        return self.state[3::4]

    @property
    def current_status(self):
        """
        Current status of the grid.

        Returns
        -------
        status : {0, 1}
            Current status.

        """
        return self.grid_status[0]

    @property
    def max_production(self):
        return self.max_import * self.current_status

    @property
    def max_consumption(self):
        return self.max_export * self.current_status

    @property
    def production_marginal_cost(self):
        return -1 * self.get_cost(1, as_source=True, as_sink=False)

    @property
    def absorption_marginal_cost(self):
        return self.get_cost(1, as_source=False, as_sink=True)

    @property
    def is_source(self):
        return True

    @property
    def is_sink(self):
        return True

    @property
    def weak_grid(self):
        """
        Whether the grid has outages or not.

        Returns
        -------
        weak_grid : bool
            True if the grid has outages.

        """
        return self._time_series[:, -1].min() < 1

    def __repr__(self):
        return f'GridModule(max_import={self.max_import}, max_export={self.max_export})'
