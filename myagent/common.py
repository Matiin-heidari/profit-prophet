# exports the name of the training algorithm
from pathlib import Path

import numpy as np
from gymnasium import spaces
from negmas.outcomes import Outcome
from scml.oneshot.rl.observation import FlexibleObservationManager
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.context import GeneralContext, SupplierContext, ConsumerContext
from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm

TrainingAlgorithm: type[BaseAlgorithm] = A2C
"""The algorithm used for training. You can use any stable_baselines3 algorithm or develop your own"""

MODEL_PATH = Path(__file__).parent / "models" / "mymodel"
"""The path in which train.py saves the trained model and from which myagent.py loads it"""


def make_context(as_supplier: bool) -> GeneralContext:
    """Generates a context as a supplier or as a consumer"""
    if as_supplier:
        return SupplierContext()

    return ConsumerContext()


class MyObservationManager(FlexibleObservationManager):
    """This is my observation manager implementing encoding and decoding the state used by the RL algorithm"""
    def make_space(self) -> spaces.MultiDiscrete | spaces.Box:
        """Creates the observation space"""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(18,),
            dtype=np.float32,
        )

    def encode(self, awi: OneShotAWI) -> np.ndarray:
        """Encodes an observation from the agent's state"""
        # =========================================================
        # NEGOTIATION STATISTICS
        # =========================================================

        buy_prices = []
        sell_prices = []

        for _, details in awi.current_negotiation_details["buy"].items():

            offer = details.nmi.state.current_offer

            if offer is None:
                continue

            unit_price = offer[2]
            buy_prices.append(unit_price)

        for _, details in awi.current_negotiation_details["sell"].items():

            offer = details.nmi.state.current_offer

            if offer is None:
                continue

            unit_price = offer[2]
            sell_prices.append(unit_price)

        # =========================================================
        # RELATIVE BUY / SELL PRICES
        # =========================================================

        trading_price_sell = awi.trading_prices[awi.my_output_product]
        trading_price_buy = awi.trading_prices[awi.my_input_product]
        avg_buy_price = (
            np.mean(buy_prices) / trading_price_buy
            if buy_prices else 0.0
        )

        avg_sell_price = (
            np.mean(sell_prices) / trading_price_sell
            if sell_prices else 0.0
        )

        # =========================================================
        # AGREEMENT / CONTRACT FEATURES
        # =========================================================

        signed_sales = awi.total_sales_at(awi.current_step) / 10
        signed_supplies = awi.total_supplies_at(awi.current_step) / 10

        # =========================================================
        # SHORTAGE PRESSURE
        # =========================================================

        shortage_pressure = np.tanh(
            (awi.needed_sales + awi.needed_supplies) / 10
        )

        # =========================================================
        # FINAL OBSERVATION VECTOR
        # =========================================================

        return np.array([

            # -----------------------------------------------------
            # TIME FEATURES
            # -----------------------------------------------------

            awi.current_step / awi.n_steps,

            # -----------------------------------------------------
            # AGENT FEATURES
            # -----------------------------------------------------

            np.tanh(awi.current_balance / 1000),

            
            #   first level = 0
            #   middle = ~0.5
            #   last = ~1
            awi.level / max(1, awi.n_processes),

            # -----------------------------------------------------
            # NEED FEATURES
            # -----------------------------------------------------

            awi.needed_sales / (awi.n_lines + 1),
            awi.needed_supplies / 10,

            # -----------------------------------------------------
            # NEGOTIATION ACTIVITY
            # -----------------------------------------------------

            len(awi.current_negotiation_details["buy"]) / 10,
            len(awi.current_negotiation_details["sell"]) / 10,

            # -----------------------------------------------------
            # PRICE FEATURES
            # -----------------------------------------------------

            np.tanh(avg_buy_price),
            np.tanh(avg_sell_price),

            # -----------------------------------------------------
            # AGREEMENT FEATURES
            # -----------------------------------------------------

            signed_sales,
            signed_supplies,

            # -----------------------------------------------------
            # MARKET PRESSURE
            # -----------------------------------------------------

            shortage_pressure,

            len(awi.my_suppliers) / 10,
            len(awi.my_consumers) / 10,
            awi.n_lines / awi.max_n_lines,
            awi.current_exogenous_input_quantity / awi.n_lines,
            awi.current_exogenous_output_quantity / awi.n_lines,
            awi.n_competitors / 10,

        ], dtype=np.float32)

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi) # to be consistent with the changed encode function

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """Gets the offers from an encoded state"""
        return super().get_offers(awi, encoded)


# ensure that the folder containing models is created
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
