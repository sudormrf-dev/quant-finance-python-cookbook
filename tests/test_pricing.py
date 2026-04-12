"""Tests for pricing.py."""

from __future__ import annotations

from patterns.pricing import (
    BlackScholesInputs,
    OptionContract,
    OptionStyle,
    OptionType,
    black_scholes_greeks,
    black_scholes_price,
)


class TestBlackScholesInputs:
    def setup_method(self):
        self.inputs = BlackScholesInputs(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
        )

    def test_is_valid(self):
        assert self.inputs.is_valid() is True

    def test_invalid_zero_spot(self):
        i = BlackScholesInputs(0.0, 100.0, 1.0, 0.05, 0.2)
        assert i.is_valid() is False

    def test_invalid_zero_vol(self):
        i = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.0)
        assert i.is_valid() is False

    def test_d1_atm(self):
        d1 = self.inputs.d1()
        assert isinstance(d1, float)

    def test_d2_lt_d1(self):
        assert self.inputs.d2() < self.inputs.d1()


class TestBlackScholesPrice:
    def setup_method(self):
        self.inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.2)

    def test_call_price_positive(self):
        price = black_scholes_price(self.inputs, OptionType.CALL)
        assert price > 0

    def test_put_price_positive(self):
        price = black_scholes_price(self.inputs, OptionType.PUT)
        assert price > 0

    def test_put_call_parity(self):
        call = black_scholes_price(self.inputs, OptionType.CALL)
        put = black_scholes_price(self.inputs, OptionType.PUT)
        import math

        s, k, r, t = 100.0, 100.0, 0.05, 1.0
        parity = call - put - s + k * math.exp(-r * t)
        assert abs(parity) < 1e-6

    def test_invalid_inputs_return_zero(self):
        invalid = BlackScholesInputs(0.0, 100.0, 1.0, 0.05, 0.2)
        assert black_scholes_price(invalid, OptionType.CALL) == 0.0

    def test_itm_call_higher_than_otm(self):
        itm = BlackScholesInputs(110.0, 100.0, 1.0, 0.05, 0.2)
        otm = BlackScholesInputs(90.0, 100.0, 1.0, 0.05, 0.2)
        assert black_scholes_price(itm, OptionType.CALL) > black_scholes_price(
            otm, OptionType.CALL
        )

    def test_higher_vol_higher_price(self):
        low_vol = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.1)
        high_vol = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.4)
        assert black_scholes_price(high_vol, OptionType.CALL) > black_scholes_price(
            low_vol, OptionType.CALL
        )


class TestBlackScholesGreeks:
    def setup_method(self):
        self.inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.2)

    def test_call_delta_between_0_and_1(self):
        g = black_scholes_greeks(self.inputs, OptionType.CALL)
        assert 0 < g["delta"] < 1

    def test_put_delta_negative(self):
        g = black_scholes_greeks(self.inputs, OptionType.PUT)
        assert g["delta"] < 0

    def test_gamma_positive(self):
        g = black_scholes_greeks(self.inputs, OptionType.CALL)
        assert g["gamma"] > 0

    def test_vega_positive(self):
        g = black_scholes_greeks(self.inputs, OptionType.CALL)
        assert g["vega"] > 0

    def test_theta_negative_call(self):
        g = black_scholes_greeks(self.inputs, OptionType.CALL)
        assert g["theta"] < 0

    def test_rho_positive_call(self):
        g = black_scholes_greeks(self.inputs, OptionType.CALL)
        assert g["rho"] > 0

    def test_invalid_inputs_zero_greeks(self):
        invalid = BlackScholesInputs(0.0, 100.0, 1.0, 0.05, 0.2)
        g = black_scholes_greeks(invalid, OptionType.CALL)
        assert g["delta"] == 0.0

    def test_all_greeks_present(self):
        g = black_scholes_greeks(self.inputs, OptionType.CALL)
        assert all(k in g for k in ["delta", "gamma", "vega", "theta", "rho"])


class TestOptionContract:
    def setup_method(self):
        self.contract = OptionContract(
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN,
            spot=105.0,
            strike=100.0,
            time_to_expiry=0.5,
            volatility=0.25,
            risk_free_rate=0.05,
        )

    def test_itm_call(self):
        assert self.contract.is_itm() is True

    def test_not_otm(self):
        assert self.contract.is_otm() is False

    def test_intrinsic_value_call(self):
        assert abs(self.contract.intrinsic_value() - 5.0) < 1e-9

    def test_atm_detection(self):
        atm = OptionContract(
            OptionType.CALL, OptionStyle.EUROPEAN, 100.0, 100.0, 1.0, 0.2, 0.05
        )
        assert atm.is_atm() is True

    def test_put_intrinsic_otm(self):
        put = OptionContract(
            OptionType.PUT, OptionStyle.EUROPEAN, 105.0, 100.0, 1.0, 0.2, 0.05
        )
        assert put.intrinsic_value() == 0.0

    def test_inputs_method(self):
        inp = self.contract.inputs()
        assert isinstance(inp, BlackScholesInputs)
        assert inp.spot == 105.0
