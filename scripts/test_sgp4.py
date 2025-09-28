import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from sgp4.model import Satrec as vSatrec
from datetime import datetime, timedelta
from sgpax.helper import jday
from sgpax.model import Satrec


def init_test_from_tle(satrec_class):
    # Hubble TLE
    sat = satrec_class.twoline2rv(
        "1 20580U 90037B   24225.65602021  .00023092  00000-0  10739-2 0  9999",
        "2 20580  28.4696 326.5721 0001735 301.1506  58.8917 15.18616974685623",
    )
    return sat


def return_result_after_hours(sat, hours):
    time_beginning = datetime(2024, 8, 18, 12, 30, 0)
    delta_t = hours * 60 * 60.0
    dt = time_beginning + timedelta(seconds=delta_t)
    jd, fr = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6
    )
    return sat.sgp4(jd, fr)


def print_sat_elems(sat, vanilla_sat):
    # Sort alphabetically and print
    for slot in sorted(vanilla_sat.__slots__):
        try:
            print(slot, getattr(vanilla_sat, slot))
        except:
            pass

    for k, v in sorted(sat.__dict__.items()):
        print(k, v)


def test_compare_against_python_sgp4():
    POSITION_ERROR_THRESHOLD_M = 0.1
    sat = init_test_from_tle(Satrec)
    v_sat = init_test_from_tle(vSatrec)

    for i in range(1, 40, 5):

        # Simulate orbits
        ve, vr, vv = return_result_after_hours(v_sat, i)
        e, r, v = return_result_after_hours(sat, i)
        
        # Compare accuracy
        position_error_m = (r - jnp.array(vr)) * 1e3
        velocity_error_ms = (v - jnp.array(vv)) * 1e3
        position_error_mag = jnp.linalg.norm(position_error_m)
        velocity_error_mag = jnp.linalg.norm(velocity_error_ms)
        
        print("After ", i, "hours")
        print("Error in position (m):      ", position_error_m)
        print("Error in velocity (m/s):    ", velocity_error_ms)
        print("Total position error (m):   ", position_error_mag)
        print("Total velocity error (m/s): ", velocity_error_mag, "\n")

        pos_err = jnp.linalg.norm(position_error_m)
        assert pos_err < POSITION_ERROR_THRESHOLD_M, "Not accurate enough compared to reference! Check 64 bit float accuracy is turned on"


def test_r_derivative():
    VELOCITY_ERROR_THRESHOLD_M_PER_S = 8
    sat = init_test_from_tle(Satrec)
    t = 1.0
    e, r, v = sat.sgp4_tsince(t)
    drdt = jax.jacfwd(lambda t: sat.sgp4_tsince(t)[1])
    gradv = drdt(t)
    vel_error = (gradv/60 - v)*1e3
    print("Velocity error via autodiff: ", vel_error)
    assert jnp.linalg.norm(
        vel_error) < VELOCITY_ERROR_THRESHOLD_M_PER_S, "Derivative is not accurate enough"


if __name__ == "__main__":
    test_compare_against_python_sgp4()
    test_r_derivative()
