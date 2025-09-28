import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from datetime import datetime, timedelta

from sgp4.model import Satrec as Satrec_original
from sgpax.helper import jday
from sgpax.model import Satrec

import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])


def init_test_from_tle(satrec_class):
    # Hubble TLE
    sat = satrec_class.twoline2rv(
        "1 20580U 90037B   24225.65602021  .00023092  00000-0  10739-2 0  9999",
        "2 20580  28.4696 326.5721 0001735 301.1506  58.8917 15.18616974685623",
    )
    return sat


def return_result_after_minutes(sat, minutes):
    time_beginning = datetime(2024, 8, 18, 12, 30, 0)
    delta_t = minutes * 60.0
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


if __name__ == "__main__":
    sat = init_test_from_tle(Satrec)
    v_sat = init_test_from_tle(Satrec_original)

    minutes = list(range(0, 24 * 60, 1))
    errors = []
    for i in minutes:
        ve, vr, vv = return_result_after_minutes(v_sat, i)
        e, r, v = return_result_after_minutes(sat, i)
        # Compare accuracy
        print("After ", i, "hours")
        position_error_m = (r - jnp.array(vr)) * 1e3
        print("error in position metres", position_error_m)
        vel_error_m = (v - jnp.array(vv)) * 1e3
        print("error in velocity ", vel_error_m)

        pos_err = jnp.linalg.norm(position_error_m)
        errors.append(pos_err)

    plt.plot(minutes, errors)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Absolute error (metres)")
    plt.savefig("error_plot.png")
