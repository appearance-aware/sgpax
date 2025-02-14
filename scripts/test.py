from sgpax.model import Satrec
from sgpax.helper import jday
from datetime import datetime, timedelta

from sgp4.model import Satrec as vSatrec
import jax.numpy as jnp


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


if __name__ == "__main__":
    sat = init_test_from_tle(Satrec)
    v_sat = init_test_from_tle(vSatrec)

    for i in range(1, 40, 5):
        print("VANILLA")
        ve, vr, vv = return_result_after_hours(v_sat, i)
        print("SGPAX")
        e, r, v = return_result_after_hours(sat, i)
        # Compare accuracy
        print("After ", i, "hours")
        print("error in position metres", (r - jnp.array(vr)) * 1e3)
        print("error in velocity ", (v - jnp.array(vv)) * 1e3)
