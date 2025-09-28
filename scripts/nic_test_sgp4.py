import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from sgp4.model import Satrec as Satrec_nondiff
from sgpax.model import Satrec

from datetime import datetime, timedelta
from sgpax.helper import jday


def init_from_tle(satrec_class, satname="Hubble"):
    if satname == "Hubble":
        tle = [
            "1 20580U 90037B   24225.65602021  .00023092  00000-0  10739-2 0  9999",
            "2 20580  28.4696 326.5721 0001735 301.1506  58.8917 15.18616974685623",
        ]
    else:
        raise ValueError(f"Unknown satellite {satname}.")
    return satrec_class.twoline2rv(tle[0], tle[1])

def return_result_after_hours(sat, hours):
    # time_beginning = datetime(2024, 8, 18, 12, 30, 0)
    time_beginning = datetime(2024, 8, 12, 15, 44, 40, 146144)
    delta_t = hours * 60 * 60.0
    dt = time_beginning + timedelta(seconds=delta_t)
    jd, fr = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6
    )
    # return sat.sgp4(jd, fr)
    
    epoch = 27253.656020210125
    whole, fraction = divmod(epoch, 1.0)
    whole_jd = whole + 2433281.5
    jdsatepoch = whole_jd
    jdsatepochF = fraction
    
    minutes_per_day = 1440.0
    tsince = (jd - jdsatepoch) * minutes_per_day + (
        fr - jdsatepochF
    ) * minutes_per_day
    
    print("Computed tsince (hrs): ", tsince / 60.0)
    return sat.sgp4_tsince(tsince)


def test_compare_against_python_sgp4(satname="Hubble"):
    
    sat1 = init_from_tle(Satrec, satname)
    sat2 = init_from_tle(Satrec_nondiff, satname)

    for i in [0]: #range(1, 40, 10):
        
        # Compute position and velocity
        elapsed_minutes = i * 60.0
        _, r1, v1 = sat1.sgp4_tsince(elapsed_minutes)
        _, r2, v2 = sat2.sgp4_tsince(elapsed_minutes)
        
        # TODO: Why does this lead to worse error?
        # TODO: It change the values for our one, but not for original SGP4
        # _, r1, v1 = sat1.sgp4_tsince(elapsed_minutes)
        # _, r2, v2 = return_result_after_hours(sat1, i)
        r2, v2 = jnp.array(r2), jnp.array(v2)
        
        # Compare accuracy
        position_error_m = jnp.abs(r1 - r2) * 1e3
        velocity_error_ms = jnp.abs(v1 - v2) * 1e3
        position_error_mag = jnp.linalg.norm(position_error_m)
        velocity_error_mag = jnp.linalg.norm(velocity_error_ms)
        
        print("After ", i, "hours")
        print("Error in position (m):      ", position_error_m)
        print("Error in velocity (m/s):    ", velocity_error_ms)
        print("Total position error (m):   ", position_error_mag)
        print("Total velocity error (m/s): ", velocity_error_mag, "\n")
        
        POSITION_ERROR_THRESHOLD_M = 0.1
        # assert position_error_mag < POSITION_ERROR_THRESHOLD_M, "Not accurate enough compared to reference! Check 64 bit float accuracy is turned on"

def test_r_derivative(satname="Hubble"):
    
    # Simulate for 1.0 minute
    time = 1.0
    sat = init_from_tle(Satrec, satname)
    _, _, v = sat.sgp4_tsince(time)
    
    # Take derivative of position and compare to velocity
    drdt = jax.jacfwd(lambda t: sat.sgp4_tsince(t)[1])
    gradv = drdt(time)
    vel_error = (gradv/60 - v)*1e3
    error_mag = jnp.linalg.norm(vel_error)
    
    print("Velocity error via autodiff: ", vel_error)
    
    VELOCITY_ERROR_THRESHOLD_M_PER_S = 8 # Why are we ok with 8m/s error?
    assert error_mag < VELOCITY_ERROR_THRESHOLD_M_PER_S, "Derivative is not accurate enough"


if __name__ == "__main__":
    test_compare_against_python_sgp4()
    # test_r_derivative()
    
    
"""
There is an initial error in the velocity of the spacecraft (not position).
Things to check when back from walk are as follows:

- Is the computed julian date epoch the same in both methods?
- Are there steps in the velocity computation that are wrong?
- Does velocity get worse over time? Make the plot
"""
