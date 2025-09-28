import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from sgpax.model import Satrec
from sgp4.model import Satrec as Satrec_nondiff


def init_from_tle(satrec_class, satname="Hubble"):
    if satname == "Hubble":
        tle = [
            "1 20580U 90037B   24225.65602021  .00023092  00000-0  10739-2 0  9999",
            "2 20580  28.4696 326.5721 0001735 301.1506  58.8917 15.18616974685623",
        ]
    elif satname == "Sentinel-6":
        tle = [
            "1 46984U 20086A   25270.59419674 -.00000044  00000-0  72435-4 0  9992",
            "2 46984  66.0427 320.4662 0008008 268.4259  91.5839 12.80930709226692",
        ]
    elif satname == "CloudSat":
        tle = [
            "1 29107U 06016A   25270.88464860  .00002404  00000-0  28508-3 0  9998",
            "2 29107  98.4214 288.3512 0006702  98.6416 261.5557 14.85542177 35550",
        ]
    else:
        raise ValueError(f"Unknown satellite {satname}.")
    return satrec_class.twoline2rv(tle[0], tle[1])


def test_compare_against_python_sgp4(
    satname="Hubble", 
    nevalpoints=11,
    verbose=True, 
):
    
    sat1 = init_from_tle(Satrec, satname)
    sat2 = init_from_tle(Satrec_nondiff, satname)

    r_errors = []
    v_errors = []
    hours = jnp.linspace(0, 48, nevalpoints)
    for i in hours:
        
        # Compute position and velocity
        elapsed_minutes = i * 60.0
        _, r1, v1 = sat1.sgp4_tsince(elapsed_minutes)
        _, r2, v2 = sat2.sgp4_tsince(elapsed_minutes)
        r2, v2 = jnp.array(r2), jnp.array(v2)
        
        # Compare accuracy
        position_error_m = jnp.abs(r1 - r2) * 1e3
        velocity_error_ms = jnp.abs(v1 - v2) * 1e3
        position_error_mag = jnp.linalg.norm(position_error_m)
        velocity_error_mag = jnp.linalg.norm(velocity_error_ms)
        
        if verbose:
            print("After ", i, "hours")
            print("Error in position (m):      ", position_error_m)
            print("Error in velocity (m/s):    ", velocity_error_ms)
            print("Total position error (m):   ", position_error_mag)
            print("Total velocity error (m/s): ", velocity_error_mag, "\n")
        
        r_errors.append(position_error_mag)
        v_errors.append(velocity_error_mag)
        
        POSITION_ERROR_THRESHOLD_M = 0.1
        assert position_error_mag < POSITION_ERROR_THRESHOLD_M, "Not accurate enough compared to reference! Check 64 bit float accuracy is turned on"
        
    return {
        "pos_errors": r_errors, 
        "vel_errors": v_errors,
        "time_hours": hours,
    }
        

def test_r_derivative(satname="Hubble", verbose=True):
    
    # Simulate for 1.0 minute
    time = 1.0
    sat = init_from_tle(Satrec, satname)
    _, _, v = sat.sgp4_tsince(time)
    
    # Take derivative of position and compare to velocity
    drdt = jax.jacfwd(lambda t: sat.sgp4_tsince(t)[1])
    gradv = drdt(time)
    vel_error = (gradv/60 - v)*1e3
    error_mag = jnp.linalg.norm(vel_error)
    
    print("Velocity error via autodiff: (m/s)", vel_error)
    
    VELOCITY_ERROR_THRESHOLD_M_PER_S = 8 # TODO: Why are we ok with 8m/s error?
    assert error_mag < VELOCITY_ERROR_THRESHOLD_M_PER_S, "Derivative is not accurate enough"


if __name__ == "__main__":
    test_compare_against_python_sgp4(verbose=False)
    test_r_derivative(verbpse=False)
