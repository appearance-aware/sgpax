from sgpax.model import Satrec
from sgpax.helper import jday
from datetime import datetime, timedelta

from sgp4.model import Satrec as vSatrec
import jax


def test_output(satrec_class):
    # print name of module the class is from
    print(f"OUTPUT FROM {satrec_class.__module__}")
    sat = satrec_class.twoline2rv(
        "1 20580U 90037B   24225.65602021  .00023092  00000-0  10739-2 0  9999",
        "2 20580  28.4696 326.5721 0001735 301.1506  58.8917 15.18616974685623",
    )
    time_beginning = datetime(2024, 8, 18, 12, 30, 0)
    delta_t = 40 * 60 * 60.0
    dt = time_beginning + timedelta(seconds=delta_t)
    jd, fr = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6
    )
    try:
        print(r:=sat.sgp4(jd, fr))
        # return r
    except Exception as e:
        print(e)
    return sat



if __name__ == "__main__":
    # sat = Satrec.twoline2rv(
    #     "1 20580U 90037B   24225.65602021  .00023092  00000-0  10739-2 0  9999",
    #     "2 20580  28.4696 326.5721 0001735 301.1506  58.8917 15.18616974685623",
    # )
    # time_beginning = datetime(2024, 8, 18, 12, 30, 0)
    # delta_t = 40 * 60 * 60.0
    # dt = time_beginning + timedelta(seconds=delta_t)
    # jd, fr = jday(
    #     dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6
    # )
    # dsgp4 = jax.jacobian(sat.sgp4, argnums=1)


    print("VANILLA")
    vs = test_output(vSatrec)
    print("SGPAX")
    s = test_output(Satrec)
    # Sort alphabetically

    for slot in sorted(vs.__slots__):
        try:
            print(slot, getattr(vs, slot))
        except:
            pass

    for k, v in sorted(s.__dict__.items()):
        print(k, v)
