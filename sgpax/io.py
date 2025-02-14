from .helper import jday, days2mdhms, invjday
from math import pi
from datetime import datetime

error_message = "ERROR: {}"


def twoline2rv(longstr1, longstr2):
    """Return a Satellite imported from two lines of TLE data.

    Provide the two TLE lines as strings `longstr1` and `longstr2`,
    and select which standard set of gravitational constants you want
    by providing `gravity_constants`:

    `sgp4.earth_gravity.wgs72` - Standard WGS 72 model
    `sgp4.earth_gravity.wgs84` - More recent WGS 84 model
    `sgp4.earth_gravity.wgs72old` - Legacy support for old SGP4 behavior

    Normally, computations are made using various recent improvements
    to the algorithm.  If you want to turn some of these off and go
    back into "opsmode" mode, then set `opsmode` to `a`.

    """

    deg2rad = pi / 180.0
    xpdotp = 1440.0 / (2.0 * pi)

    line = longstr1.rstrip()

    try:
        longstr1.encode("ascii")
        longstr2.encode("ascii")
    except UnicodeEncodeError:
        r1 = repr(longstr1)[1:-1]
        r2 = repr(longstr2)[1:-1]
        raise ValueError(
            "your TLE lines are broken because they contain"
            " non-ASCII characters:\n\n%s\n%s" % (r1, r2)
        )

    if (
        len(line) >= 64
        and line.startswith("1 ")
        and line[8] == " "
        and line[23] == "."
        and line[32] == " "
        and line[34] == "."
        and line[43] == " "
        and line[52] == " "
        and line[61] == " "
        and line[63] == " "
    ):

        satnum_str = line[2:7]
        classification = line[7] or "U"
        intldesg = line[9:17].rstrip()
        two_digit_year = int(line[18:20])
        epochdays = float(line[20:32])
        ndot = float(line[33:43])
        nddot = float(line[44] + "." + line[45:50])
        nexp = int(line[50:52])
        bstar = float(line[53] + "." + line[54:59])
        ibexp = int(line[59:61])
        ephtype = line[62]
        elnum = int(line[64:68])
    else:
        raise ValueError(error_message.format(line))

    line = longstr2.rstrip()

    if (
        len(line) >= 68
        and line.startswith("2 ")
        and line[7] == " "
        and line[11] == "."
        and line[16] == " "
        and line[20] == "."
        and line[25] == " "
        and line[33] == " "
        and line[37] == "."
        and line[42] == " "
        and line[46] == "."
        and line[51] == " "
    ):

        if satnum_str != line[2:7]:
            raise ValueError("Object numbers in lines 1 and 2 do not match")

        inclo = float(line[8:16])
        nodeo = float(line[17:25])
        ecco = float("0." + line[26:33].replace(" ", "0"))
        argpo = float(line[34:42])
        mo = float(line[43:51])
        no_kozai = float(line[52:63])
        revnum = line[63:68]
    # except (AssertionError, IndexError, ValueError):
    else:
        # raise ValueError(error_message.format(2, LINE2, line))
        raise ValueError()

    #  ---- find no, ndot, nddot ----
    no_kozai = no_kozai / xpdotp
    #   rad/min
    nddot = nddot * pow(10.0, nexp)
    bstar = bstar * pow(10.0, ibexp)

    #  ---- convert to sgp4 units ----
    ndot = ndot / (xpdotp * 1440.0)
    #   ? * minperday
    nddot = nddot / (xpdotp * 1440.0 * 1440)

    #  ---- find standard orbital elements ----
    inclo = inclo * deg2rad
    nodeo = nodeo * deg2rad
    argpo = argpo * deg2rad
    mo = mo * deg2rad

    """
    // ----------------------------------------------------------------
    // find sgp4epoch time of element set
    // remember that sgp4 uses units of days from 0 jan 1950 (sgp4epoch)
    // and minutes from the epoch (time)
    // ----------------------------------------------------------------

    // ---------------- temp fix for years from 1957-2056 -------------------
    // --------- correct fix will occur when year is 4-digit in tle ---------
    """
    if two_digit_year < 57:
        year = two_digit_year + 2000
    else:
        year = two_digit_year + 1900

    mon, day, hr, minute, sec = days2mdhms(year, epochdays)
    sec_whole, sec_fraction = divmod(sec, 1.0)

    epochyr = year
    jdsatepoch = sum(jday(year, mon, day, hr, minute, sec))
    try:
        epoch = datetime(
            year,
            mon,
            day,
            hr,
            minute,
            int(sec_whole),
            int(sec_fraction * 1000000.0 // 1.0),
        )
    except ValueError:
        # Sometimes a TLE says something like "2019 + 366.82137887 days"
        # which would be December 32nd which causes a ValueError.
        year, mon, day, hr, minute, sec = invjday(jdsatepoch)
        epoch = datetime(
            year,
            mon,
            day,
            hr,
            minute,
            int(sec_whole),
            int(sec_fraction * 1000000.0 // 1.0),
        )

    #  ---------------- initialize the orbit at sgp4epoch -------------------
    return (
        satnum_str,
        jdsatepoch - 2433281.5,
        bstar,
        ndot,
        nddot,
        ecco,
        argpo,
        inclo,
        mo,
        no_kozai,
        nodeo,
    )
