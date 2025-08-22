import jax
jax.config.update("jax_enable_x64", True)

from sgpax.earth_gravity import wgs72old, wgs72, wgs84
from sgpax.propagation import sgp4, init_sgp4

from .helper import jday, invjday
from . import io

import equinox as eqx

WGS72OLD = 0
WGS72 = 1
WGS84 = 2

gravity_constants = wgs72old, wgs72, wgs84  # indexed using enum values above
minutes_per_day = 1440.0


class Satrec(eqx.Module):
    """
    Class to store SGP4 constants and perform SGP4 propagation

    Largely follows the API of the python-sgp4 library
    """

    # TODO: Some of these parameters would be better condensed into structs
    # Orbital parameters
    n0: float  # Mean motion at epoch
    i0: float  # Mean inclination at epoch
    e0: float  # Mean eccentricity at epoch
    w0: float  # Mean argument of perigee at epoch
    M0: float  # Mean anomaly at epoch
    raan0: float  # Mean RAAN at epoch
    Bstar: float  # Drag coefficient
    ndot: float  # Time derivative of mean motion
    nddot: float  # Second time derivative of mean motion

    satnum_str: str = eqx.field(static=True)  # Satellite name

    # SGP4 parameters
    jdsatepoch: float  # Julian date of epoch
    jdsatepochF: float  # Fractional
    epochyr: float  # Epoch
    epochdays: float  # Epoch
    classification: str = eqx.field(static=True)  # Classification

    # Constants
    low_altitude: bool
    eps: float
    A30: float
    ke: float
    k2: float
    radiusearthkm: float
    sini0: float

    a0_dp: float
    n0_dp: float

    eta: float
    theta: float
    C1: float
    C2: float
    C3: float
    C4: float
    C5: float
    # Only used for near-earth
    D2: float  # optional?
    D3: float  # optional?
    D4: float  # optional?

    t2_coef: float
    # Only used for near-earth
    t3_coef: float  # optional?
    t4_coef: float  # optional?
    t5_coef: float  # optional?

    dw_coef: float
    dM_coef: float
    raan_coef: float

    M_dot: float
    w_dot: float
    raan_dot: float

    def __init__(
        self,
        whichconst,
        opsmode,
        satnum,
        epoch,
        bstar,
        ndot,
        nddot,
        ecco,
        argpo,
        inclo,
        mo,
        no_kozai,
        nodeo,
    ):
        """
        Initialise SGP4 constants.

        Function API copied from https://github.com/brandon-rhodes/python-sgp4
        """
        if opsmode != "i":
            raise NotImplementedError(
                "Only improved versions of SGP4 are supported by sgpax"
            )

        whichconst = gravity_constants[whichconst]
        whole, fraction = divmod(epoch, 1.0)
        whole_jd = whole + 2433281.5

        # Go out on a limb: if `epoch` has no decimal digits past the 8
        # decimal places stored in a TLE, then assume the user is trying
        # to specify an exact decimal fraction.
        if round(epoch, 8) == epoch:
            fraction = round(fraction, 8)

        self.jdsatepoch = whole_jd
        self.jdsatepochF = fraction

        y, m, d, H, M, S = invjday(whole_jd)
        jan0, _ = jday(y, 1, 0, 0, 0, 0.0)
        self.epochyr = y % 100
        self.epochdays = whole_jd - jan0 + fraction

        self.classification = "U"

        # TODO: Look into properly jitting in the future
        init_sgp4(
            whichconst,
            satnum,
            bstar,
            ndot,
            nddot,
            ecco,
            argpo,
            inclo,
            mo,
            no_kozai,
            nodeo,
            self,
        )

    def sgp4init(
        self,
        whichconst,
        opsmode,
        satnum,
        epoch,
        bstar,
        ndot,
        nddot,
        ecco,
        argpo,
        inclo,
        mo,
        no_kozai,
        nodeo,
    ):
        return self.__init__(
            whichconst,
            opsmode,
            satnum,
            epoch,
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

    @classmethod
    def twoline2rv(cls, line1, line2, whichconst=WGS72):
        return cls(whichconst, "i", *io.twoline2rv(line1, line2))

    @property
    def no(self):
        return self.n0

    @property
    def satnum(self):
        # TODO: Look into alpha5 encoding stuff
        return self.satnum_str

    def sgp4(self, jd, fr):
        tsince = (jd - self.jdsatepoch) * minutes_per_day + (
            fr - self.jdsatepochF
        ) * minutes_per_day
        return self.sgp4_tsince(tsince)

    def sgp4_tsince(self, tsince):
        e, r, v = sgp4(self, tsince)
        return e, r, v

    def sgp4_array(self, jd, fr):
        """Compute positions and velocities for the times in a NumPy array.

        Given NumPy arrays ``jd`` and ``fr`` of the same length that
        supply the whole part and the fractional part of one or more
        Julian dates, return a tuple ``(e, r, v)`` of three vectors:

        * ``e``: nonzero for any dates that produced errors, 0 otherwise.
        * ``r``: position vectors in kilometers.
        * ``v``: velocity vectors in kilometers per second.

        Function copied from https://github.com/brandon-rhodes/python-sgp4
        """
        return jax.vmap(self.sgp4, in_axes=(0, 0))(jd, fr)
