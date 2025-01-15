import jax.numpy as jnp

from sgpax.earth_gravity import wgs72old, wgs72, wgs84
from sgpax.propagation import sgp4, sgp4init

WGS72OLD = 0
WGS72 = 1
WGS84 = 2
gravity_constants = wgs72old, wgs72, wgs84  # indexed using enum values above
minutes_per_day = 1440.0

class Satrec(object):
    
    def __init__(self):
        raise NotImplementedError
    
    @property
    def no(self):
        return self.n0
    
    @property
    def satnum(self):
        raise NotImplementedError("Return satellite number as string (from TLE)")
    
    @classmethod
    def twoline2rv(cls, line1, line2, whichconst=WGS72):
        raise NotImplementedError
    
    def sgp4init(self, whichconst, opsmode, satnum, epoch, bstar,
                 ndot, nddot, ecco, argpo, inclo, mo, no_kozai, nodeo):
        """
        Initialise SGP4 constants.
        
        Function copied from https://github.com/brandon-rhodes/python-sgp4
        """
        
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

        # TODO: Haven't written any of these yet!
        y, m, d, H, M, S = invjday(whole_jd)
        jan0 = jday(y, 1, 0, 0, 0, 0.0)
        self.epochyr = y % 100
        self.epochdays = whole_jd - jan0 + fraction

        self.classification = 'U'

        sgp4init(whichconst, opsmode, satnum, epoch, bstar, ndot, nddot,
                 ecco, argpo, inclo, mo, no_kozai, nodeo, self)
        
    def sgp4(self, jd, fr):
        tsince = ((jd - self.jdsatepoch) * minutes_per_day +
                  (fr - self.jdsatepochF) * minutes_per_day)
        return self.sgp4_tsince(tsince)
    
    def sgp4_tsince(self, tsince):
        r, v = sgp4(self, tsince)
        return self.error, r, v
    
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
        array = self.array
        if array is None:
            Satrec.array = jnp.array

        results = []
        z = list(zip(jd, fr))
        for jd_i, fr_i in z:
            results.append(self.sgp4(jd_i, fr_i))
        elist, rlist, vlist = zip(*results)

        e = array(elist)
        r = array(rlist)
        v = array(vlist)

        r.shape = v.shape = len(jd), 3
        return e, r, v
