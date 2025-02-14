from collections import namedtuple
import jax.numpy as jnp


def getgravconst(whichconst):
    """
    Get the gravitational constants for a given gravity model
    Returns:
    (tumin, mu, radiusearthkm, xke, j2, j3, j4)
    """
    if whichconst == "wgs72old":
        mu = 398600.79964  #  in km3 / s2
        radiusearthkm = 6378.135  #  km
        xke = 0.0743669161
        tumin = 1.0 / xke
        j2 = 0.001082616
        j3 = -0.00000253881
        j4 = -0.00000165597

        #  ------------ wgs-72 constants ------------
    elif whichconst == "wgs72":
        mu = 398600.8  #  in km3 / s2
        radiusearthkm = 6378.135  #  km
        xke = 60.0 / jnp.sqrt(radiusearthkm * radiusearthkm * radiusearthkm / mu)
        tumin = 1.0 / xke
        j2 = 0.001082616
        j3 = -0.00000253881
        j4 = -0.00000165597

    elif whichconst == "wgs84":
        #  ------------ wgs-84 constants ------------
        mu = 398600.5  #  in km3 / s2
        radiusearthkm = 6378.137  #  km
        xke = 60.0 / jnp.sqrt(radiusearthkm * radiusearthkm * radiusearthkm / mu)
        tumin = 1.0 / xke
        j2 = 0.00108262998905
        j3 = -0.00000253215306
        j4 = -0.00000161098761
    else:
        raise ValueError("Invalid gravity model: " + whichconst)

    return tumin, mu, radiusearthkm, xke, j2, j3, j4


EarthGravity = namedtuple(
    "EarthGravity",
    "tumin mu radiusearthkm xke j2 j3 j4",
)

wgs72old = EarthGravity(*getgravconst("wgs72old"))
wgs72 = EarthGravity(*getgravconst("wgs72"))
wgs84 = EarthGravity(*getgravconst("wgs84"))
