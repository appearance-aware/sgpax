from collections import namedtuple
import jax.numpy as jnp


"""
/* -----------------------------------------------------------------------------
*
*                           function getgravconst
*
*  this function gets constants for the propagator. note that mu is identified to
*    facilitiate comparisons with newer models. the common useage is wgs72.
*
*  author        : david vallado                  719-573-2600   21 jul 2006
*
*  inputs        :
*    whichconst  - which set of constants to use  wgs72old, wgs72, wgs84
*
*  outputs       :
*    tumin       - minutes in one time unit
*    mu          - earth gravitational parameter
*    radiusearthkm - radius of the earth in km
*    xke         - reciprocal of tumin
*    j2, j3, j4  - un-normalized zonal harmonic values
*    j3oj2       - j3 divided by j2
*
*  locals        :
*
*  coupling      :
*    none
*
*  references    :
*    norad spacetrack report #3
*    vallado, crawford, hujsak, kelso  2006
  --------------------------------------------------------------------------- */
"""


def getgravconst(whichconst):
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
