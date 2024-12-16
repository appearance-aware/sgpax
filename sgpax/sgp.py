import jax.numpy as jnp


# TODO: Initial orbital params from TLE (or equivalent)
n0 = 1              # Mean motion at epoch
i0 = 1              # Mean inclination at epoch
e0 = 1              # Mean eccentricity at epoch
w0 = 1              # Mean argument of perigee at epoch
M0 = 1              # Mean anomaly at epoch
Omega0 = 1          # Mean RAAN at epoch

t = 1               # Time now
t0 = 1              # Time of epoch
dt = t - t0         # Time since epoch

# TODO: Constants
aE = 1.0            # Equatorial radius of Earth (distance units are in Earth radii)
J2 = 1              # Second graviational zonal harmonic of Earth
J3 = 1              # Third graviational zonal harmonic of Earth
J4 = 1              # Fourth graviational zonal harmonic of Earth
ke = 1              # sqrt(G*M) where M is mass of the Earth

# SGP4 parameters
Bstar = 0                   # SGP4 type drag coefficient (TODO: assume 0? Check positive?)
XKMPER = 6378.135           # kilometers/Earth radii
s = aE + 78 / XKMPER        # Parameter for the SGP4 density function
q0 = aE + 120 / XKMPER      
qoms2t = (q0 - s)**4                  

# Standard constants/variables
k2 = 0.5 * J2 * aE**2
k4 = -3/8 * J4 * aE**4
A30 = - J3 * aE**3

# Compute original mean motion and semimajor axis from input elements
a1 = (ke / n0)**(2/3)
delta1 = 3/2 * k2/(a1**2) * (3*jnp.cos(i0)**2 - 1) / (1 - e0**2)**(3/2)
a0 = a1 * (1 - delta1/3 - delta1**2 - 134/81 * delta1**3)
delta0 = 3/2 * k2/(a0**2) * (3*jnp.cos(i0)**2 - 1) / (1 - e0**2)**(3/2)
n0_dp = n0 / (1 + delta0)
a0_dp = a0 / (1 - delta0)

# For perigee less than 220km, the isimp flag is set and
# the equations are truncated to linear variation in sqrt(a)
# and quadratic variation in mean anomaly. Also, the c3 term,
# the delta omega term, and the delta m term are dropped.
isimp = 0
perigee = (a0_dp * (1 - e0) - aE) * XKMPER
if perigee < 220.0:
    isimp = 1
# if a0_dp * (1 - e0) / aE < (220.0 / XKMPER + aE):
# TODO: The fortran code uses the above, which is not the same
#       if aE is not 1.0. Check this later
    
# Alter value of s, (q0 - s)^4 for different perigees
qoms24 = qoms2t
s4 = s                                          # Standard notation for s star
if perigee <= 156.0:
    s4 = perigee - 78.0
    if perigee <= 98.0:
        s4 = 20.0
    qoms24 = ((120.0 - s4) * aE / XKMPER)**4
    s4 = s4 / XKMPER + aE

# Calculate SGP4 constants
theta = jnp.cos(i0)
xi = 1 / (a0_dp - s4)
beta0 = (1 - e0**2)**(1/2)
eta = a0_dp * e0 * xi

C2 = qoms24 * xi**4 * n0_dp * (1 - eta**2)**(-7/2) * (
    a0_dp * (1 + 3/2 * eta**2 + 4 * e0 * eta + e0 * eta**3) + 
    3/2 * k2 * xi / (1 - eta**2) * (-1/2 + 3/2 * theta**2) * 
    (8 + 24 * eta**2 + 3 * eta**4)
)
C1 = Bstar*C2
C3 = qoms24 * xi**5 * A30 * n0_dp * aE * jnp.sin(i0) / (k2 * e0)
C4 = 2.0 * n0_dp * qoms24 * xi**4 * beta0**2 * (1 - eta)**(-7/2) *(
    (2 * eta * (1 + e0 * eta) + 0.5 * e0 + 0.5 * eta**3) -
    2 * k2 * xi / (a0_dp * (1 - eta**2)) * (
        3 * (1 - 3*theta**2) * (1 + 3/2 * eta**2 - 2* e0 * eta - 0.5 * e0 * eta**3) +
        3/4 * (1 - theta**2) * (2*eta**2 - e0*eta - e0*eta**3) * jnp.cos(2 * w0)
    )
)
C5 = 2 * qoms24 * xi**4 * a0_dp * beta0**2 (1 - eta**2)**(-7/2) * (
    1 + 11/4 * eta * (eta + e0) + e0 * eta**3
)

D2 = 4 * a0_dp * xi * C1**2
D3 = 4/3 * a0_dp * xi**2 * (17 * a0_dp + s4) * C1**3
D4 = 2/3 * a0_dp * xi**3 * (221 * a0_dp + 31 * s4) * C1**4

# These terms account for secular effects of atmospheric drag and graviation
M_DF = M0 + (
    1 + 3 * k2 * (-1 + 3 * theta**2) / (2 * a0_dp**2 * beta0**3) + 
    3 * k2**2 * (13 - 78 * theta**2 + 137 * theta**4) / (16 * a0_dp**4 * beta0**7)
) * n0_dp * dt

w_DF = w0 + (
    -3 * k2 (1 - 5 * theta**2) / (2 * a0_dp**2 * beta0**4) + 
    3 * k2**2 * (7 - 114 * theta**2 + 395 * theta**4) / (16 * a0_dp**4 * beta0**8) +
    5 * k4 * (3 - 36 * theta**2 + 49 * theta**4) / (4 * a0_dp**4 * beta0**8)
) * n0_dp * dt

Omega_DF = Omega0 + (
    -3 * k2 * theta / (a0_dp**2 * beta0**4) + 
    3 * k2**2 * (4 * theta - 19 * theta**3) / (2 * a0_dp**4 * beta0**8) + 
    5 * k4 * theta * (3 - 7 * theta**2) / (2 * a0_dp**4 * beta0**8)
) * n0_dp * dt

delta_w = Bstar * C3 * jnp.cos(w0) * dt

delta_M = -2/3 * qoms24 * Bstar * xi**4 * aE/(e0 * eta) * (
    (1 + eta * jnp.cos(M_DF))**3 - (1 + eta * jnp.cos(M0))**3
)

Mp = M_DF + delta_w + delta_M

w = w_DF - delta_w - delta_M

Omega = Omega_DF - 21/2 * n0_dp * k2 * theta / (a0_dp**2 * beta0**2) * C1 * dt**2

e = e0 - Bstar * C4 * dt - Bstar * C5 * (jnp.sin(Mp) - jnp.sin(M0))

a = a0_dp * (1 - C1 * dt - D2 * dt**2 - D3 * dt**3 - D4 * dt**4)**2

L = Mp + w + Omega + n0_dp * (
    3/2 * C1 * dt**2 + 
    (D2 + 2 * C1**2) * dt**3 + 
    1/4 * (3 * D3 + 12 * C1 * D2 + 10 * C1**3) * dt**4 +
    1/5 * (3 * D4 + 12 * C1 * D3 + 6 * D2**2 + 30 * C1**2 * D2 + 15 * C1**4) * dt **5
)

beta = jnp.sqrt(1 - e**2)
n = ke / a**(3/2)


# TODO: Apparently we don't need all of these terms if the perigee height (at epoch) 
# is less than 220km. 
# See pytorch implementation: https://github.com/esa/dSGP4/blob/master/dsgp4/sgp4init.py#L187
# or matlab version: https://au.mathworks.com/matlabcentral/fileexchange/62013-sgp4
