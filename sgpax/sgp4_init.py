import jax.numpy as jnp


# TODO: Initial orbital params from TLE (or equivalent)
n0 = 0.123          # Mean motion at epoch
i0 = 1              # Mean inclination at epoch
e0 = 0.1            # Mean eccentricity at epoch
w0 = 1              # Mean argument of perigee at epoch
M0 = 1              # Mean anomaly at epoch
raan0 = 1           # Mean RAAN at epoch

Bstar = 0           # Drag coefficient
ndot = 0            # Time derivative of mean motion
nddot = 0           # Second time derivative of mean motion


###########################################################################

# TODO: Constants for the gravitational centre (eg: Earth)
aE = 1.0            # Equatorial radius of Earth (distance units are in Earth radii)
J2 = 1              # Second graviational zonal harmonic of Earth
J3 = 1              # Third graviational zonal harmonic of Earth
J4 = 1              # Fourth graviational zonal harmonic of Earth
ke = 1              # sqrt(G*M) where M is mass of the Earth

# SGP4 parameters
radisuearthkm = 6378.135        # kilometers/Earth radii
s = aE + 78 / radisuearthkm     # Parameter for the SGP4 density function
qoms2t = ((120.0 - 78.0) / radisuearthkm)**4


###########################################################################

## Initialisation of coefficients from TLE (shared with SDP4 if we implement it)

# Useful
sini0 = jnp.sin(i0)
cosi0 = jnp.cos(i0)

# Standard constants/variables
k2 = 0.5 * J2 * aE**2
k4 = -3/8 * J4 * aE**4
A30 = -J3 * aE**3

# Compute original mean motion and semimajor axis from input elements
a1 = (ke / n0)**(2/3)
d1 = 3/2 * k2 * (3*cosi0**2 - 1) / (1 - e0**2)**(3/2)
del1 = d1 / a1**2
a0 = a1 * (1 - del1/3 - del1**2 - 134/81 * del1**3)
del0 = d1 / a0**2
n0_dp = n0 / (1 + del0)
a0_dp = a0_dp = (ke / n0_dp)**(2/3)         # Same as a0 / (1 - del0)

# TODO: We should store the resulting n0_dp, a0_dp as the initial
#       mean motion and semi-major axis in a struct for later computation

# Check that the satellite is actually in a valid orbit
if (e0 > 1.0) or n0_dp < 0.0:
    raise ValueError("Satellite is not in a valid orbit")


###########################################################################

## SGP4 initialisation

# sgp4fix divisor for divide by zero check on inclination
# the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
# 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
eps = 1.5e-12

# Treat low altitudes differently
low_altitude = False                       # TODO: Store this flag in the satellite struct
perigee = (a0_dp * (1 - e0) - aE) * radisuearthkm
if perigee < 220.0:
    low_altitude = True
    
# Alter value of s, (q0 - s)^4 for different perigees
# Standard notation for s star seems to be s4
qoms24 = qoms2t
s4 = s
if perigee <= 156.0:
    s4 = perigee - 78.0
    if perigee <= 98.0:
        s4 = 20.0
    qoms24 = ((120.0 - s4) * aE / radisuearthkm)**4
    s4 = s4 / radisuearthkm + aE

# Calculate SGP4 constants/coefficients
theta = cosi0
xi = 1 / (a0_dp - s4)
beta0 = (1 - e0**2)**(1/2)
eta = a0_dp * e0 * xi

C2 = qoms24 * xi**4 * n0_dp * (1 - eta**2)**(-7/2) * (
    a0_dp * (1 + 3/2 * eta**2 + 4 * e0 * eta + e0 * eta**3) + 
    3/2 * k2 * xi / (1 - eta**2) * (-1/2 + 3/2 * theta**2) * 
    (8 + 24 * eta**2 + 3 * eta**4)
)

C1 = Bstar*C2

C3 = 0.0
if e0 > 1e-4:
    C3 = qoms24 * xi**5 * A30 * n0_dp * aE * sini0 / (k2 * e0)
    
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

# These terms account for secular effects of atmospheric drag and graviation
M_dot = (
    1 + 3 * k2 * (-1 + 3 * theta**2) / (2 * a0_dp**2 * beta0**3) + 
    3 * k2**2 * (13 - 78 * theta**2 + 137 * theta**4) / (16 * a0_dp**4 * beta0**7)
) * n0_dp

w_dot = (
    -3 * k2 (1 - 5 * theta**2) / (2 * a0_dp**2 * beta0**4) + 
    3 * k2**2 * (7 - 114 * theta**2 + 395 * theta**4) / (16 * a0_dp**4 * beta0**8) +
    5 * k4 * (3 - 36 * theta**2 + 49 * theta**4) / (4 * a0_dp**4 * beta0**8)
) * n0_dp

raan_dot = (
    -3 * k2 * theta / (a0_dp**2 * beta0**4) + 
    3 * k2**2 * (4 * theta - 19 * theta**3) / (2 * a0_dp**4 * beta0**8) + 
    5 * k4 * theta * (3 - 7 * theta**2) / (2 * a0_dp**4 * beta0**8)
) * n0_dp

# Pre-compute a bunch of coefficients
dw_coef = Bstar * C3 * jnp.cos(w0)

dM_coef = 0.0
if e0 > 1e-4:
    dM_coef = -2/3 * qoms24 * Bstar * xi**4 * aE/(e0 * eta)

raan_coef = -21/2 * n0_dp * k2 * theta / (a0_dp**2 * beta0**2) * C1

t2_coef = 3/2 * C1

# Handles dividing by zero if inclination is 180 deg
if (1 + theta) > eps:
    LL_coef = A30 * sini0 / (8 * k2) * (3 + 5*theta) / (1 + theta)
else:
    LL_coef = A30 * sini0 / (8 * k2) * (3 + 5*theta) / eps
    
ay_coef = A30 * sini0 / (4 * k2)

# Special variable if not in deep space (perigee < 220km)
if not low_altitude:
    
    D2 = 4 * a0_dp * xi * C1**2
    D3 = 4/3 * a0_dp * xi**2 * (17 * a0_dp + s4) * C1**3
    D4 = 2/3 * a0_dp * xi**3 * (221 * a0_dp + 31 * s4) * C1**4
    
    t3_coef = D2 + 2 * C1**2
    t4_coef = 1/4 * (3 * D3 + 12 * C1 * D2 + 10 * C1**3)
    t5_coef = 1/5 * (3 * D4 + 12 * C1 * D3 + 6 * D2**2 + 30 * C1**2 * D2 + 15 * C1**4)


###########################################################################

# TODO: Get the time correct

t = 1               # Time now
t0 = 0              # Time of epoch
dt = t - t0         # Time since epoch


###########################################################################

## SGP4 - Integrate through time

# Integrate through time
M_DF = M0 + M_dot * dt
w_DF = w0 + w_dot * dt
raan_DF = raan0 + raan_dot * dt

# Set up intermediate variables
w_temp = w_DF
e_temp = Bstar * C4 * dt
a_temp = 1 - C1 * dt
l_temp = t2_coef * dt**2

# For (perigee) altitude higher than 220km
if not low_altitude:
    
    dw = dw_coef * dt
    dM = dM_coef * (
        (1 + eta * jnp.cos(M_DF))**3 - (1 + eta * jnp.cos(M0))**3
    )

    Mp = M_DF + dw + dM
    w_temp = w_temp - dw - dM
    
    e_temp = e_temp + Bstar * C5 * (jnp.sin(Mp) - jnp.sin(M0))
    a_temp = a_temp - D2 * dt**2 - D3 * dt**3 - D4 * dt**4
    l_temp = l_temp + t3_coef * dt**3 + t4_coef * dt**4 + t5_coef * dt**5
    
# Compute values of orbital elements at time delta dt
e = e0 - e_temp
a = a0_dp * a_temp**2
i = i0
n = ke / a**(3/2)
w = w_temp
raan = raan_DF + raan_coef * dt**2
Mm = Mp + (n0_dp * l_temp)          # Mean anomaly
L = Mm + w + raan

# Angle wrapping
twopi = jnp.pi
w = jnp.mod(w, twopi)
raan = jnp.mod(raan, twopi)
Mm = jnp.mod(Mm, twopi)
L = jnp.mod(L, twopi)

# Check for error in eccentricity and fix for numerical precision
if (e >= 1.0) or (e < -1e-3):
    raise ValueError("Orbit eccentricity outside of valid bounds.")
elif (e < 1e-6):
    e = 1e-6
    
# TODO: Store the singly averaged mean elements in the struct?
# These are: e, a, i, raan, w, Mm, n
# (eccentricity, semi-major axis, inclination, RAAN, arg. perigee, mean anomaly, mean motion)
# satrec.am = am;
# satrec.em = em;
# satrec.im = inclm;
# satrec.Om = nodem;
# satrec.om = argpm;
# satrec.mm = mm;
# satrec.nm = nm;

# ----------- Add the long-period periodic terms -----------

beta = jnp.sqrt(1 - e**2)
ax_N = e * jnp.cos(w)
ay_NL = A30 * sini0 / (4 * k2 * a * beta**2)
ay_N = e * jnp.sin(w) + ay_NL

# TODO: L_L might be taken as zero if not deep space. Unclear?
if jnp.fabs(1 + theta) > eps:
    L_L = 0.5 * ay_NL * ax_N * (3 + 5*theta) / (1 + theta)
else:
    L_L = 0.5 * ay_NL * ax_N * (3 + 5*theta) / eps
L_T = L + L_L

# ----------- Solve Kepler's equation for (E + w) -----------

U = jnp.mod(L_T - raan, twopi)
Ew1 = U
temp = 9999.9
k_iter = 1

while (jnp.fabs(temp) >= 1.0e-12) and (k_iter <= 10):
    
    # Get the update delta on (E + w)
    sinEw1 = jnp.sin(Ew1)
    cosEw1 = jnp.cos(Ew1)
    denom = 1.0 - ay_N * sinEw1 - ax_N * cosEw1
    num = U - ay_N * cosEw1 + ax_N * sinEw1 - Ew1
    temp = num / denom
    
    # Regulate update so it's not too large
    if jnp.fabs(temp) >= 0.95:
        temp = jnp.sign(temp) * 0.95
    
    # Update estimate
    Ew1 = Ew1 + temp
    k_iter = k_iter + 1
    
E_plus_w = Ew1
cosEw = jnp.cos(E_plus_w)
sinEw = jnp.sin(E_plus_w)

# ------------- Short period preliminary quantities -----------

ecosE = ax_N * cosEw + ay_N * sinEw
esinE = ax_N * sinEw - ay_N * cosEw

eL2 = ax_N**2 + ay_N**2
pL = a * (1.0 - eL2)
if pL < 0.0:
    raise ValueError("Value out of bounds (need a better error message)")

r = a * (1.0 - ecosE)
rdot = ke * jnp.sqrt(a) / r * esinE
rfdot = ke * jnp.sqrt(pL) / r
temp = esinE / (1.0 + jnp.sqrt(1 - eL2))
cosu = a / r * (cosEw - ax_N + ay_N * temp)
sinu = a / r * (sinEw - ay_N - ax_N * temp)
u = jnp.atan2(sinu, cosu)

sin2u = 2 * sinu * cosu
cos2u = 1 - 2 * sinu**2

# ------------- Updates for short period periodics -----------

delta_r = k2 / (2 * pL) * (1 - theta**2) * cos2u
delta_u = -k2 / (4 * pL**2) * (7*theta**2 - 1) * sin2u
delta_raan = 3 * k2 * theta / (2 * pL**2) * sin2u
delta_i = 3 * k2 * theta / (2 * pL**2) * sini0 * cos2u
delta_rdot = -k2 * n / pL * (1 - theta**2) * sin2u
delta_rfdot = k2 * n / pL * ((1 - theta**2) * cos2u - 3/2 * (1 - 3*theta**2))

r_k = r * (1 - 3/2 * k2 * jnp.sqrt(1 - eL2) / pL**2 * (3*theta**2 - 1)) * delta_r
u_k = u + delta_u
raan_k = raan + delta_raan
i_k = i + delta_i
rdot_k = rdot + delta_rdot
rfdot_k = rfdot + delta_rfdot

# ------------- Orientation vectors -----------

sinuk = jnp.sin(u_k)
cosuk = jnp.cos(u_k)
sinik = jnp.sin(i_k)
cosik = jnp.cos(i_k)
sin_raan_k = jnp.sin(raan_k)
cos_raan_k = jnp.cos(raan_k)

ux = sinuk * (-sin_raan_k * cosik) + cosuk * (cos_raan_k)
uy = sinuk * (cos_raan_k * cosik) + cosuk * (sin_raan_k)
uz = sinuk * (sinik)

vx = cosuk * (-sin_raan_k * cosik) - sinuk * (cos_raan_k)
vy = cosuk * (cos_raan_k * cosik) - sinuk * (sin_raan_k)
vz = cosuk * (sinik)

uvec = jnp.array([ux, uy, uz])
vvec = jnp.array([vx, vy, vz])

# ------------- Position and velocity (in km and km/sec) -------------

# TODO: Check scaling/units here, might need to multiply vel by (vkmpersec / ke)
r_eci = r_k * uvec * radisuearthkm
v_eci = rdot_k * uvec + rfdot_k * vvec

# Check for decaying satellites
if r_k < 1.0:
    raise ValueError("Satellite radius has decayed and crashed.")
