import jax.numpy as jnp

def sgp4init(whichconst, opsmode, satnum, epoch, Bstar, ndot, nddot,
                 e0, w0, i0, M0, n0, raan0, satrec):
    """TODO: Add docs later"""
    
    satrec.operationmode = opsmode
    satrec.satnum_str = str(satnum)
    satrec.classification = 'U'
    
    # Initial params
    satrec.n0 = n0              # Mean motion at epoch
    satrec.i0 = i0              # Mean inclination at epoch
    satrec.e0 = e0              # Mean eccentricity at epoch
    satrec.w0 = w0              # Mean argument of perigee at epoch
    satrec.M0 = M0              # Mean anomaly at epoch
    satrec.raan0 = raan0        # Mean RAAN at epoch
    satrec.Bstar = Bstar        # Drag coefficient
    satrec.ndot = ndot          # Time derivative of mean motion
    satrec.nddot = nddot        # Second time derivative of mean motion
    
    # SGP4 and gravitational parameters
    # aE:   Equatorial radius of Earth (distance units are in Earth radii)
    # J2:   Second graviational zonal harmonic of Earth
    # J3:   Third graviational zonal harmonic of Earth
    # J4:   Fourth graviational zonal harmonic of Earth
    # ke:   sqrt(G*M) where M is mass of the Earth
    # TODO: Where do tumin and mu get used?!
    (tumin, mu, radiusearthkm, ke, J2, J3, J4) = whichconst
    aE = 1.0
    
    s = aE + 78 / radiusearthkm     # Parameter for the SGP4 density function
    qoms2t = ((120.0 - 78.0) / radiusearthkm)**4
    
    k2 = 0.5 * J2 * aE**2
    k4 = -3/8 * J4 * aE**4
    A30 = -J3 * aE**3
    
    # Useful
    sini0 = jnp.sin(i0)
    cosi0 = jnp.cos(i0)
    
    # Compute original mean motion and semimajor axis from input elements
    a1 = (ke / n0)**(2/3)
    d1 = 3/2 * k2 * (3*cosi0**2 - 1) / (1 - e0**2)**(3/2)
    del1 = d1 / a1**2
    a0 = a1 * (1 - del1/3 - del1**2 - 134/81 * del1**3)
    del0 = d1 / a0**2
    n0_dp = n0 / (1 + del0)
    a0_dp = (ke / n0_dp)**(2/3)         # Same as a0 / (1 - del0)
    
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
    perigee = (a0_dp * (1 - e0) - aE) * radiusearthkm
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
        qoms24 = ((120.0 - s4) * aE / radiusearthkm)**4
        s4 = s4 / radiusearthkm + aE

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
        
    # Store all the relevant information for propagation in the struct
    satrec.low_altitude = low_altitude
    satrec.eps = eps
    
    satrec.A30 = A30
    satrec.ke = ke
    satrec.k2 = k2
    satrec.radiusearthkm = radiusearthkm
    
    satrec.sini0 = sini0
    
    satrec.a0_dp = a0_dp
    satrec.n0_dp = n0_dp
    
    satrec.eta = eta
    satrec.theta = theta
    
    satrec.C1 = C1
    satrec.C4 = C4
    satrec.C5 = C5
    satrec.D2 = D2
    satrec.D3 = D3
    satrec.D4 = D4
    
    satrec.t2_coef = t2_coef
    satrec.t3_coef = t3_coef
    satrec.t4_coef = t4_coef
    satrec.t5_coef = t5_coef
    
    satrec.dw_coef = dw_coef
    satrec.dM_ceof = dM_coef
    satrec.raan_ceof = raan_coef
    
    satrec.M_dot = M_dot
    satrec.w_dot = w_dot
    satrec.raan_dot = raan_dot
        
        
def sgp4(satrec, tsince, whichconst=None):
    """TODO: Write doces. tsince = t - t0 is time since epoch (to be calculated)"""
        
        
    ## SGP4 - Integrate through time

    # Integrate through time
    M_DF = satrec.M0 + satrec.M_dot * tsince
    w_DF = satrec.w0 + satrec.w_dot * tsince
    raan_DF = satrec.raan0 + satrec.raan_dot * tsince

    # Set up intermediate variables
    w_temp = w_DF
    e_temp = satrec.Bstar * satrec.C4 * tsince
    a_temp = 1 - satrec.C1 * tsince
    l_temp = satrec.t2_coef * tsince**2

    # For (perigee) altitude higher than 220km
    if not satrec.low_altitude:
        
        dw = satrec.dw_coef * tsince
        dM = satrec.dM_coef * (
            (1 + satrec.eta * jnp.cos(M_DF))**3 - 
            (1 + satrec.eta * jnp.cos(satrec.M0))**3
        )

        Mp = M_DF + dw + dM
        w_temp = w_temp - dw - dM
        
        e_temp = e_temp + satrec.Bstar * satrec.C5 * (jnp.sin(Mp) - jnp.sin(satrec.M0))
        a_temp = a_temp - satrec.D2 * tsince**2 - satrec.D3 * tsince**3 - satrec.D4 * tsince**4
        l_temp = l_temp + satrec.t3_coef * tsince**3 + satrec.t4_coef * tsince**4 + satrec.t5_coef * tsince**5
        
    # Compute values of orbital elements at time delta tsince
    e = satrec.e0 - e_temp
    a = satrec.a0_dp * a_temp**2
    i = satrec.i0
    n = satrec.ke / a**(3/2)
    w = w_temp
    raan = raan_DF + satrec.raan_coef * tsince**2
    Mm = Mp + (satrec.n0_dp * l_temp)          # Mean anomaly
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
        
    # Store the singly averaged mean elements in the struct
    # These are: (eccentricity, semi-major axis, inclination, 
    #             RAAN, arg. perigee, mean anomaly, mean motion)
    satrec.am = a
    satrec.em = e
    satrec.im = i
    satrec.Om = raan
    satrec.om = w
    satrec.mm = Mm
    satrec.nm = n


    # ----------- Add the long-period periodic terms -----------

    beta = jnp.sqrt(1 - e**2)
    ax_N = e * jnp.cos(w)
    ay_NL = satrec.A30 * satrec.sini0 / (4 * satrec.k2 * a * beta**2)
    ay_N = e * jnp.sin(w) + ay_NL

    # TODO: L_L might be taken as zero if not deep space. Unclear?
    if jnp.fabs(1 + satrec.theta) > satrec.eps:
        L_L = 0.5 * ay_NL * ax_N * (3 + 5*satrec.theta) / (1 + satrec.theta)
    else:
        L_L = 0.5 * ay_NL * ax_N * (3 + 5*satrec.theta) / satrec.eps
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
    rdot = satrec.ke * jnp.sqrt(a) / r * esinE
    rfdot = satrec.ke * jnp.sqrt(pL) / r
    temp = esinE / (1.0 + jnp.sqrt(1 - eL2))
    cosu = a / r * (cosEw - ax_N + ay_N * temp)
    sinu = a / r * (sinEw - ay_N - ax_N * temp)
    u = jnp.atan2(sinu, cosu)

    sin2u = 2 * sinu * cosu
    cos2u = 1 - 2 * sinu**2

    # ------------- Updates for short period periodics -----------

    delta_r = satrec.k2 / (2 * pL) * (1 - satrec.theta**2) * cos2u
    delta_u = -satrec.k2 / (4 * pL**2) * (7*satrec.theta**2 - 1) * sin2u
    delta_raan = 3 * satrec.k2 * satrec.theta / (2 * pL**2) * sin2u
    delta_i = 3 * satrec.k2 * satrec.theta / (2 * pL**2) * satrec.sini0 * cos2u
    delta_rdot = -satrec.k2 * n / pL * (1 - satrec.theta**2) * sin2u
    delta_rfdot = satrec.k2 * n / pL * ((1 - satrec.theta**2) * cos2u - 3/2 * (1 - 3*satrec.theta**2))

    r_k = r * (1 - 3/2 * satrec.k2 * jnp.sqrt(1 - eL2) / pL**2 * (3*satrec.theta**2 - 1)) * delta_r
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
    r_eci = r_k * uvec * satrec.radiusearthkm
    v_eci = rdot_k * uvec + rfdot_k * vvec

    # Check for decaying satellites
    if r_k < 1.0:
        raise ValueError("Satellite radius has decayed and crashed.")
    
    return r_eci, v_eci
