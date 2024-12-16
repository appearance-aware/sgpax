import jax.numpy as jnp

a1 = (ke / n0)**(2/3)

delta1 = 3/2 * k2/a1**2 * (3*jnp.cos(i0)**2 - 1) / (1-e0**2)**(3/2)

a0 = a1 * (1 - 1/3 * delta1 - delta1 **2 - 134/81 * delta1**3)

delta0 = 3/2 * k2/a0**2 * (3*jnp.cos(i0)**2 - 1) / (1-e0**2)**(3/2)

n0_dprime = n0 / (1+delta0)
a0_dprime = a0 / (1-delta0)

s_star = a0_dprime * (1-e0) - s + aE

s_star = 20 / XKMPER + aE

q0subs_star_pfour = ((q0subs_star_pfour) ** (1/4) + s - s_star) ** 4

theta = jnp.cos(i0)
zeta = 1 / (a0_dprime - s)

beta0 = (1 - e0**2)**(1/2)

eta = a0_dprime * e0 * zeta

c2 = q0subs_star_pfour * zeta **4 *n0_dprime * (1-eta**2)**(-7/2) * (a0_dprime * (1 + 3/2 * eta**2 + 4 * e0 * eta + e0 * eta**3) + 3/2 * k2*zeta/(1-eta**2) * (-1/2 + 3/2 * theta**2) * (8+24*eta**2 + 3 * eta**4))

c1 = bstar*c2

c3 = q0subs_star_pfour * zeta**5 * a30 * n0_dprime * aE * jnp.sin(i0) / (k2 * e0)



