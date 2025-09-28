import scienceplots
import matplotlib.pyplot as plt

plt.style.use(["science", "ieee"])

from test_sgp4 import test_compare_against_python_sgp4

# Set up the plot
_, axs = plt.subplots(nrows=2, ncols=1)

for sat in ["Hubble", "Sentinel-6", "CloudSat"]:
    
    # Compute errors over time
    test_results = test_compare_against_python_sgp4(
        satname=sat, verbose=False, nevalpoints=201
    )

    hours = test_results["time_hours"]
    r_errors = test_results["pos_errors"]
    v_errors = test_results["vel_errors"]

    # Make the plot
    axs[0].plot(hours, r_errors, label=sat)
    axs[1].plot(hours, v_errors, label=sat)

axs[1].set_xlabel("Time (hours from epoch)")
axs[0].set_ylabel("Position error (m)")
axs[1].set_ylabel("Velocity error (m/s)")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
plt.legend()

plt.tight_layout()
plt.savefig("test_error.pdf")
