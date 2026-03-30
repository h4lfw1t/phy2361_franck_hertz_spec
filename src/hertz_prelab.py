import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "out"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Define the positions of the electrodes
d_ca = 4.0  # Distance from cathode to anode in cm
d_ac = 1.0  # Distance from anode to collector in cm

# Define the potentials
V_c = -10.0  # Potential at the cathode in V
V_a = 0.0  # Potential at the anode in V
V_coll = -1.5  # Potential at the collector in V

# Create the x-axis for the plot
x1 = np.linspace(0, d_ca, 100)
x2 = np.linspace(d_ca, d_ca + d_ac, 50)
x_total = np.concatenate((x1, x2))

# Calculate the potential V(x)
V1 = V_c + (V_a - V_c) * (x1 / d_ca)
V2 = V_a + (V_coll - V_a) * ((x2 - d_ca) / d_ac)
V_total = np.concatenate((V1, V2))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_total, V_total, label="Electrostatic Potential")
plt.axvline(x=0, color="r", linestyle="--", label="Cathode")
plt.axvline(x=d_ca, color="g", linestyle="--", label="Anode")
plt.axvline(x=d_ca + d_ac, color="b", linestyle="--", label="Collector")

# Add labels and title
plt.xlabel("Position (cm)")
plt.ylabel("Electrostatic Potential (V)")
plt.title("Electrostatic Potential between Electrodes")
plt.legend()
plt.grid(True)
plt.savefig(OUT_DIR / "hertz_potential_prelab.png")

# Part (ii): Kinetic Energy of the electron

# Assume the collision happens at the midpoint between cathode and anode
collision_pos = d_ca / 2.0

# Create x-axis for different segments of the electron's path
x_before_collision = np.linspace(0, collision_pos, 50)
x_after_collision = np.linspace(collision_pos, d_ca, 50)
x_anode_to_collector = np.linspace(d_ca, d_ca + d_ac, 50)

# Potential V(x) for each segment
V_before_collision = V_c + (V_a - V_c) * (x_before_collision / d_ca)
V_after_collision = V_c + (V_a - V_c) * (x_after_collision / d_ca)
V_anode_to_collector = V_a + (V_coll - V_a) * (
    (x_anode_to_collector - d_ca) / d_ac
)

# Kinetic Energy KE(x) in eV. KE = - (V(x) - V_c)
KE_before_collision = -(V_before_collision - V_c)

# At the collision point
KE_at_collision = -((V_c + (V_a - V_c) * (collision_pos / d_ca)) - V_c)
KE_after_collision_point = KE_at_collision - 3.0  # Electron loses 3 eV

# KE after collision until anode
KE_after_collision = KE_after_collision_point - (
    V_after_collision - (V_c + (V_a - V_c) * (collision_pos / d_ca))
)

# KE from anode to collector
KE_at_anode = KE_after_collision[-1]
KE_anode_to_collector = KE_at_anode - (V_anode_to_collector - V_a)


# Combine the segments for plotting
x_total_ke = np.concatenate(
    (x_before_collision, x_after_collision, x_anode_to_collector)
)
KE_total = np.concatenate(
    (KE_before_collision, KE_after_collision, KE_anode_to_collector)
)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_total_ke, KE_total, label="Kinetic Energy of Electron")
plt.axvline(x=0, color="r", linestyle="--", label="Cathode")
plt.axvline(x=d_ca, color="g", linestyle="--", label="Anode")
plt.axvline(x=d_ca + d_ac, color="b", linestyle="--", label="Collector")
plt.axvline(x=collision_pos, color="orange", linestyle=":", label="Collision")

# Add labels and title
plt.xlabel("Position (cm)")
plt.ylabel("Kinetic Energy (eV)")
plt.title("Kinetic Energy of an Electron with one Inelastic Collision")
plt.legend()
plt.grid(True)
plt.savefig(OUT_DIR / "hertz_energy_prelab.png")