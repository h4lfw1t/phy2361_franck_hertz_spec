# Configuration
DIFFRACTION_GRATING = 600  # lines/mm
MEASUREMENT_UNCERTAINTY = 0.165  # degrees (0.33/2)
SODIUM_WAVELENGTH = 533  # nm (calculated calibration)
SODIUM_DOUBLET_ANGLE = 313.66  # degrees
STARTING_ANGLE = 295  # degrees (reference for all measurements)

# Calculate offset from sodium doublet for each gas
SODIUM_OFFSET = SODIUM_DOUBLET_ANGLE - STARTING_ANGLE  # 18.66 degrees

# Spectral data organized by gas
SPECTRAL_DATA = {
    "sodium": {
        "calibration": True,
        "wavelength": 533,  # nm
        "doublet_angle": 313.66,
        "offset_from_start": 18.66,
        "lines": []
    },
    "hydrogen": {
        "lines": {
            "magenta": {"angle": 293, "offset": -2},
            "blue": {"angle": 277.5, "offset": -17.5},
            "cyan": {"angle": 276.6, "offset": -18.4},
            "red": {"angle": 269, "offset": -26}
        }
    },
    "helium": {
        "lines": {
            "purple": {"angle": 277, "offset": -18},
            "cyan": {"angle": 276, "offset": -19},
            "green": {"angle": 275.3, "offset": -19.7},
            "yellow": {"angle": 275, "offset": -20},
            "orange": {"angle": 271.6, "offset": -23.4},
            "red": {"angle": 268.6, "offset": -26.4}
        }
    },
    "nitrogen": {
        "lines": {
            "distinct": {"angle": 293, "offset": -2},
            "blue_1": {"angle": 278.6, "offset": -16.4},
            "blue_2": {"angle": 276, "offset": -19},
            "broad_bands": {"angle_range": (270, 276), "note": "Broader with wider slit"}
        }
    },
    "krypton": {
        "lines": {
            "blue_1_2": {"angle": 277.6, "offset": -17.4, "note": "Measurement uncertainty"},
            "blue_3": {"angle": 277.3, "offset": -17.7},
            "blue_4_5": {"angle": 277, "offset": -18, "note": "Measurement uncertainty"},
            "green": {"angle": 273, "offset": -22},
            "orange": {"angle": 271.6, "offset": -23.4},
            "red": {"angle": 269, "offset": -26}
        }
    }
}