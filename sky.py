import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def deg2rad(deg):
    return np.radians(deg)

def plot_ellipse(ax, l, b, l_err, b_err, color, label):
    l = (l + 180) % 360 - 180
    l = -l  # Flip left to right
    l_rad, b_rad = deg2rad(l), deg2rad(b)
    ax.scatter(l_rad, b_rad, color=color, s=40, edgecolors='black', zorder=5, label=label)
    ellipse = mpatches.Ellipse(
        (l_rad, b_rad), deg2rad(2 * l_err), deg2rad(2 * b_err),
        edgecolor=color, facecolor=color, alpha=0.3, lw=1, zorder=4
    )
    ax.add_patch(ellipse)

points = {
    'R×H²/S²': {'coords': (168.431, -0.702), 'l_err': 104.449, 'b_err': 33.116, 'color': 'red'},
    'Nil': {'coords': (132.048, 0.632), 'l_err': 87.062, 'b_err': 33.929, 'color': 'green'},
    'Solv': {'coords': (151.639, 21.053), 'l_err': 103.349, 'b_err': 66.260, 'color': 'blue'},
    'B-I_CS': {'coords': (290.3, 9), 'l_err': -14.15, 'b_err': 17, 'color': 'yellow'},
    'B-I_DW': {'coords': (214, -41.6), 'l_err': -6, 'b_err': 7.15, 'color': 'brown'},
    'B-I_LVMF': {'coords': (222, -46), 'l_err': -5, 'b_err': 6.6, 'color': 'pink'},
    'B-I_MF': {'coords': (289.9, 10), 'l_err': -12.9, 'b_err': -1, 'color': 'violet'}
}

fundamental_points = {
    'CMB dipole': (264.02, 48.25),
    'CMB quadrupole': (238.50, 13.40),
    'CMB octupole': (239.00, 25.70),
    'CMB parity asymmetry': (279.73, 45.82),
    'CMB kinematic dipole': (264.00, 42.00),
    'SN1a dipole': (297, 3),
    'SN-Q (HC)': (316.08, 4.53),
    'SN-Q (DF)': (327.55, 51.01),
    'SNe (CMB frame)': (242.00, 59.00),
    'SNe (Heliocentric frame)': (252.00, 65.00),
    'TGSS radio galaxies': (243.00, 45.00),
    'NVSS radio galaxies': (253.00, 27.00),
    'Handedness of spiral galaxies': (232.00, 158.50),
    'Anisotropy of cosmic acceleration': (247.50, 23.40),
    'Distribution of fine structure constant': (331.00, -104.00),
    'Pantheon': (286.93, 27.02),
    'Pantheon+ (Ωₘ, 90°) (Local matter underdensity)': (308.40, -18.20),
    'Pantheon+ (H₀, 90°)': (313.40, -16.80),
    'Bulk flow-I': (297.00, -4.00),
    'Bulk flow-II': (298.00, -7.00),  
    'Large-scale velocity flows': (282.00, 84.00),
    'Galaxy cluster-I': (303.00, -27.00),
    'Galaxy cluster-II': (280.00, -15.00),  
    'GRB': (82.97, -15.09),
    'Quasar - I': (288.92, 6.10),
    'Quasar - II': (238.00, 28.00),
    'Quasar flux': (201.50, -29.37),
    'Polarization of QSOs': (267.00, 69.00),
    'Soft X-ray effects': (118.00, 7.00),
}

fundamental_colors = [
    'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'lime',
    'teal', 'yellow', 'pink', 'brown', 'gray', 'olive', 'chocolate', 'gold',
    'navy', 'maroon', 'turquoise', 'darkgreen', 'darkorange', 'orchid',
    'indigo', 'crimson', 'coral', 'slateblue', 'tomato', 'mediumseagreen', 'deeppink'
]

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='mollweide')
ax.set_facecolor('#d0f0f8')
plt.grid(True, color='black', ls='--', alpha=0.5)

for label, p in points.items():
    plot_ellipse(ax, *p['coords'], p['l_err'], p['b_err'], p['color'], label)

for (label, (l, b)), color in zip(fundamental_points.items(), fundamental_colors):
    l = (l + 180) % 360 - 180
    l = -l  # Flip left to right
    ax.scatter(deg2rad(l), deg2rad(b), color=color, s=30, marker='*', edgecolors='black', zorder=6)
    ax.text(deg2rad(l), deg2rad(b), f" {label}", fontsize=7, ha='left', va='center', color='black')

xticks_deg = [150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210]
xticks_deg_flipped = [-((x + 180) % 360 - 180) for x in xticks_deg]
xticks = deg2rad(xticks_deg_flipped)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{deg}°" for deg in xticks_deg])

yticks_deg = np.arange(-75, 90, 15)
ax.set_yticks(deg2rad(yticks_deg))
ax.set_yticklabels([f"{t}°" for t in yticks_deg])

handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                       markerfacecolor=p['color'], markersize=8, markeredgecolor='black')
           for label, p in points.items()]
plt.legend(handles=handles, loc='lower left', fontsize=8)

plt.tight_layout()
plt.savefig("sky.png", dpi=300)
