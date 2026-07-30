import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters: (mean, std) for Om_m, Om_k & sigma^2
params = {
    'RH2S2': [(0.458, 0.028), (0.016, 0.003), (4.9e-5, 9.1e-5)],
    'Nil': [(0.409, 0.028), (0.005, 0.003), (2.25e-4, 2.1e-4)],
    'Solv': [(0.413, 0.060), (0.022, 0.015), (1.6e-5, 4e-5)]
}

num_samples = 100_000

# Sampling and computing Omega_Lambda for each model
omega_samples = {}
for label, (I, K, s2) in params.items():
    samples_I = np.random.normal(*I, num_samples)
    samples_K = np.random.normal(*K, num_samples)
    samples_s2 = np.random.normal(*s2, num_samples)
    omega_samples[label] = 1 - samples_I - samples_K - samples_s2

# Plotting
plt.figure(figsize=(10, 6))
colors = {'RH2S2': 'r', 'Nil': 'g', 'Solv': 'b'}
labels = {'RH2S2': r'$R \times H^2/S^2$', 'Nil': 'Nil', 'Solv': 'Solv'}

for label, samples in omega_samples.items():
    mean, std = norm.fit(samples)
    x = np.linspace(samples.min(), samples.max(), 1000)
    pdf = norm.pdf(x, mean, std)
    pdf /= pdf.max()  # Normalize for equal amplitude
    plt.plot(x, pdf, color=colors[label], label=labels[label])

plt.xlabel(r'$\Omega_\Lambda$')
plt.legend()
plt.grid(True)
plt.savefig("om_l_equal_amplitude.png", dpi=300)
plt.show()