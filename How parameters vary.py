import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

alpha = np.pi/4                  #Arbitrary
wbar = 1/3                          #Radiation domination
d_w = 0                                 #Very nearly isotropic 

def equations(z, y):
    A, h, sigma, e2, om_im, om_k  = y
    
    A = (1 / (1+z)) * ((np.abs(1 - e2 * np.cos(alpha) * np.cos(alpha)) ** (1/2)) / (np.abs(1 - e2) ** (1/3))) 
    dAdz = ((A*A) * (np.abs(1-e2) ** (1/3)) * (np.abs(1 - e2 * np.cos(alpha) * np.cos(alpha)) ** (1/2))) / ((sigma + 1) * e2 * np.cos(alpha) * np.cos(alpha) + (2 - 3 * np.cos(alpha) * np.cos(alpha)) * sigma - 1)
    
    dhdz = dAdz * (-3 * h / 2) * (1 + sigma * sigma - om_k/3 + (1 - sigma * sigma - om_k) * wbar) / A
    dsigmadz = dAdz * ((3 * sigma * wbar + 2 * d_w) * (1 - sigma * sigma - om_k) / 2 + (1 + sigma) * (sigma * sigma - sigma + 1) + (2 - sigma) * (om_k - (1 + sigma) * (1 + sigma)) / 2) / A
    de2dz = dAdz * (6 * sigma * (1 - e2)) / A
    dom_imdz = dAdz * (-om_im * (2 * sigma * d_w + 3 * (wbar - 1) * sigma * sigma + om_k * (1 + 3 * wbar))) / A
    dom_kdz = dAdz * (-2 * om_k * (1 + sigma + (-3/2)*(1 + sigma * sigma - om_k/3 + (1 - sigma * sigma - om_k)*wbar))) / A
    
    return [dAdz, dhdz, dsigmadz, de2dz, dom_imdz, dom_kdz]

z_span = (0, 5000)
#z_span = (0, 3500)                                                                                      #Radiation domination
#z_span = (0, 1e9)                                                                                      #BBN
z_eval = np.linspace(z_span[0], z_span[1], 500)
y0 = [1.0, 0.7, 0.001, 0.001, 0.3, 0.0001]
solution = solve_ivp(equations, z_span, y0, t_eval=z_eval)

z_vals = solution.t
plot_quantity = solution.y[2]

plt.figure(figsize=(8, 6))
plt.plot(z_vals, plot_quantity)
plt.xlabel('z')
plt.ylabel(r'$\sigma$')
plt.grid(True)
plt.savefig('Shear_RD.png')

#Interpolation
print("-" * 40)
while True:
    user_input = input(f"Enter a value for z (Range {z_span[0]} - {z_span[1]}) or 'q' to quit: ").strip()
    
    if user_input.lower() == 'q':
        print("Exiting query loop...")
        break
        
    if not user_input:
        continue
        
    try:
        user_z = float(user_input)
        if z_span[0] <= user_z <= z_span[1]:
            interp_val = np.interp(user_z, z_vals, plot_quantity)
            print(f"\n{'·' * 20}")
            print(f"z = {user_z:.2f}")
            #print(f"σ/H = {interp_val:.6e}")
            print(f"Shear = {interp_val:.6e}")
            print(f"{'·' * 20}\n")
        else:
            print(f" >> Error: {user_z} is out of bounds. Try again.")
            
    except ValueError:
        print(f" >> Error: '{user_input}' is not a number. Try again.")
        
print("-" * 40 + "\n")
