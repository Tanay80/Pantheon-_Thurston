import os
import multiprocessing
import numpy as np
from scipy.integrate import solve_ivp, odeint
import pandas as pd
from astropy.coordinates import SkyCoord
from multiprocessing import Pool, cpu_count
from astropy import units as u
import emcee
from tqdm import tqdm
import getdist
import getdist.plots as gdplt
from getdist import plots, MCSamples
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, romb, simpson
import corner

#EoS parameters for wdm
#wbar will be a constraining parameter
d_w = 0

#Ansatz for selecting data table
def select_data(data_table, n=1701, *args):
    col = []
    for i in args:
        coli = np.array(data_table[i][0:n])
        col.append(coli)
    if len(col) == 0:
        print("Please enter a Column name")
        return None
    elif len(col) == 1:
        return col[0]
    else:
        return col

#Defining 'alpha'
def ang_sep(ls, bs, la, ba):                                                                                                   
    c1 = SkyCoord(l=ls, b=bs, unit="deg", frame="galactic")                                                                     
    c2 = SkyCoord(l=la, b=ba, unit="deg", frame="galactic")                                                                       
    alpha = c1.separation(c2).rad                                                                                                
    return alpha

#Evolution equations
def evol_equ(z, var, alpha, wbar): 
    A, h, sigma, e2, om_im, om_k = var

    A = (1 / (1+z)) * ((np.abs(1 - e2 * np.sin(alpha) * np.sin(alpha)) ** (1/2)) / (np.abs(1 - e2) ** (1/3))) 
    dAdz = ((A*A) * (np.abs(1-e2) ** (1/3)) * (np.abs(1 - e2 * np.sin(alpha) * np.sin(alpha)) ** (1/2))) / ((sigma + 1) * e2 * np.sin(alpha) * np.sin(alpha) + (2 - 3 * np.sin(alpha) * np.sin(alpha)) * sigma - 1)
    
    dhdz = dAdz * (-3 * h) * (sigma**2 + (om_k/3) + (1 + wbar) * (om_im/2)) / A
    dsigmadz = dAdz * ((3*sigma/2)*((wbar + 1)*om_im + sigma**2 + om_k - 1) + (d_w * om_im) + (1 + sigma) * (sigma * sigma - sigma + 1) + (2 - sigma) * (-om_k - (1 + sigma) * (1 + sigma)) / 2 - om_k * (1 + sigma)) / A
    de2dz = dAdz * (6 * sigma * (1 - e2)) / A
    dom_imdz = dAdz * (om_im * (6*sigma**2 + 2*om_k - 3*(1 + wbar)*(1 - om_im))) / A
    dom_kdz = dAdz * (-2 * om_k * (1 - 2*sigma - 3*sigma**2 - om_k - 3*(1 + wbar)*om_im/2)) / A
                                                                                                                                                                             
    return [dAdz, dhdz, dsigmadz, de2dz, dom_imdz, dom_kdz]                                                                                

#Cosmological integral in 'z'
def integralFunc(z, A, h, sigma, e2, alpha):

    A = (1 / (1+z)) * ((np.abs(1 - e2 * np.sin(alpha) * np.sin(alpha)) ** (1/2)) / (np.abs(1 - e2) ** (1/3))) 
    dAdz = ((A*A) * (np.abs(1-e2) ** (1/3)) * (np.abs(1 - e2 * np.sin(alpha) * np.sin(alpha)) ** (1/2))) / ((sigma + 1) * e2 * np.sin(alpha) * np.sin(alpha) + (2 - 3 * np.sin(alpha) * np.sin(alpha)) * sigma - 1)

    G = (1 - e2) ** (1/6)
    H = np.sqrt(1 - e2 * np.cos(alpha) * np.cos(alpha))
    f = G / H 
    
    integral = dAdz * f / (h * A**2) 
    return integral

# Loading the data file
df = pd.read_csv("./pplus.csv")                                                                                                                                       
N = df.shape[0]                                                                                                           
z, z_hel, is_anchor_data, m_b_corr, mu_host, raEquatorial, decEquatorial = select_data(df, N, "zCMB", "zHEL", "IS_CALIBRATOR", "m_b_corr", "CEPH_DIST", "RA", "DEC") 

# Loading cov matrix
Matrix = np.loadtxt("pplus_cov.txt")
C = Matrix.reshape(1701,1701)                                                                           
C_inv = np.linalg.inv(C)   

#Defining SNIa coordinates in galactic system
cEquatorial = SkyCoord(ra=raEquatorial, dec=decEquatorial, frame='icrs', unit='deg')                                                                                                                         
cGalactic = cEquatorial.galactic                                                                                                
ls = cGalactic.l                                                                                                             
bs = cGalactic.b   

#nsteps should be 2^k + 1 for romb integral solver
nsteps = np.empty_like(z, dtype=int) 
for i in range(len(z)):                                
    if (z[i]<=0.01):                                   
        nsteps[i] = 17                                  
    elif (z[i]>0.01 and z[i]<=1):                       
        nsteps[i] = 33                                  
    else:                                               
        nsteps[i] = 65
        
mask = (is_anchor_data != 1)

def blowup(z, var, alpha, wbar):
    h, sigma, e2, om_im, om_k = var[1:]
    return min(20.0 - max(abs(sigma), abs(om_k), abs(om_im)), h - 1e-3, 0.99 - e2)
blowup.terminal = True

def rhs_tau(tau, y, wbar):                                                              #tau = ln A, measured from z = 0 (tau = 0 at z = 0)
    h, sigma, e2, om_im, om_k = y
    dh = (-3 * h) * (sigma**2 + (om_k/3) + (1 + wbar) * (om_im/2))
    dsigma = (3*sigma/2)*((wbar + 1)*om_im + sigma**2 + om_k - 1) + (d_w * om_im) + (1 + sigma) * (sigma * sigma - sigma + 1) + (2 - sigma) * (-om_k - (1 + sigma) * (1 + sigma)) / 2 - om_k * (1 + sigma)
    de2 = 6 * sigma * (1 - e2)
    dom = om_im * (6*sigma**2 + 2*om_k - 3*(1 + wbar)*(1 - om_im))
    dok = -2 * om_k * (1 - 2*sigma - 3*sigma**2 - om_k - 3*(1 + wbar)*om_im/2)
    return [dh, dsigma, de2, dom, dok]

def blowup_tau(tau, y, wbar):
    h, sigma, e2, om_im, om_k = y
    return min(20.0 - max(abs(sigma), abs(om_k), abs(om_im)), h - 1e-3, 0.99 - e2)
blowup_tau.terminal = True

TAU_END = -(np.log1p(z.max()) + 0.5)

def lng(e2, t2):
    return 0.5*np.log(1 - e2*t2) - np.log(1 - e2)/3

def tau_of_z(sol, e2_0, t2, zs, tau_min):
    lng0 = lng(e2_0, t2)
    lnz = np.log1p(zs)
    resid = lambda t: lng(sol.sol(t)[2], t2) - lng0 - t - lnz
    lo = np.full_like(zs, tau_min)
    hi = np.zeros_like(zs)
    if np.any(resid(lo) <= 0):
        return None
    for _ in range(50):
        mid = 0.5*(lo + hi)
        pos = resid(mid) > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return 0.5*(lo + hi)

GL_X, GL_W = np.polynomial.legendre.leggauss(24)

def dist_int(sol, e2_0, t2, tau_i):
    S = tau_i[:, None] * (1 - GL_X[None, :]) / 2
    y = sol.sol(S.ravel())
    h_ = y[0].reshape(S.shape)
    e_ = y[2].reshape(S.shape)
    f = (1 - e_)**(1/6) / np.sqrt(1 - e_ * (1 - t2)[:, None])
    integrand = f * np.exp(-S) / h_
    A_start = np.exp(lng(e2_0, t2))
    return (-tau_i / 2) * (integrand @ GL_W) / A_start

#Likelihood function
def my_likelihood(h, sigma, e2, om_im, om_k, la, ba, M0, wbar):
    if (1 - om_k - om_im - sigma * sigma) < 0:
        return -1e30

    alpha = ang_sep(ls, bs, la, ba)
    t2 = (np.sin(alpha)**2)[mask]

    tau_end = TAU_END
    for _ in range(4):
        sol = solve_ivp(rhs_tau, (0, tau_end), [h, sigma, e2, om_im, om_k], method="DOP853", args=(wbar,), events=blowup_tau, dense_output=True, rtol=1e-9, atol=1e-12)
        if sol.status < 0 or not np.all(np.isfinite(sol.y)):
            return -1e30
        tau_min = sol.t[-1]
        tau_i = tau_of_z(sol, e2, t2, z[mask], tau_min)
        if tau_i is not None:
            break
        if sol.status == 1:
            return -1e30
        tau_end -= 1.0
    else:
        return -1e30

    tg = np.linspace(0, tau_min, 400)
    yg = sol.sol(tg)
    e_, s_, t_ = yg[2][None, :], yg[1][None, :], t2[:, None]
    dl = (-t_/(2*(1 - e_*t_)) + 1/(3*(1 - e_))) * 6*s_*(1 - e_) - 1
    if np.any((dl > -1e-3) & (tg[None, :] >= tau_i[:, None])):
        return -1e30

    I = dist_int(sol, e2, t2, tau_i)
    if np.any(I <= 0) or not np.all(np.isfinite(I)):
        return -1e30

    temp = np.empty_like(z)
    temp[~mask] = m_b_corr[~mask] - M0 - mu_host[~mask]
    temp[mask] = m_b_corr[mask] - M0 - (5*np.log10((1 + z_hel[mask]) * I) + 42.384103515)

    chi2 = temp @ C_inv @ temp
    if not np.isfinite(chi2):
        return -1e30
    return -chi2 / 2
