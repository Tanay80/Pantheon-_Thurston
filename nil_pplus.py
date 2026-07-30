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
    
    dhdz = dAdz * (-3 * h / 2) * (1 + sigma * sigma - om_k/3 + (1 - sigma * sigma - om_k) * wbar) / A
    dsigmadz = dAdz * ((3 * sigma * wbar + 2 * d_w) * (1 - sigma * sigma - om_k) / 2 + (1 + sigma) * (sigma * sigma - sigma + 1) + (2 - sigma) * (-om_k - (1 + sigma) * (1 + sigma)) / 2 - om_k * (1 + sigma)) / A
    de2dz = dAdz * (6 * sigma * (1 - e2)) / A
    dom_imdz = dAdz * (-om_im * (2 * sigma * d_w + 3 * (wbar - 1) * sigma * sigma + om_k * (1 + 3 * wbar))) / A
    dom_kdz = dAdz * (-2 * om_k * (1 - 2 * sigma + (-3/2)*(1 + sigma * sigma - om_k/3 + (1 - sigma * sigma - om_k) * wbar))) / A
                                                                                                                                                                             
    return [dAdz, dhdz, dsigmadz, de2dz, dom_imdz, dom_kdz]                                                                                

#Cosmological integral in 'z'
def integralFunc(z, A, h, sigma, e2, alpha):

    A = (1 / (1+z)) * ((np.abs(1 - e2 * np.sin(alpha) * np.sin(alpha)) ** (1/2)) / (np.abs(1 - e2) ** (1/3))) 
    dAdz = ((A*A) * (np.abs(1-e2) ** (1/3)) * (np.abs(1 - e2 * np.sin(alpha) * np.sin(alpha)) ** (1/2))) / ((sigma + 1) * e2 * np.sin(alpha) * np.sin(alpha) + (2 - 3 * np.sin(alpha) * np.sin(alpha)) * sigma - 1)

    G = (1 - e2) ** (1/6)
    H = np.sqrt(1 - e2 * np.sin(alpha) * np.sin(alpha))
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

#Likelihood function
def my_likelihood(h, sigma, e2, om_im, om_k, la, ba, M0, wbar):
    if (1 - om_k - om_im - sigma * sigma) < 0:
        return -1e30

    A = 1
    z0 = 0                                                                                                      
    muTh = 0                                                                                      
    temp = np.empty_like(z)                                                                                     
    alpha = ang_sep(ls, bs, la, ba)                                                                             
                                                                                                                                                                                                                          
    for i in range(len(z)):
        z_ = np.linspace(z0, z[i], nsteps[i])  
        dz = z_[1]-z_[0]

        #Solving coupled ODEs
        var = [A, h, sigma, e2, om_im, om_k]                                               
        par = solve_ivp(evol_equ, t_span=(z0, z[i]), y0=var, t_eval=z_, args=(alpha[i], wbar))
        I0 = integralFunc(z_, par.y.T[:, 0], par.y.T[:, 1], par.y.T[:, 2], par.y.T[:, 3], alpha[i])
        I = -romb(I0, dx=dz) 
        
        if I < 0:                                                                                               
            return -1e30                                                                                 
        
        if is_anchor_data[i] == 1:                                                                              
            temp[i] = m_b_corr[i] - M0 - mu_host[i]                                                             
        else:                                                                                                   
            muTh = 5 * np.log10((1 + z_hel[i]) * I) + 42.384103515                                           
            temp[i] = m_b_corr[i] - M0 - muTh                                                                
    
    chi2 = temp @ C_inv @ temp                                                                                                  
    return -chi2 / 2
