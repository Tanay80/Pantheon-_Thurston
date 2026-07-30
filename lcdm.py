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
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, romb

#Loading Pantheon+ data
df = pd.read_csv("pplus.csv")
N = df.shape[0]
z = np.array(df["zHD"])
z_hel = np.array(df["zHEL"])
is_anchor_data = np.array(df["IS_CALIBRATOR"])
m_b_corr = np.array(df["m_b_corr"])
mu_host = np.array(df["CEPH_DIST"])

#Loading covariance matrix and inverse
Matrix = np.loadtxt('cov.txt')
C = Matrix.reshape(1701,1701)                                                   
C_inv = np.linalg.inv(C)                                                        

#Defining the std. model
def integral(z, om_m):
    return 1/np.sqrt(om_m*(1+z)**3 + (1-om_m))

# Log-likelihood function
def my_likelihood(h, om_m, M0):                                                    

    muTh = 0                                                  
    temp = np.empty_like(z)                                                    

    for i in range(len(z)):              
        z_ = np.linspace(0, z[i], 3)
        dz = z_[1] - z_[0]
        integrand = integral(z_, om_m)
        I = romb(integrand, dx=dz)                  

        if I < 0:                                                               
            return -np.inf                                                      

        if is_anchor_data[i] == 1:                                              
            temp[i] = m_b_corr[i] - M0 - mu_host[i]                             
        else:                                                                     
            muTh = 5 * np.log10((1 + z_hel[i]) * I / h) + 42.384103515       
            temp[i] = m_b_corr[i] - M0 - muTh                                

    chi2 = temp @ C_inv @ temp                    
    return -0.5 * chi2
