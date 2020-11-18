#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
from scipy import integrate
from scipy import special
from scipy import signal, interpolate
import itertools
import constnumberfile as cn
import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'CAMBPS.ipynb'])


# In[5]:


#make the power spectrum
pars = camb.CAMBparams()
pars.set_cosmology(H0=70.4, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0);
pars.set_matter_power(redshifts=np.linspace(0,40,101), kmax=1e2)

#Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1e2, npoints = 100)
s8 = np.array(results.get_sigma8())
#print(results.get_sigma8())


# In[6]:


#make the interpolation (ndpk is nondimensional power spectrum.)
ndpk=pk*(kh[None,:]**3/(2*np.pi**2))
f2 = interpolate.interp2d(kh,z, ndpk)
#define the power spectrum function
def PSz(kin,zin):
    matterpower=f2(kin,zin)
    return matterpower


# In[ ]:




