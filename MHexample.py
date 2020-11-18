#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import camb
from camb import model, initialpower
from matplotlib.font_manager import FontProperties
from scipy import integrate
from scipy import special
from scipy import signal, interpolate
from numba import jit
from memory_profiler import profile
import MHfunctionfile as ff
import sys; sys.path.append('../..')
import CAMBPS
import pandas as pd
import itertools
import constnumberfile as cn
import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'MHexample.ipynb'])


# In[2]:


zmin=10
zlist=np.arange(zmin,40+1,2)
Mlist=np.linspace(ff.MJ(zmin),ff.Mvir(zmin,1e4),20)
dMlist=np.linspace(ff.MJ(zmin),ff.Mvir(zmin,1e4),21)
dxy=30
J=5
k01=np.logspace(-J,0,dxy)
k02=np.logspace(0,-J,dxy)
k05=np.hstack((-k02,[0],k01))
k06=np.hstack(([0],k01))


# In[4]:


kp=np.logspace(-4,4,500)
C,Rs,rv,Mv,Tgasave,Tbave,dTb,biasw=ff.TBave(dMlist,Mlist,zlist,k05,k06,0)
sigmavari=ff.varidensity(zlist,kp,CAMBPS.PSz(kp,zlist).transpose(1,0),ff.Wcy(zlist,kp,20))
variTb=biasw[:,None]*dTb[:,None]*sigmavari
Tbaveigm=ff.Tbaveigm(zlist,1e-4)
Tbvariigm=Tbaveigm[:,None]*sigmavari


# In[5]:


f1 = interpolate.interp1d(zlist, abs(variTb)+abs(Tbvariigm)+abs(ff.Noise(8*1e5,20,3,1000,zlist)[:,None]),axis=0,kind='linear')
def vari(x):
    y=f1(x)
    return y


# In[6]:


figsize_px = np.array([640, 480])
dpi = 100
figsize_inch = figsize_px / dpi
fig, ax = plt.subplots(figsize=figsize_inch, dpi=dpi)
plt.semilogy(zlist,abs(variTb[:,0]),color='black',linestyle='dotted',label='MH')
#plt.semilogy(zlist,abs(Tbave[:,0]),color='black',linestyle='dotted',label='MH')
plt.semilogy(zlist,abs(Tbvariigm[:,0]),color='black',linestyle='dashed',label='IGM')
plt.semilogy(zlist,abs(Tbvariigm[:,0])+abs(variTb[:,0]),color='black',linewidth=2,label='MH+IGM')
plt.semilogy(zlist,f1(zlist)[:,0],color='red')
plt.semilogy(zlist,ff.Noise(8*1e5,20,3,1000,zlist),color='black',linestyle='dashdot',label='NoiseSKA1-low')
#plt.semilogy(zlist,ff.Noise(8*1e7,20,3,1000,zlist),color='black',linestyle='dashdot',label='NoiseSKA2')
#plt.axvline(x=20)
#plt.xlim(15,28)
plt.ylim(1e-7,1e-1)
plt.legend(fontsize=11)
plt.xlabel('z',size=15)
plt.ylabel(r'$\langle \delta T_{b}^2 \rangle^{1/2}$[K]',size=15)
plt.tick_params(labelsize=13)
plt.savefig('/Users/kokoorikunihiko/Desktop/noise.eps',bbox_inches='tight' ,dpi=300)


# In[ ]:




