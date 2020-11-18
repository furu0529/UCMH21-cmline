#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy import integrate
from scipy import special
from scipy import signal, interpolate

import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'constnumberfile.ipynb'])

#physical constant     
h=0.7010
om=0.1408*h**(-2)
ok=0
ol=0.692
ob=0.02205*h**(-2)
odm=0.1187*h**(-2)
H0m=100*h*1e3 #m s^-1 Mpc^-1                                                    
H0=h/(3.085677*1e17) #s^-1                                                      
rc0=1.878*(1e-26)*(h**2) #kg/m^3                                                
mu=0.6
G=6.67*1e-11     #m^3 kg^-1 s^-2                                                
kb=1.38*1e-23    #J/K                                                           
mp=1.67*1e-27   #kg                                                             
A10=2.85*1e-15  #s^-1                                                           
tf=0.068     #K                                                                 
tcmb0=2.73
YY=0.25
mo=1.989*1e30   #kg                                                             
c0=3.0*1e8          #m/s                                                        
nu0=1420*1e6     #Hz 
kpc=3.0856*1e19 #kpc to m                                                       
Mpc=3.0856*1e22 #Mpc to m                                                       
deltac=1.68
omrad=4.1577*1e-5*h**(-2)
# observation parameter                                                         
zobs=np.array([15,20,25]) #observed redshift                                    
aobs=1/(1+zobs)
Aff=48900 #m^2　望遠鏡の性能の指標                                              
Dtheta=np.array([3,20])    # '　分解能                                          
Dthetaradian=Dtheta*(1/60)*(np.pi/180) # radian 分解能                          
Dnu=3        #MHz 望遠鏡の観測できる厚み？                                      
tobs=np.array([100,1000])    #hour (observe time)                               
#parameter                                                                                                         
kgm_MMpc=1.47*1e28*1e9
#cal constant                                                                   
C=np.sqrt((5*kb)/(3*mp))
zde=175 # decouping redshift                                                    
tau0=(3.0*c0**2*A10*tf/(32.0*np.pi*nu0**2))
def Ros(z,A):
    Ros=A*(1+z)**3*om*rc0
    return Ros
def Da(z):
    Da=(c0/(H0m*(1+z)))*2*(1-(1/np.sqrt(1+z)))
    return Da


# In[ ]:





# In[ ]:




