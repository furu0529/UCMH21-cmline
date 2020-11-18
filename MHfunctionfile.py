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
import sys; sys.path.append('..')
import CAMBPS
import pandas as pd
import itertools
import constnumberfile as cn
import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'MHfunctionfile.ipynb'])


# In[ ]:





# In[2]:


def laterndim(x,n):
    a=x.ndim
    for i in range(n):
        b=np.expand_dims(x,axis=i+a)
        x=b
    return x
def forendim(x,n):
    a=x.ndim
    for i in range(n):
        b=np.expand_dims(x,axis=0)
        x=b
    return x

def rhob(z):
    rho=cn.ob*(1+z)**3*cn.rc0
    return rho
def rhoDM(z):
    rho=cn.odm*(1+z)**3*cn.rc0
    return rho
def rhoM(z):
    rho=cn.om*(1+z)**3*cn.rc0
    return rho
def Tcmb(z):
    tcmb=cn.tcmb0*(1+z)
    return tcmb


# In[3]:


def Mvir(z,Tvir):
    Mvir=1.98**(-1.5)*1e2*cn.om**(-0.5)*cn.h**(0.5)*Tvir**(1.5)*((1+z)/10)**(-1.5)
    return Mvir

def MJ(z):
    MJ=5.73*1e3*((cn.om*cn.h**2)/0.15)**(-0.5)*((cn.ob*cn.h**2)/0.022)**(-0.6)*((1+z)/(10))**(1.5)
    return MJ


# In[4]:


def mNFW(x):
    return np.log(x)-(x/(1+x))
def mNFWint(x):
    return 1-(np.log(1+x)/x)
def gaN(c):
    return (2*(6*c**2+6*c+1)/(1+3*c)**2)-(c**2/((1+3*c)*(1+c)*mNFW(c)))
def etaN(ga,c):
    return ga**(-1)*(3*((1+c)/(1+3*c))+3*(ga-1)*(c/mNFW(c))*mNFWint(c))


# In[5]:


def TRgas(x,c,tg0,rg0):
    ygg=1.0-((3.0*((gaN(c)-1.0)*c)*mNFWint(x+1e-15))/((etaN(gaN(c),c)*gaN(c)*mNFW(c))))
    yg=ygg**(1.0/(gaN(c)-1.0))
    return tg0*ygg,rg0*yg

def TRgasave(Tgas,Rgas,x):
    Tave=integrate.simps(x**2*Tgas,x)*3
    Rave=integrate.simps(x**2*Rgas,x)*3
    return Tave,Rave


# In[6]:


def TRgas2(x,y,c,tg0,rg0):
    r=np.sqrt(x[:,None]**2+y[None,:]**2)*c+1e-15
    ygg=1.0-((3.0*((gaN(c)-1.0)*c)*mNFWint(r))/(etaN(gaN(c),c)*gaN(c)*mNFW(c)))
    yg=ygg**(1.0/(gaN(c)-1.0))
    return tg0*ygg,rg0*yg


# In[7]:


#プレスシェヒター理論
def tophat(M):
    k=np.logspace(-4,4,500)
    R=(3*M*cn.mo/(4*np.pi*cn.om*cn.rc0))**(1/3)/cn.Mpc
    kR=k[None,:]*R[:,None]
    topwin=3*(np.sin(kR)-kR*np.cos(kR))/(kR)**3
    return topwin,k
def varips(M,zlist):
    topwin,k=tophat(M)
    dvari=CAMBPS.PSz(k,zlist)[None,:,:]*topwin[:,None,:]**2/k[None,None,:]
    varisum2=integrate.simps(dvari,k[None,None,:],axis=2)
    vari=varisum2**(0.5)
    return vari
def PSMF(M,zlist):
    sigma=varips(M,zlist)
    dlnM=np.diff(np.log(sigma),axis=0)/np.diff(np.log(M))[:,None]
    nuc=cn.deltac/sigma[0:len(M)-1]
    print(nuc.shape,M.shape,dlnM.shape)
    dndm=-np.sqrt(2/np.pi)*(cn.rc0*cn.om*1.47*1e28*1e9/M[0:len(M)-1,None]**2)*dlnM*nuc*np.exp(-nuc**2/2)
    return dndm

def nueff(z,Tgasave):
    nu=(cn.nu0/(cn.c0*(1+z)))*np.sqrt(2*np.pi*cn.kb*Tgasave/cn.mp)
    return nu

def diffTb(z,Tbave):
    deltaTb=(Tbave/(1+z))-cn.tcmb0
    return deltaTb

#background radiation
def diffTbave(z,rv,Tgasave,Tbave,dn_dm,M):
    Const=(cn.c0*(1+z)**4/(cn.nu0*cn.H0m*(1+z)**(1.5)))
    Ahalo=np.pi*rv**2
    dim3=nueff(z,Tgasave)*diffTb(z,Tbave)*Ahalo
    ddTb=Const*dn_dm*dim3
    dTb=integrate.simps(ddTb,M)
    return dTb,ddTb
#bias
def dbiasPS(sigmaMH):
    deltac=1.68
    nu=deltac/sigmaMH
    dbias=1+(nu**2-1)/deltac
    return dbias 

def wbMH(bias,ddTb,M):
    bdTb=integrate.simps(bias*ddTb,M)
    dTb=integrate.simps(ddTb,M)
    beta=bdTb/dTb
    return beta


# In[8]:


def Wcy(z,k,Dnu):
    Rcylinder=cn.Dthetaradian[None,:]*(1+z)[:,None]*cn.Da(z)[:,None]/2 #Mpc
    Lcylinder=(cn.c0/cn.H0m)*(Dnu*1e6/cn.nu0)*(1+z)**(0.5) #Mpc
    xcy=np.linspace(1e-15,1-1e-15,1500)
    SBv=k[:,None,None,None]*Rcylinder[None,:,:,None]*(1-xcy[None,None,None,:]**2)**(0.5)
    Jv=special.jv(1,SBv)
    winup=np.sin(k[:,None,None,None]*Lcylinder[None,:,None,None]*xcy[None,None,None,:]/2)**2*Jv**2
    winunder=xcy**2*(1-xcy**2)
    win=winup/winunder[None,None,None,:]
    winsum=integrate.simps(win,xcy[None,None,None,:],axis=3)
    winsum0=(16/(Rcylinder[None,:,:]**2*Lcylinder[None,:,None]**2*k[:,None,None]**4))*winsum
    return winsum0

def varidensity(z,k,matterpower,Wcy):
    dsigma2M=(matterpower[:,:,None]*Wcy)/k[:,None,None]
    sigma2M=integrate.simps(dsigma2M,k[:,None,None],axis=0)
    sigmaM=sigma2M**(0.5)
    return sigmaM


# In[9]:


def Tbaveigm(z,xHI):
    zre=200
    Tgas=cn.tcmb0*(1+zre)*((1+z)/(1+zre))**2
    rgas=cn.ob*(1+z)**3*cn.rc0
    nHI=(1-cn.YY)*(rgas/cn.mp)
    cHI=3.1*1e-17*Tgas**(0.357)*np.exp(-32/Tgas)
    ce=10**(-9.607+0.5*np.log10(Tgas)*np.exp(-np.log10(Tgas)/1800))*1e-6
    xc=(cn.tf/(cn.A10*cn.tcmb0*(1+z)))*nHI*((1-xHI)*cHI+xHI*ce+xHI*3.2*cHI)
    Ts=(((cn.tcmb0*(1+z))**(-1)+xc*Tgas**(-1))/(1+xc))**(-1)
    Tbaveigm=9*(1-xHI)*(1+z)**(0.5)*1e-3*(1-(cn.tcmb0*(1+z))/Ts)
    return Tbaveigm


# In[10]:


@jit ('f8(f8[:],f8[:],f8[:],f8[:],f8[:],f8)')
def TBave(dMlist,Mlist,zlist,k05,k06,xHI):
    C=np.zeros((len(Mlist),len(zlist)))
    Rs=np.zeros((len(Mlist),len(zlist)))
    Rv=np.zeros((len(Mlist),len(zlist)))
    Mv=np.zeros((len(Mlist),len(zlist)))
    TGasave=np.zeros((len(Mlist),len(zlist)))
    TBav=np.zeros((len(Mlist),len(zlist)))
    for i,mv in enumerate(Mlist):
        for k,z in enumerate(zlist):
            rv=(((3*mv*cn.mo/(200*cn.om*4*np.pi*cn.rc0))**(1/3))*(1/(1+z)))*(1/cn.Mpc)
            zc=z 
            rs=((1+zc)*rv/10)*(1e14/mv)**(-0.20) 
            c=rv/rs
            MnDM=4*np.pi*rs**(3.0)*mNFW(c)
            ros=mv/MnDM
            tg0=(cn.G*cn.mu*mv*cn.mo*cn.mp*etaN(gaN(c),c))/(3*rv*cn.Mpc*cn.kb)
            FF=(3.0*(gaN(c)-1.0)*c)/(etaN(gaN(c),c)*gaN(c)*mNFW(c))
            x=np.linspace(1e-5,c,4000)
            ygas_x=(1.0-FF*mNFWint(x))**(1.0/(gaN(c)-1))*x**(2.0)
            mgas=integrate.simps(ygas_x , x)
            rg0=(cn.ob*ros*mNFW(c))/(cn.om*mgas*cn.kgm_MMpc)
            Tgas,Rgas=TRgas(k06,c,tg0,rg0)
            Tgasave,Rgasave=TRgasave(Tgas,Rgas,k06)
            Tgas,Rgas=TRgas2(k05,k05,c,tg0,rg0)
            nHI=(1-cn.YY)*(Rgas/cn.mp)
            cHI=np.zeros((len(k05),len(k05)))
            for l,x in enumerate(k05):
                for m,y in enumerate(k05):
                    if (Tgas[l,m]<1000):
                        cHI[l,m]=3.1*1e-17*Tgas[l,m]**(0.357)*np.exp(-32/Tgas[l,m])
                    else:
                        cHI[l,m]=4*3.1*1e-17*(Tgas[l,m]/3)**(0.357)*np.exp(-32/(Tgas[l,m]/3))
            ce=10**(-9.607+0.5*np.log10(Tgas)*np.exp(-np.log10(Tgas)/1800))*1e-6
            xc=(cn.tf/(cn.A10*cn.tcmb0*(1+z)))*nHI*((1-xHI)*cHI+xHI*ce+3.2*xHI*cHI)
            TS=(((cn.tcmb0*(1+z))**(-1)+xc*Tgas**(-1))/(1+xc))**(-1)
            rv0=rv*cn.Mpc
            Tgas,Rgas=TRgas2(k05,k05,c,tg0,rg0)
            phinu=(cn.c0/cn.nu0)*np.sqrt(cn.mp/(2*np.pi*cn.kb*Tgas))
            nHI=(1-cn.YY)*(Rgas/cn.mp)
            TBr=np.zeros((len(k05)))
            for n,w in enumerate(k05):
                tau=cn.tau0*rv0*np.sqrt(1-w**2)*nHI[n,:]*phinu[n,:]/TS[n,:]
                taucum=integrate.cumtrapz(tau , k05 ,axis=0, initial=0)
                Tb01=cn.tau0*rv0*nHI[n,:]*phinu[n,:]*np.exp(-taucum)
                Tb0=Tb01*np.sqrt(1-w**2)
                TBr[n]=Tcmb(z)*np.exp(-taucum[len(k05)-1])+integrate.simps(Tb0,k05)
            Tbh=TBr[len(k06)-1:len(TBr)]
            Tbav=integrate.simps(k06*Tbh,k06)*2
            C[i,k]=c
            Rs[i,k]=rs
            Rv[i,k]=rv
            Mv[i,k]=mv
            TGasave[i,k]=Tgasave
            TBav[i,k]=Tbav
        if(i%10==0):
            print(i)
    dndm=abs(PSMF(dMlist,zlist))
    dTB1=np.zeros((len(zlist)))
    biasw=np.zeros((len(zlist)))
    for k,z in enumerate(zlist):
        DTb,DdTb=diffTbave(z,Rv[:,k],TGasave[:,k],TBav[:,k],dndm[:,k],Mlist)
        Dbias=dbiasPS(varips(Mlist,zlist))
        biasW=wbMH(Dbias[:,k],DdTb,Mlist)
        dTB1[k]=DTb
        biasw[k]=biasW
    return C,Rs,Rv,Mv,Tgasave,TBav,dTB1,biasw


# In[11]:


def Noise(Aff,Dtheta,Dnu,tobs,z):
    Noiseflu=7.36*(1e5/Aff)*(1/Dtheta)**(2)*(1/Dnu)**(1/2)*(100/tobs)**(1/2)*(1+z)**(4.6)*1e-6
    return Noiseflu


# In[ ]:





# In[ ]:




