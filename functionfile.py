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
import constnumberfile as cn
from numba import jit

import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'functionfile.ipynb'])

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

def kvir(z,Tvir):
    Mvir=1.98**(-1.5)*1e2*cn.om**(-0.5)*cn.h**(0.5)*Tvir**(1.5)*((1+z)/10)**(-1.5)
    k4=(4*np.pi*(2*np.pi)**3*cn.rc0*cn.om/(3*Mvir*cn.mo))**(1/3)*cn.Mpc
    return k4

def MJ(z):
    MJ=5.73*1e3*((cn.om*cn.h**2)/0.15)**(-0.5)*((cn.ob*cn.h**2)/0.022)**(-0.6)*((1+z)/(10))**(1.5)
    return MJ

#The unit of Mpc^{-1}
def kJb(z):
    zre=200
    kJ=95*((1+zre)/(1+z))**(0.5)
    return kJ

def kJm(z):
    zre=200
    kJ=95*np.sqrt(6)*((1+zre)/(1+z))**(0.5)
    return kJ


# In[2]:


def m(x):
    NFW=np.log(1+x)-(x/(1+x))
    Moore=2*(np.log(np.sqrt(x)+np.sqrt(1+x))-np.sqrt(x/(1+x)))
    return NFW,Moore

def mint(x):
    NFW=1-(np.log(1+x)/x)
    Moore=2*(np.sqrt((1+x)/x)-(np.log(np.sqrt(x)+np.sqrt(1+x))/x))
    return NFW,Moore

def ga(c):
    NFW=(2*(6*c**2+6*c+1)/(1+3*c)**2)-(c**2/((1+3*c)*(1+c)*m(c)[0]))
    Moore=(16*c**2+20*c+5)/(3*(1+2*c)**2)-((2*c**(1.5))/(3*(1+2*c)*(1+c)**(0.5)*m(c)[1]))
    return NFW,Moore

def eta(c):
    NFW=ga(c)[0]**(-1)*(3*((1+c)/(1+3*c))+3*(ga(c)[0]-1)*(c/m(c)[0])*mint(c)[0])
    Moore=ga(c)[1]**(-1)*(3*((1+c)/(1+3*c))+3*(ga(c)[1]-1)*(c/m(c)[1])*mint(c)[1])
    return NFW,Moore

def powertrans(k,Ac):    
    down=(cn.omrad/cn.om)**2
    Ck=4*np.sqrt(cn.omrad)*k*cn.c0/(np.sqrt(3)*cn.H0m*cn.om)
    up=81*(-(7/2)+0.577+np.log(Ck)[:,None])**2*Ac[None,:]
    Am=up/down
    return Am


# In[3]:


def TRgas(x,c,tg0,rg0,i):
    r=x*c+1e-15
    ygg=1.0-((3.0*((ga(c)[i]-1.0)*c)*mint(r)[i])/(eta(c)[i]*ga(c)[i]*m(c)[i]))
    yg=ygg**(1.0/(ga(c)[i]-1.0))
    return tg0*ygg,rg0*yg

def TRgasave(x,c,z,tg0,rg0,i):
    Tgas0,Rgas0=TRgas(x,c,tg0,rg0,i)
    Tgasigm,Rgasigm=TRgasigm(z)
    Tgas=Tgas0+Tgasigm
    Rgas=Rgas0+Rgasigm
    Tave=integrate.simps(x**2*Tgas,x)*3
    Rave=integrate.simps(x**2*Rgas,x)*3
    return Tave,Rave

def TRgas2(xlist,ylist,c,tg0,rg0,i):
    r=np.sqrt(xlist[:,None]**2+ylist[None,:]**2)*c+1e-15
    ygg=1.0-((3.0*((ga(c)[i]-1.0)*c)*mint(r)[i])/(eta(c)[i]*ga(c)[i]*m(c)[i]))
    yg=ygg**(1.0/(ga(c)[i]-1.0))
    tg=tg0*ygg
    return tg,rg0*yg


# In[4]:


@jit ('f8(f8[:],f8[:],f8,f8,f8,f8,f8,f8[:],f8,f8,f8)')
def singleUCMH(xlist,ylist,z,zc,k,A0,B0,k06,xHI,i,ex):
    c1=[np.linspace(0.1,300,20000),np.linspace(0.1,1000,20000)][i]
    if (zc<101):
        c1=np.linspace(0.1,10,20000)
    fv=(B0/(3*A0))*c1**3*((1+z)/(1+zc))**3
    A=np.argmin(np.abs(fv-m(c1)[i]))
    c=c1[A]
    Rs=0.7*((1+zc)*k)**(-1.0)
    rv=c*Rs
    mv=(200*4*np.pi*cn.om*(1+z)**3*cn.rc0*(rv**3)*cn.kgm_MMpc)/3
    tg0=(cn.G*cn.mu*mv*cn.mo*cn.mp*eta(c)[i])/(3*rv*cn.Mpc*cn.kb)
    FF=(3.0*(ga(c)[i]-1.0)*c)/(eta(c)[i]*ga(c)[i]*m(c)[i])
    xr=np.linspace(1e-8,c,1000)
    ygas_x=(1.0-FF*mint(xr)[i])**(1.0/(ga(c)[i]-1))*xr**2
    mgas=integrate.simps(ygas_x,xr)
    if (k<kJm(z)):
        rg0=(cn.Ros(zc,A0)*m(c)[i]*cn.ob)/(cn.odm*mgas)
    else:
        if(zc>1000):
            zc=1000
        cs0=3*1e4*(1/1000)
        Mac=(4*np.pi*cn.ob*cn.rc0*cn.G**2*(mv*cn.mo)**2)/((cs0)**3)
        Mbondi=(Mac/cn.H0)*(2/3)*(1/(1+z)**(1.5)-1/(1+zc)**(1.5))
        rg0a=Mbondi/(4*np.pi*mgas*(Rs*cn.Mpc)**3)
        rg0na=(cn.Ros(zc,A0)*m(c)[i]*cn.ob)/(cn.odm*mgas)
        if(rg0a<rg0na):
            rg0=rg0a
        else:
            rg0=rg0na
    Tgasave,Rgasave=TRgasave(k06,c,z,tg0,rg0,i)
    Tgasigm,Rgasigm=TRgasigm(z)
    Tgas0,Rgas0=TRgas2(xlist,ylist,c,tg0,rg0,i)
    Tgas=Tgas0+Tgasigm
    Rgas=Rgas0+Rgasigm
    nHI=(1-cn.YY)*(Rgas/cn.mp)
    cHI=np.zeros((len(xlist),len(ylist)))
    for i,x in enumerate(xlist):
        for j,y in enumerate(ylist):
            if (Tgas[i,j]<1000):
                cHI[i,j]=3.1*1e-17*Tgas[i,j]**(0.357)*np.exp(-32/Tgas[i,j])
            elif (Tgas[i,j]>1e4 and ex==1):
                cHI[i,j]=0
                Rgas[i,j]=0
            else:
                cHI[i,j]=4*3.1*1e-17*(Tgas[i,j]/3)**(0.357)*np.exp(-32/(Tgas[i,j]/3))
    ce=10**(-9.607+0.5*np.log10(Tgas)*np.exp(-np.log10(Tgas)/1800))*1e-6
    xc=(cn.tf/(cn.A10*cn.tcmb0*(1+z)))*nHI*((1-xHI)*cHI+xHI*ce+3.2*xHI*cHI)
    Ts=(((cn.tcmb0*(1+z))**(-1)+xc*Tgas**(-1))/(1+xc))**(-1)
    rv0=rv*cn.Mpc
    phinu=(cn.c0/cn.nu0)*np.sqrt(cn.mp/(2*np.pi*cn.kb*Tgas))
    nHI=(1-cn.YY)*(Rgas/cn.mp)
    Tbsc=np.zeros((len(xlist)))
    Taucum=np.zeros((len(xlist),len(ylist)))
    for i,x in enumerate(xlist):
        tau=cn.tau0*rv0*np.sqrt(1-x**2)*nHI[i,:]*phinu[i,:]/Ts[i,:]
        taucum=integrate.cumtrapz(tau , ylist ,axis=0, initial=0)
        Tb01=cn.tau0*rv0*nHI[i,:]*phinu[i,:]*np.exp(-taucum)
        Tb0=Tb01*np.sqrt(1-x**2)
        Tbsc[i]=Tcmb(z)*np.exp(-taucum[len(ylist)-1])+integrate.simps(Tb0,ylist) 
        #Taucum[i]=taucum
        Tbh=Tbsc[len(k06)-1:len(Tbsc)]
        Tbave=integrate.simps(k06*Tbh,k06)*2
    return c,rv,mv,tg0,Tgasave,Rgasave,Tbave


# In[5]:


def f(t):
    deltac=1.68
    co1=np.array([1,(2.5)**(0.5),0.5*(2.5)**(0.5)])
    co2=np.array([5/2,5/8])
    A=co1.ndim
    B=co2.ndim
    C=t.ndim
    erf=special.erf(laterndim(co1,C)*forendim(t,A))
    f0t=t**3-3*t
    f1t=np.exp(-laterndim(co2,C)*forendim(t,B)**2)
    f2t=((31/4)*t**2+8/5)*f1t[1]+((t**2/2)-8/5)*f1t[0]
    fmt=(f0t/2)*(erf[1]+erf[2])+(2/(5*np.pi))**(0.5)*f2t
    return fmt

def g1(t):
    g=t**3-3*t
    h1=(1/2)*(special.erf(np.sqrt(5/2)*t)+special.erf(np.sqrt(5/2)*(t/2)))
    h2=np.sqrt(2/(5*np.pi))*(((31/4)*t**2+8/5)*np.exp(-(5/8)*t**2)+((1/2)*t**2-8/5)*np.exp((-5/2)*t**2))
    F=g*h1+h2
    dg=3*t**2-3
    dh1=np.sqrt(5/(2*np.pi))*(np.exp((-5/2)*t**2)+(1/2)*np.exp(-(5/8)*t**2))
    dh2=np.sqrt(2/(5*np.pi))*(t/2)*((-(155/8)*t**2+27)*np.exp(-(5/8)*t**2)+(-5*t**2+18)*np.exp((-5/2)*t**2))
    dF=(t/F)*(dg*h1+g*dh1+dh2)
    return dF

def dn(k,Am_sum,zc):
    deltac=1.68
    t=(deltac/Am_sum[None,:]**(0.5))*(1+zc[:,None])
    co1=np.array([1,(2.5)**(0.5),0.5*(2.5)**(0.5)])
    co2=np.array([5/2,5/8])
    A=co1.ndim
    B=co2.ndim
    C=t.ndim
    erf=special.erf(laterndim(co1,C)*forendim(t,A))
    f0t=t**3-3*t
    f1t=np.exp(-laterndim(co2,C)*forendim(t,B)**2)
    f2t=((31/4)*t**2+8/5)*f1t[1]+((t**2/2)-8/5)*f1t[0]
    fmt=(f0t/2)*(erf[1]+erf[2])+(2/(5*np.pi))**(0.5)*f2t
    hnua=(1/((2*np.pi)**2*3**(1.5)))*(fmt*np.exp(-t**(2.0)/2.0)*t)
    dn_dzc=(k**3/(1+zc)[:,None])*hnua
    return dn_dzc

def nueff(z,Tgasave):
    nu=(cn.nu0/(cn.c0*(1+z)))*np.sqrt(2*np.pi*cn.kb*Tgasave/cn.mp)
    return nu

def diffTb(z,Tbave):
    deltaTb=(Tbave/(1+z))-cn.tcmb0
    return deltaTb

def diffTbave(z,zc,rv,Tgasave,Tbave,Am,k):
    dn_dzc=dn(k,Am,zc)
    Const=(cn.c0*(1+z)**4/(cn.nu0*cn.H0m*(1+z)**(1.5)))
    Ahalo=np.pi*rv**2
    dim3=nueff(z,Tgasave)*diffTb(z,Tbave)*Ahalo
    ddTb=Const*dn_dzc*dim3[:,None]
    dTb=integrate.simps(ddTb,zc[:,None],axis=0)
    return dTb,ddTb


# In[6]:


def dbiasPT(Am,zc):
    deltac=1.68
    ac=1/(1+zc)
    vari=Am[None,:]**(0.5)*ac[:,None]
    nu=deltac/vari
    g=g1(nu)
    dbias=1+(nu**2-g)/deltac
    return dbias

def wbUCMH(bias,ddTb,zc):
    bdTb=integrate.simps(bias*ddTb,zc[:,None],axis=0)
    dTb=integrate.simps(ddTb,zc[:,None],axis=0)
    beta=bdTb/dTb
    return beta,bdTb


# In[7]:


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

def variUCMH(z,k,matterpower,Wcy):
    dsigma2M=(matterpower[:,:,None]*Wcy)/k[:,None,None]
    sigma2M=integrate.simps(dsigma2M,k[:,None,None],axis=0)
    sigmaM=sigma2M**(0.5)
    return sigmaM


# In[8]:


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


# In[9]:


def TRgasigm(z):
    zre=200
    Tgas=cn.tcmb0*(1+zre)*((1+z)/(1+zre))**2
    Rgas=cn.ob*(1+z)**3*cn.rc0
    return Tgas,Rgas


# In[ ]:




