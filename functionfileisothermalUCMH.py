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
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'functionfileisothermalUCMH.ipynb'])

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

def mMoore(x):
    return 2*(np.log(np.sqrt(x)+np.sqrt(1+x))-np.sqrt(x/(1+x)))
def mMooreint(x):
    return 2*(np.sqrt((1+x)/x)-(np.log(np.sqrt(x)+np.sqrt(1+x))/x))
def mdm_u2(x):
    return 2.0*(np.sqrt((1.0+x)/x)-(1.0/x)*np.log(np.sqrt(x)+np.sqrt(1.0+x)))
def gaM(c):
    return (16*c**2+20*c+5)/(3*(1+2*c)**2)-((2*c**(1.5))/(3*(1+2*c)*(1+c)**(0.5)*mMoore(c)))
def etaM(c):
    return gaM(c)**(-1)*(3*((1+c)/(1+3*c))+3*(gaM(c)-1)*(c/mMoore(c))*mMooreint(c))

def mNFW(x):
    return np.log(x)-(x/(1+x))
def mNFWint(x):
    return 1-(np.log(1+x)/x)
def gaN(c):
    return (2*(6*c**2+6*c+1)/(1+3*c)**2)-(c**2/((1+3*c)*(1+c)*mNFW(c)))
def etaN(c):
    return gaN(c)**(-1)*(3*((1+c)/(1+3*c))+3*(gaN(c)-1)*(c/mNFW(c))*mNFWint(c))

def kvir(z,Tvir):
    Mvir=1.98**(-1.5)*1e2*cn.om**(-0.5)*cn.h**(0.5)*Tvir**(1.5)*((1+z)/10)**(-1.5)
    k4=(4*np.pi*(2*np.pi)**3*cn.rc0*cn.om/(3*Mvir*cn.mo))**(1/3)*cn.Mpc
    return k4

def MJ(z):
    MJ=5.73*1e3*((cn.om*cn.h**2)/0.15)**(-0.5)*((cn.ob*cn.h**2)/0.022)**(-0.6)*((1+z)/(10))**(1.5)
    return MJ

def ksUCMHprofiledata(z,zc,k):
    c1=np.logspace(-1,4,20000)
    fv=(20/9)*c1**3*((1+z)/(1+zc))**3
    A=np.argmin(np.abs(fv-mMoore(c1)))
    c=c1[A]
    Rs=0.7*((1+zc)*k)**(-1.0)
    rv=c*Rs
    mv=(200*4*np.pi*cn.om*(1+z)**3*cn.rc0*(rv**3)*cn.kgm_MMpc)/3
    return c,Rs,rv,mv


# In[2]:


def trg0(z,zc,c,rv,mv,A,etaran,B):
    FF=(3.0*c)/(etaran*mMoore(c))
    rangec=np.linspace(1e-8,c,1000)
    ygas_x=np.exp((-FF[:,None]*mdm_u2(rangec)[None,:]))*rangec[None,:]**2
    mgas=integrate.simps(ygas_x,rangec[None,:],axis=1)
    rg0=(cn.Ros(zc,A)*mMoore(c)*cn.ob)/(cn.odm*mgas)
    argeta=np.argmin(abs(rg0-B*rhob(z)*np.exp(6*c/(etaran*mMoore(c)))))
    etaM=etaran[argeta]
    rg0=rg0[argeta]
    tg0=(cn.G*cn.mu*mv*cn.mo*cn.mp*etaM)/(3*rv*cn.Mpc*cn.kb)
    x=np.logspace(-5,5,100)
    ygas=np.exp(-FF[argeta]*mdm_u2(x))
    return tg0,rg0,etaM#,rg0*ygas,x
    
def Mass_rg0(Mass,z,zc,k,etaM):
    c,rs,rv,mv=ksUCMHprofiledata(z,zc,k)
    tg0=(cn.G*cn.mu*mv*cn.mo*cn.mp*etaM)/(3*rv*cn.Mpc*cn.kb)
    FF=(3.0*c)/(etaM*mMoore(c))
    rangec=np.linspace(1e-8,c,1000)
    ygas_x=np.exp(-FF*mdm_u2(rangec))*rangec**2
    mgas=integrate.simps(ygas_x,rangec)
    rg0=Mass/(4*np.pi*mgas*(rs*cn.Mpc)**3)
    return tg0,rg0

#baryon darkmatter 相対速度
def bondic(k,z,zc,etaM):
    c,rs,rv,mv=ksUCMHprofiledata(z,zc,k)
    cs0=3*1e4*(1/1000)
    Mac=(np.pi*cn.ob*cn.rc0*cn.G**2*(mv*cn.mo)**2)/((cs0)**3)
    Mbondi=(Mac/cn.H0)*(2/3)*(1/(1+z)**(1.5)-1/(1+zc)**(1.5))
    tg0,rg0=Mass_rg0(Mbondi,z,zc,k,etaM)
    return tg0,rg0

#宇宙論的速度
@jit ('f8(f8,f8,f8,f8)')
def bondi(k,z,zc,etaM):
    if (zc<1000):
        zrange=np.arange(z,zc,5)
    else:
        zrange=np.arange(z,1000,5)
    mv=np.zeros(len(zrange))
    Tk=np.zeros(len(zrange))
    for i,zr in enumerate(zrange):
        mv[i]=ksUCMHprofiledata(zr,zc,k)[3]
        if (zr<175):
            Tk[i]=cn.tcmb0*(1+zr)
        else:
            Tk[i]=cn.tcmb0*(1+cn.zde)*((1+z)/(1+cn.zde))**2
    cs=cn.C*np.sqrt(np.array(Tk))
    rhoz=cn.ob*(1+zrange)**3*cn.rc0
    Macc=(np.pi*rhoz*cn.G**2*(np.array(mv)*cn.mo)**2)/(cs**3)
    dMacc=Macc/(cn.H0*(1+zrange)**(2.5))
    Maccsum=integrate.simps(dMacc,zrange)
    tg0,rg0=Mass_rg0(Maccsum,z,zc,k,etaM)
    return tg0,rg0


# In[3]:


def TRgas(x,c,tg0,rg0,etaM):
    r=x*c+1e-15
    yg=np.exp(-((3.0*c*mdm_u2(r))/(etaM*mMoore(c))))
    return tg0,rg0*yg


# In[4]:


def TRgasave(Tgas,Rgas,x):
    Tave=integrate.simps(x**2*Tgas,x)*3
    Rave=integrate.simps(x**2*Rgas,x)*3
    return Tave,Rave


# In[5]:


def TRgas2(x,y,c,tg0,rg0,etaM):
    r=np.sqrt(x[:,None]**2+y[None,:]**2)*c+1e-15
    yg=np.exp(-((3.0*c*mdm_u2(r))/(etaM*mMoore(c))))
    return tg0,rg0*yg


# In[6]:


#@jit
def Ts(xlist,ylist,z,c,k,xHI,tg0,rg0,etaM):
    Tgasigm,Rgasigm=TRgasigm(z)
    Tgas0,Rgas0=TRgas2(xlist,ylist,c,tg0,rg0,etaM)
    Tgas=Tgas0+Tgasigm
    Rgas=Rgas0+Rgasigm
    nHI=(1-cn.YY)*(Rgas/cn.mp)
    if (Tgas<1000):
        cHI=3.1*1e-17*Tgas**(0.357)*np.exp(-32/Tgas)
    else:
        cHI=4*3.1*1e-17*(Tgas/3)**(0.357)*np.exp(-32/(Tgas/3))
    ce=10**(-9.607+0.5*np.log10(Tgas)*np.exp(-np.log10(Tgas)/1800))*1e-6
    xc=(cn.tf/(cn.A10*cn.tcmb0*(1+z)))*nHI*((1-xHI)*cHI+xHI*ce+3.2*xHI*cHI)
    Ts=(((cn.tcmb0*(1+z))**(-1)+xc*Tgas**(-1))/(1+xc))**(-1)
    return Ts,xc


# In[7]:


@jit ('f8(f8[:],f8[:],f8,f8,f8,f8,f8,f8[:,:],f8,f8)')
def Tbr(xlist,ylist,z,k,c,tg0,rg0,Ts,rv,etaM):
    rv0=rv*cn.Mpc
    r=np.sqrt(xlist[:,None]**2+ylist[None,:]**2)*c+1e-15
    Tgas0,Rgas0=TRgas2(xlist,ylist,c,tg0,rg0,etaM)
    Tgasigm,Rgasigm=TRgasigm(z)
    Tgas=Tgas0+Tgasigm
    Rgas=Rgas0+Rgasigm
    phinu=(cn.c0/cn.nu0)*np.sqrt(cn.mp/(2*np.pi*cn.kb*Tgas))
    nHI=(1-cn.YY)*(Rgas/cn.mp)
    Tbsc=np.zeros((len(xlist)))
    Taucum=np.zeros((len(xlist),len(ylist)))
    for i,x in enumerate(xlist):
        tau=cn.tau0*rv0*np.sqrt(1-x**2)*nHI[i,:]*phinu/Ts[i,:]
        taucum=integrate.cumtrapz(tau , ylist ,axis=0, initial=0)
        Tb01=cn.tau0*rv0*nHI[i,:]*phinu*np.exp(-taucum)
        Tb0=Tb01*np.sqrt(1-x**2)
        Tbsc[i]=Tcmb(z)*np.exp(-taucum[len(ylist)-1])+integrate.simps(Tb0,ylist) 
        Taucum[i]=taucum
    return np.array(Tbsc),np.array(Taucum)


# In[8]:


def TbaveM(Tb,k06):
    Tbh=Tb[len(k06)-1:len(Tb)]
    Tbave=integrate.simps(k06*Tbh,k06)*2
    return Tbave


# In[9]:


def shmass(zo,zcoll):
    x0=(2/3)*(cn.C/cn.H0)*(cn.tcmb0**(0.5)/(1+cn.zde))*((1+cn.zde)**(1.5)-(1+zo)**(1.5))
    x1=(cn.C/cn.H0)*cn.tcmb0**(0.5)*np.log((1+zcoll)/(1+cn.zde))
    SHM=(4*np.pi/3)*cn.om*cn.rc0*(x0[None,:]+x1[:,None])**3
    return SHM


# In[ ]:





# In[10]:


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


# In[11]:


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
    return beta

def powertrans(k,Am):    
    up=(cn.omrad/cn.om)**2*Am[None,:]
    Ck=4*np.sqrt(cn.omrad)*k[:,None]*cn.c0/(np.sqrt(3)*cn.H0m*cn.om)
    down=81*(-(7/2)+0.577+np.log(Ck))**2
    Acurvature=up/down
    return Acurvature


# In[12]:


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


# In[13]:


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


# In[14]:


def TRgasigm(z):
    zre=200
    Tgas=cn.tcmb0*(1+zre)*((1+z)/(1+zre))**2
    Rgas=cn.ob*(1+z)**3*cn.rc0
    return Tgas,Rgas


# In[15]:


#c,Rs,rv,mv=ksUCMHprofiledata(25,25,1e2)
#tg0,rg0,etaM,Tgas,x=trg0(25,25,c,rv,mv,30)
#print(tg0,rg0,etaM)
#print(x.shape,Tgas.shape)
#print(c)
#plt.loglog(x,Tgas)
#plt.axhline(y=rhob(25))
#plt.axhline(y=rg0)
#plt.axvline(x=rv/Rs)
#plt.axvline(x=1)


# In[ ]:





# In[ ]:




