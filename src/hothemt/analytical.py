from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from hothemt import K, W, mm, um, mm, m 

if TYPE_CHECKING:
     from dependencies.HotHEMT.src.hothemt.hemt import HEMT

def analytical_RthWEff2D(hemt:HEMT):
    """Computes the analytical solution for a single finger in 2D.
    
    The solution is from Eq 4.121, pg 311 of Heat Transfer Handbook by Bejan and Kraus (2003).

    The 2D simplification makes several assumptions about the geometry which will be asserted,
    but basically it's an infinitely wide (w_f=l_chip) single finger (n_f=1) on a uniform slab
    (k300_Rel=k300_GaN=k300_Sub) with heat flux only out the bottom (h_bot specified, h_con=0,
    h_gat=0).  Temperature dependence of thermal conductivity can be captured but this is not
    implementd yet (so nthr_* must be zero and T_A=300K).
    """
    assert hemt.k300_Rel==hemt.k300_GaN, "Analytical solution only for equal conductivities"
    assert hemt.k300_Rel==hemt.k300_Sub, "Analytical solution only for equal conductivities"
    assert hemt.n_f==1, "Analytical solution only for single finger"
    assert hemt.rows==1, "Analytical solution only for single row"
    assert hemt.w_f==hemt.l_chipy, "Analytical solution only for infinite width"
    if hemt.nthr_GaN!=0 or hemt.nthr_Rel!=0 or hemt.nthr_Sub!=0:
        raise NotImplementedError("Warning: 2D analytical solution only implemented for constant thermal conductivity")
    assert hemt.T_A==300*K, "Analytical solution only for T_A=300K"
    assert hemt.h_bot is not None, "Analytical solution requires specified h_bot"
    assert hemt.h_con==0, "Analytical solution only for h_con=0"
    assert hemt.h_gat==0, "Analytical solution only for h_gat=0"

    t=hemt.t_GaN+hemt.t_Rel+hemt.t_Sub
    a=hemt.L_h/2
    c=hemt.l_chipx/2
    k=hemt.k300_GaN
    h=hemt.h_bot
    eps=a/c
    tau=t/c
    Bi=h*c/k

    print("Analytical solution:")
    print("------------------")

    n=np.arange(1,1000)
    phi_n=(n*np.pi+Bi*np.tanh(n*np.pi*tau))/(n*np.pi*np.tanh(n*np.pi*tau)+Bi)
    RsprW= 1/(k*eps**2*np.pi**3)*np.sum((np.sin(n*np.pi*eps)**2/n**3)*phi_n) # Equation 4.121, page 311
    print(f"RspreadW: {RsprW/(K*mm/W):.2f} K mm/W")

    RslabW = t/(k*(2*c)) + 1/(h*(2*c))
    print(f"RslabW: {RslabW/(K*mm/W):.2f} K mm/W")
    RthWEff2D = RslabW + RsprW
    print(f"RthWEff2D: {RthWEff2D/(K*mm/W):.2f} K mm/W")
    return RthWEff2D

def analytical_RthWEff3D(hemt: HEMT):
    """Computes the analytical solution for a single finger in 3D.
    
    Possible cases:
        two-layer rectangular slabs with h_bot
            from Eq 4.110, pg 305 of Heat Transfer Handbook by Bejan and Kraus
            this is a two-layer model, so either GaN+Rel or Rel+Sub must be combined
            into a single layer with uniform k
        pure spreading into half-space (h_bot is None)
            from Eq 4.50, pg 280 of Heat Transfer Handbook by Bejan and Kraus
            this is a pure spreading model (uniform k), so k300_GaN=k300_Rel=k300_Sub

    If all layers have the same thermal conductivity exponent, temperature dependence
    of thermal conductivity can be captured.
    """

    assert hemt.n_f==1, "Analytical solution only for single finger"
    assert hemt.rows==1, "Analytical solution only for single finger"
    assert hemt.h_con==0, "Analytical solution only for h_con=0"
    assert hemt.h_gat==0, "Analytical solution only for h_gat=0"
    Wtot=hemt.w_f*hemt.n_f*hemt.rows
    if hemt.h_bot is not None:
        if hemt.k300_Rel==hemt.k300_Sub:
            t1=hemt.t_GaN; t2=hemt.t_Rel+hemt.t_Sub
            k1=hemt.k300_GaN; k2=hemt.k300_Rel
        elif hemt.k300_GaN==hemt.k300_Rel:
            t1=hemt.t_GaN+hemt.t_Rel; t2=hemt.t_Sub
            k1=hemt.k300_GaN; k2=hemt.k300_Sub
        else:
            raise NotImplementedError("Analytical solution only for GaN/Rel or Rel/Sub combined layer")
        a=hemt.L_h/2
        b=hemt.w_f/2
        c=hemt.l_chipx/2
        d=hemt.l_chipy/2
        kappa=k2/k1
        L=hemt.w_f
        Bi=hemt.h_bot*L/k1
        alpha=(1-kappa)/(1+kappa)

        sum_limiter=2000
        max_n=int(sum_limiter*(d/(np.pi*(t1+t2))))
        max_m=int(sum_limiter*(c/(np.pi*(t1+t2))))
        n=np.arange(1,max_n)
        m=np.arange(1,max_m)

        lambda_n = n*np.pi/d
        delta_m = m*np.pi/c
        delta_m_2d = np.atleast_2d(delta_m)
        lambda_n_2d = np.atleast_2d(lambda_n).T
        beta_mn_2d = np.sqrt(delta_m_2d**2+lambda_n_2d**2)

        def phi(xi):
            overflow_limiter=100/(4*(t1+t2))
            xi_nanmask=np.array(xi,copy=True)
            xi_nanmask[xi>overflow_limiter]=np.nan
            xi=xi_nanmask
            from numpy import exp
            numerat = alpha*(kappa*xi*L-Bi)*exp(4*xi*t1) \
                    + (kappa*xi*L-Bi)*exp(2*xi*t1) \
                    + (kappa*xi*L+Bi)*exp(2*xi*(2*t1+t2)) \
                    + alpha*(kappa*xi*L+Bi)*exp(2*xi*(t1+t2))
            denomin = alpha*(kappa*xi*L-Bi)*exp(4*xi*t1) \
                    - (kappa*xi*L-Bi)*exp(2*xi*t1) \
                    +(kappa*xi*L+Bi)*exp(2*xi*(2*t1+t2)) \
                    -alpha*(kappa*xi*L+Bi)*exp(2*xi*(t1+t2))
            res=numerat/denomin
            res[np.isnan(res)]=1  # Handle overflows by taking limit xi->inf
            return res

        phi_m_delta = phi(delta_m)
        phi_n_lambda = phi(lambda_n)
        phi_mn_beta_2d = phi(beta_mn_2d)

        Rspr1 = 1/(2*a**2*c*d*k1) *np.sum(np.sin(a*delta_m)**2/delta_m**3 * phi_m_delta)
        Rspr2 = 1/(2*b**2*c*d*k1) *np.sum(np.sin(b*lambda_n)**2/lambda_n**3 * phi_n_lambda)
        Rspr3 = 1/(a**2*b**2*c*d*k1) *np.sum(np.sin(a*delta_m_2d)**2*np.sin(b*lambda_n_2d)**2
                                            /(delta_m_2d**2*lambda_n_2d**2*beta_mn_2d) * phi_mn_beta_2d)
        RsprW = (Rspr1 + Rspr2 + Rspr3)*Wtot
        print("Analytical solution:")
        print("------------------")

        print(f"RspreadW: {RsprW/(K*mm/W):.2f} K mm/W @ LP")

        RslabW = Wtot * (t1/k1 + t2/k2 + 1/hemt.h_bot) / (hemt.l_chipx*hemt.l_chipy)
        print(f"RslabW: {RslabW/(K*mm/W):.2f} K mm/W @ LP")
        RthWEff3D = RslabW + RsprW
        print(f"RthWEff3D: {RthWEff3D/(K*mm/W):.2f} K mm/W @ LP")
    else:
        assert hemt.k300_GaN==hemt.k300_Rel==hemt.k300_Sub, "Analytical (pure spreading) solution only for equal k"
        k = hemt.k300_GaN
        A = hemt.w_f * hemt.L_h
        eps = hemt.w_f/hemt.L_h
        phi = np.sqrt(eps)/np.pi*(1/eps*np.arcsinh(eps)+np.arcsinh(1/eps)+eps/3*(1+1/eps**3-(1+1/eps**2)**1.5))
        print(eps,phi)
        RthWEff3D = phi/(k*np.sqrt(A)) * Wtot

    if hemt.nthr_GaN!=0 or hemt.nthr_Rel!=0 or hemt.nthr_Sub!=0:
        print("Need to Kirchoff transform for T-dependent k")
        assert (hemt.nthr_GaN==hemt.nthr_Rel) and (hemt.nthr_Rel==hemt.nthr_Sub),\
            "Cannot handle different n_th for GaN/Rel/Sub in Kirchoff"
        if hemt.h_bot is None:
            T0=hemt.T_A
        else:
            assert (1/hemt.h_bot) < .05*(1/k1 + 1/k2), "Bottom conductivity should be negligibly large for Kirchoff"
            # As per Bagnall 2014, treat mean temp of bottom as the T0 of Kirchoff
            T0 = hemt.T_A + hemt.P_per_W * Wtot /(hemt.l_chipx * hemt.l_chipy * hemt.h_bot)  # Bagnall 2014, Eq 39
        n_th = hemt.nthr_GaN
        dtheta_top = hemt.P_per_W * RthWEff3D + hemt.T_A - T0
        Ttop = T0 * (1+(dtheta_top*(1-n_th)/T0))**(1/(1-n_th))
        print(f"Kirchoff-transformed top temperature: {Ttop/K:.2f} K")
        RthWEff3D = (Ttop - hemt.T_A) / (hemt.P_per_W)
        print(f"RthWeff3D after Kirchoff: {RthWEff3D/(K*mm/W):.2f} K mm/W @ {hemt.P_per_W/(W/mm):.2f} W/mm")

    return RthWEff3D
