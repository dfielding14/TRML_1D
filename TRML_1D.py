################################################################
# Author: Drummond Fielding & Zirui Chen
# Reference: in prep
# Date: 05 Nov 2021
# Brief: This code calculates the structure of 
#        1D Turbulent Radiative Mixing Layers.
#
# Execution:
# >> python TRML_1D.py
#
# Output:
#   --- TO DO : explain the output
# Overview:
#   --- TO DO : explain the the parameters that need to be set
################################################################


import numpy as np
from scipy import interpolate
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
matplotlib.rc('font', family='sans-serif', size=12)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = 'round'

cycle=plt.rcParams['axes.prop_cycle'].by_key()['color']

import matplotlib.colors as colors
from matplotlib import gridspec

from scipy.interpolate import interp1d
import scipy.integrate as integrate
import scipy.optimize 

gamma               = 5.0/3.0 

def mach(z,w,j): # terminate when supersonic
    T = w[1]
    v = (1/(2*j))*(M - (M**2 - 4*(j**2)*T)**(0.5))
    cs = np.sqrt(gamma*T)
    return np.abs(v/cs) - 0.999

mach.terminal = True

def dip(z,w,j): # terminate when T exceeds the final T
    T = w[1]
    return T/T_cold - 0.999

dip.terminal = True

def bump(z,w,j): # terminate when T exceeds the initial T
    T = w[1]
    return T/T_hot - 1.001

bump.terminal = True

def edot_cool_double_equilibrium(T, vz):
    pressure = T * j / vz
    alpha_heat          = (np.log10(T_cold/T_peak)/np.log10(density_contrast) * (beta_lo - beta_hi)) - beta_hi
    heating_coefficient = (T_cold/T_peak)**((beta_hi-beta_lo)*(1.0 + (np.log10(T_cold/T_peak)/np.log10(density_contrast))))
    Edot_cool = 1.5 * P_hot/t_cool_eff * (T/T_peak)**np.where(T<T_peak, -beta_lo, -beta_hi) * (pressure/P_hot)**2 #* (T/T_peak)**np.where(T<T_peak, -beta_lo, -beta_hi)
    Edot_heat = 1.5 * P_hot/t_cool_eff * heating_coefficient * (pressure/P_hot)**2
    Edot_heat *= np.where(T<(1+epsilon_T)*T_hot, (T/T_peak)**alpha_heat, ((1+epsilon_T)*T_hot/T_peak)**alpha_heat * (T/((1+epsilon_T)*T_hot))**(-beta_hi-0.5))
    return (Edot_cool - Edot_heat)

def edot_cool(T,vz):
    if (T <= T_peak):
        pressure = T * j / vz
        edot_cool = 1.5 * P_hot/t_cool_eff * (pressure/P_hot)**2 * ((T/T_peak)**-beta_lo - heating_coefficient*(T/T_peak)**alpha_heat)
    else:
        pressure = T * j / vz
        edot_cool = 1.5 * P_hot/t_cool_eff * (pressure/P_hot)**2 * ((T/T_peak)**-beta_hi - heating_coefficient*(T/T_peak)**alpha_heat)
    return edot_cool
edot_cool = np.vectorize(edot_cool)

def vx_cos(z):
    return np.where( (z>0)&(z<h),  -(vrel/2)*np.cos(np.pi*z/h), np.sign(z-h/2.) * (vrel/2))

def dvx_dz_cos(z):
    return np.where( (z>0)&(z<h),  (vrel/2) * np.pi/h * np.sin(np.pi*z/h),0)

def d2vx_dz2_cos(z):
    return np.where( (z>0)&(z<h),  (vrel/2) * (np.pi/h)**2 * np.cos(np.pi*z/h),0)



def integrator(z, w, j): 
    dT_dz, T, dvz_dz, vz = w
    if (z <= 0):
        dvx_dz = 0
        d2vx_dz2 =0 
    elif (z > h):
        dvx_dz = 0
        d2vx_dz2 =0 
    else:
        dvx_dz = (vrel/2) * np.pi/h * np.sin(np.pi*z/h)
        d2vx_dz2 = (vrel/2) * (np.pi/h)**2 * np.cos(np.pi*z/h)

    if (T <= T_peak):
        pressure = T * j / vz
        edot_cool = 1.5 * P_hot/t_cool_eff * (pressure/P_hot)**2 * ((T/T_peak)**-beta_lo - heating_coefficient*(T/T_peak)**alpha_heat)
    elif (T > T_peak):
        pressure = T * j / vz
        edot_cool = 1.5 * P_hot/t_cool_eff * (pressure/P_hot)**2 * ((T/T_peak)**-beta_hi - heating_coefficient*(T/T_peak)**alpha_heat)

    cond        = (j/vz) * (f_nu * h**2 * dvx_dz + kappa_0 / (j/vz))
    visc        = Prandtl * (j/vz) * (f_nu * h**2 * dvx_dz + kappa_0 / (j/vz))
    dcond       = f_nu * h**2 * ((j/vz) * d2vx_dz2 - (j/vz**2) * dvx_dz * dvz_dz)
    dvisc       = Prandtl * (f_nu * h**2 * ((j/vz) * d2vx_dz2 - (j/vz**2) * dvx_dz * dvz_dz))
    d2T_dz2     = edot_cool/cond + j*vz/cond * (T/vz**2 * dvz_dz + dT_dz/(gamma-1)/vz) - dT_dz * dcond/cond - visc/cond * (dvx_dz**2 + (4/3.)*dvz_dz**2)
    d2vz_dz2    = (3./4.) * (j/visc) * (dT_dz/vz + dvz_dz*(1-T/vz**2)) - dvz_dz * dvisc/visc
    return np.array([d2T_dz2, dT_dz, d2vz_dz2, dvz_dz])


def do_and_plot(j_over_j_crit):
    j                   = j_crit * j_over_j_crit
    T_initial           = T_hot
    vz_initial          = j/rho_hot
    dT_dz_initial       = -1e-6
    dvz_dz_initial      = 1e-6
    initial_conditions  = [dT_dz_initial, T_initial, dvz_dz_initial, vz_initial]
    stop_distance       = 10**4
    sol = solve_ivp(integrator, [0, stop_distance], initial_conditions, 
        dense_output=True, 
        events=[dip, bump],
        rtol=3e-14, #atol=[1e-9,1e-11,1e-9,1e-11],
        args=[j])
    
    print (sol.message)
    z       = sol.t
    dTdz    = sol.y[0]
    T       = sol.y[1]
    dvzdz   = sol.y[2]
    vz      = sol.y[3]
    P       = T*(j/vz)
    rho     = (j/vz)
    vx      = vx_cos(z)
    dvxdz   = dvx_dz_cos(z)

    plt.plot(sol.t,  T,color='k', label=r'$T  $')
    plt.plot(sol.t,  vz,           label=r'$v_z $')
    plt.plot(sol.t,  vx,           label=r'$v_x $')
    plt.plot(sol.t,  P,           label=r'$P  $')
    # plt.plot(sol.t,  rho,           label=r'$\rho  $')
    plt.legend(loc='best',fontsize=12)
    plt.xlim((1e-2,h))
    plt.show()
    plt.clf()

def find_final_gradient(j_over_j_crit):
    j                   = j_crit * j_over_j_crit
    T_initial           = T_hot
    vz_initial          = j/rho_hot
    dT_dz_initial       = -1e-6
    dvz_dz_initial      = 1e-6
    initial_conditions  = [dT_dz_initial, T_initial, dvz_dz_initial, vz_initial]
    stop_distance       = 10**4
    sol = solve_ivp(integrator, [0, stop_distance], initial_conditions, 
        dense_output=True, 
        events=[dip, bump],
        rtol=3e-14, #atol=[1e-9,1e-11,1e-9,1e-11],
        args=[j])
    return sol.y[0][-1]

def calculate_solution(j_over_j_crit):
    j                   = j_crit * j_over_j_crit
    T_initial           = T_hot
    vz_initial          = j/rho_hot
    dT_dz_initial       = -1e-6
    dvz_dz_initial      = 1e-6
    initial_conditions  = [dT_dz_initial, T_initial, dvz_dz_initial, vz_initial]
    stop_distance       = 10**4
    sol = solve_ivp(integrator, [0, stop_distance], initial_conditions, 
        dense_output=True, 
        events=[dip, bump],
        rtol=3e-14, #atol=[1e-9,1e-11,1e-9,1e-11],
        args=[j])
    return sol

def calculate_Q_cool(sol,j_over_j_crit):
    j           = j_crit * j_over_j_crit
    z           = sol.t
    T           = sol.y[1]
    vz          = sol.y[3]
    edot        = edot_cool_double_equilibrium(T, vz)
    return integrate.simps(edot,z)

def calculate_Hvisc(sol,j_over_j_crit):
    j           = j_crit * j_over_j_crit
    z           = sol.t
    vz          = sol.y[3]
    dvz_dz      = sol.y[2]
    dvx_dz      = dvx_dz_cos(z)
    visc        = Prandtl * (j/vz) * (f_nu * h**2 * dvx_dz + kappa_0 / (j/vz))
    return integrate.simps(visc*(dvx_dz**2 + 4/3. * dvz_dz**2),z)

def calculate_Work(sol,j_over_j_crit):
    j           = j_crit * j_over_j_crit
    z           = sol.t
    dT_dz       = sol.y[0]
    T           = sol.y[1]
    dvz_dz      = sol.y[2]
    vz          = sol.y[3]
    return integrate.simps(j*(dT_dz - dvz_dz*T/vz),z)

def plot_solution(sol,j_over_j_crit,name=None):
    j           = j_crit * j_over_j_crit
    z           = sol.t
    dT_dz       = sol.y[0]
    T           = sol.y[1]
    dvz_dz      = sol.y[2]
    vz          = sol.y[3]
    P           = T*(j/vz)
    rho         = (j/vz)
    vx          = vx_cos(z)
    dvx_dz      = dvx_dz_cos(z)
    d2vx_dz2    = d2vx_dz2_cos(z)
    
    dEdot_dlogT = (T/-dT_dz)*edot_cool_double_equilibrium(T, vz)
    dM_dlogT    = (T/-dT_dz)*rho

    cond        = (j/vz) * (f_nu * h**2 * dvx_dz + kappa_0 / (j/vz))
    visc        = Prandtl * (j/vz) * (f_nu * h**2 * dvx_dz + kappa_0 / (j/vz))
    dcond       = f_nu * h**2 * ((j/vz) * d2vx_dz2 - (j/vz**2) * dvx_dz * dvz_dz)
    dvisc       = Prandtl * (f_nu * h**2 * ((j/vz) * d2vx_dz2 - (j/vz**2) * dvx_dz * dvz_dz))
    d2T_dz2     = edot_cool_double_equilibrium(T, vz)/cond + j*vz/cond * (T/vz**2 * dvz_dz + dT_dz/(gamma-1)/vz) - dT_dz * dcond/cond - visc/cond * (dvx_dz**2 + (4/3.)*dvz_dz**2)
    d2vz_dz2    = (3./4.) * (j/visc) * (dT_dz/vz + dvz_dz*(1-T/vz**2)) - dvz_dz * dvisc/visc

    fig, ((ax1,ax3),(ax2,ax4)) = plt.subplots(2,2)
    ax1.plot(z,  T,color='k', label=r'$T  $')
    ax1.plot(z,  vz,          label=r'$v_z $')
    ax1.plot(z,  vx+0.5*vrel, label=r'$v_x $')
    ax1.plot(z,  P,           label=r'$P  $')

    ax2.semilogy(z,  2.5*j*dT_dz,   color='k',              label=r'$\frac{\gamma}{\gamma-1} j \frac{d T}{d z}$')
    ax2.semilogy(z,  dcond*dT_dz + cond*d2T_dz2,            color=cm.tab10.colors[0], label=r'$\frac{d}{d z} \left( \kappa \frac{d T}{d z} \right)$')
    ax2.semilogy(z,  -edot_cool_double_equilibrium(T, vz),  color=cm.tab10.colors[1], label=r'$-\mathcal{L} $')
    ax2.semilogy(z,  visc*(dvx_dz**2),                      color=cm.tab10.colors[2], label=r'$\mu \left(\frac{d v_x}{d z} \right)^2$')
    ax2.semilogy(z,  visc*((4/3.)*dvz_dz**2),               color=cm.tab10.colors[3], label=r'$\mu \frac{4}{3} \left(\frac{d v_z}{d z} \right)^2$')
    ax2.semilogy(z,  j*(dT_dz - dvz_dz*T/vz),               color=cm.tab10.colors[4], label=r'$v_z \frac{d P}{d z}$')

    ax2.semilogy(z,  -2.5*j*dT_dz,   color='k', ls='--')
    ax2.semilogy(z,  -(dcond*dT_dz + cond*d2T_dz2),         color=cm.tab10.colors[0], ls='--')
    ax2.semilogy(z,  edot_cool_double_equilibrium(T, vz),   color=cm.tab10.colors[1], ls='--')
    ax2.semilogy(z,  -visc*(dvx_dz**2),                     color=cm.tab10.colors[2], ls='--')
    ax2.semilogy(z,  -visc*((4/3.)*dvz_dz**2),              color=cm.tab10.colors[3], ls='--')
    ax2.semilogy(z,  -j*(dT_dz - dvz_dz*T/vz),              color=cm.tab10.colors[4], ls='--')

    ax3.loglog(T,  dEdot_dlogT,   label=r'$\frac{d \dot{E}}{d \log T}$')
    ax3.loglog(T,  dM_dlogT,      label=r'$\frac{d M}{d \log T}$')
    i_lo = np.argmin(np.abs(T-1.2*T_cold))
    i_hi = np.argmin(np.abs(T-0.8*T_hot))
    maximum = np.max([np.max(dEdot_dlogT[i_hi:i_lo]),np.max(dM_dlogT[i_hi:i_lo])])
    minimum = np.min([np.min(np.abs(dEdot_dlogT)[i_hi:i_lo]),np.min(np.abs(dM_dlogT)[i_hi:i_lo])])
    ax3.set_ylim((minimum,maximum))

    ax4.loglog(T,  2.5*j*dT_dz,   color='k',              label=r'$\frac{\gamma}{\gamma-1} j \frac{d T}{d z}$')
    ax4.loglog(T,  dcond*dT_dz + cond*d2T_dz2,            color=cm.tab10.colors[0], label=r'$\frac{d}{d z} \left( \kappa \frac{d T}{d z} \right)$')
    ax4.loglog(T,  -edot_cool_double_equilibrium(T, vz),  color=cm.tab10.colors[1], label=r'$-\mathcal{L} $')
    ax4.loglog(T,  visc*(dvx_dz**2),                      color=cm.tab10.colors[2], label=r'$\mu \left(\frac{d v_x}{d z} \right)^2$')
    ax4.loglog(T,  visc*((4/3.)*dvz_dz**2),               color=cm.tab10.colors[3], label=r'$\mu \frac{4}{3} \left(\frac{d v_z}{d z} \right)^2$')
    ax4.loglog(T,  j*(dT_dz - dvz_dz*T/vz),               color=cm.tab10.colors[4], label=r'$v_z \frac{d P}{d z}$')

    ax4.loglog(T,  -2.5*j*dT_dz,   color='k', ls='--')
    ax4.loglog(T,  -(dcond*dT_dz + cond*d2T_dz2),         color=cm.tab10.colors[0], ls='--')
    ax4.loglog(T,  edot_cool_double_equilibrium(T, vz),   color=cm.tab10.colors[1], ls='--')
    ax4.loglog(T,  -visc*(dvx_dz**2),                     color=cm.tab10.colors[2], ls='--')
    ax4.loglog(T,  -visc*((4/3.)*dvz_dz**2),              color=cm.tab10.colors[3], ls='--')
    ax4.loglog(T,  -j*(dT_dz - dvz_dz*T/vz),              color=cm.tab10.colors[4], ls='--')

    ax2.set_ylim((1e-4,1e2))
    ax4.set_ylim((1e-4,1e2))

    ax2.set_xlabel('z')
    ax4.set_xlabel('T')
    ax1.legend(loc='best',fontsize=6)
    ax2.legend(loc='upper right',fontsize=6,ncol=3)
    ax3.legend(loc='best',fontsize=6)
    ax4.legend(loc='lower right',fontsize=6,ncol=3)
    fig.set_size_inches(6.5,6.5)
    if name:
        plt.savefig(name+'.pdf',dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')
    return


dT_dz_initial       = -1e-6
dvz_dz_initial      = 1e-6

vrel                = 0.5
h                   = 1
f_nu                = 0.01
kappa_0             = 1e-2
beta_tcool          = 1.0
Prandtl             = 1.0e-2
t_cool_min          = 1e-1

beta_lo             = -2
beta_hi             = 1
density_contrast    = 100.
T_peak_over_T_cold  = density_contrast**(1/3.)

P_hot               = 1.0
T_cold              = 1.0/density_contrast
T_hot               = 1.0
T_peak              = T_peak_over_T_cold * T_cold
epsilon_T           = 0.005
rho_hot             = P_hot / T_hot
j_crit              = rho_hot * np.sqrt(T_hot)
t_cool_eff          = t_cool_min**beta_tcool * (h/vrel)**(1 - beta_tcool)
alpha_heat          = (np.log10(T_cold/T_peak)/np.log10(density_contrast) * (beta_lo - beta_hi)) - beta_hi
heating_coefficient = (T_cold/T_peak)**((beta_hi-beta_lo)*(1.0 + (np.log10(T_cold/T_peak)/np.log10(density_contrast))))

Prs     = np.logspace(-2,0,9)
f_nus   = np.logspace(-2,-0.5,7)
js = np.zeros((len(Prs),len(f_nus)))
Qs = np.zeros((len(Prs),len(f_nus)))
Hs = np.zeros((len(Prs),len(f_nus)))
Ws = np.zeros((len(Prs),len(f_nus)))
for i_Pr in range(len(Prs)):
    for i_f_nu in range(len(f_nus)):
        Prandtl             = Prs[i_Pr]
        f_nu                = f_nus[i_f_nu]
        try:
            # target  = scipy.optimize.root_scalar(find_final_gradient, bracket = [0.1,1], rtol=1e-4)
            j = 1.0
            final_gradient = find_final_gradient(j)
            positive = 1
            factor = 0.9
            while positive > 0:
                j *= factor
                next_final_gradient = find_final_gradient(j)
                print(next_final_gradient, end='\r')
                positive = next_final_gradient*final_gradient
                final_gradient = next_final_gradient
            target = scipy.optimize.root_scalar(find_final_gradient, bracket = [j,j/factor], rtol=1e-8)#, xtol=3e-14, rtol=3e-14)
            sol     = calculate_solution(target.root)
            js[i_Pr,i_f_nu] = target.root
            Qs[i_Pr,i_f_nu] = calculate_Q_cool(sol,target.root)
            Hs[i_Pr,i_f_nu] = calculate_Hvisc(sol,target.root)
            Ws[i_Pr,i_f_nu] = calculate_Work(sol,target.root)
            print(target.root, calculate_Q_cool(sol,target.root)/target.root , calculate_Hvisc(sol,target.root)/target.root, calculate_Work(sol,target.root)/target.root)
        except:
            print("nothing for", Prandtl, f_nu)


for i in range(len(f_nus)):
    plt.axhline(2.5*0.99, color='k',label=r"$\frac{\gamma}{\gamma-1} \Delta T$")
    plt.axhline((2.5*0.99+0.5*vrel**2),color='grey',label=r"$\frac{\gamma}{\gamma-1} \Delta T + \frac{1}{2} v_{\rm rel}^2$")
    plt.loglog(Prs,(Qs[:,i]-Hs[:,i]-Ws[:,i])/js[:,i], 'o--', color='k',label=r"$Q - H - W$")
    plt.loglog(Prs,Qs[:,i]/js[:,i], 'o-', color='tab:blue',label=r"$Q$")
    plt.loglog(Prs,Hs[:,i]/js[:,i], 'o-', color='tab:orange',label=r"$H$")
    plt.loglog(Prs,Ws[:,i]/js[:,i], 'o-', color='tab:green',label=r"$W$")
    plt.loglog(Prs,-Ws[:,i]/js[:,i],'o--', color='tab:green')
    plt.legend(loc='best', ncol=2, fontsize=8)
    plt.title(r"$f_\nu = "+str(np.round(f_nus[i],3))+r" \quad v_{\rm rel} ="+str(np.round(vrel,3))+r"$")
    plt.xlabel(r"$Prandtl = \mu/\kappa$")
    plt.savefig("integrated_quantities_versus_Pr_with_f_nu"+str(np.round(f_nus[i],3))+"_vrel"+str(np.round(vrel,3))+".pdf",dpi=200)
    plt.clf()

for i in range(len(Prs)):
    plt.axhline(2.5*0.99, color='k', label=r"$\frac{\gamma}{\gamma-1} \Delta T$")
    plt.axhline((2.5*0.99+0.5*vrel**2),color='grey', label=r"$\frac{\gamma}{\gamma-1} \Delta T + \frac{1}{2} v_{\rm rel}^2$")
    plt.loglog(f_nus,(Qs[i]-Hs[i]-Ws[i])/js[i], 'o--', color='k', label=r"$Q - H - W$")
    plt.loglog(f_nus,Qs[i]/js[i], 'o-', color='tab:blue', label=r"$Q$")
    plt.loglog(f_nus,Hs[i]/js[i], 'o-', color='tab:orange', label=r"$H$")
    plt.loglog(f_nus,Ws[i]/js[i], 'o-', color='tab:green', label=r"$W$")
    plt.loglog(f_nus,-Ws[i]/js[i],'o--', color='tab:green')
    plt.legend(loc='best', ncol=2, fontsize=8)
    plt.xlabel(r"$f_\nu$")
    plt.title(r"$Pr = "+str(np.round(Prs[i],2))+r" \quad v_{\rm rel} ="+str(np.round(vrel,3))+r"$")
    plt.savefig("integrated_quantities_versus_f_nu_with_Pr"+str(np.round(Prs[i],2))+"_vrel"+str(np.round(vrel,3))+".pdf",dpi=200)
    plt.clf()


Prandtl = 1.0e-2
f_nu = 1.0e-2
vrels = np.logspace(-1.25,0.25,7)[::-1]
j_vrels_Pr2_fnu2 = np.zeros_like(vrels)
W_vrels_Pr2_fnu2 = np.zeros_like(vrels)
H_vrels_Pr2_fnu2 = np.zeros_like(vrels)
Q_vrels_Pr2_fnu2 = np.zeros_like(vrels)
for i_vrel,vrel in enumerate(vrels):
    factor = 0.9
    j = 1.0
    final_gradient = find_final_gradient(j)
    positive = 1
    while positive > 0:
        j *= factor
        next_final_gradient = find_final_gradient(j)
        print(next_final_gradient,end='\r')
        positive = next_final_gradient*final_gradient
        final_gradient = next_final_gradient

    target = scipy.optimize.root_scalar(find_final_gradient, bracket = [j,j/factor], rtol=1e-8)#, xtol=3e-12, rtol=3e-12)
    sol = calculate_solution(target.root)
    Q_vrels_Pr2_fnu2[i_vrel] = calculate_Q_cool(sol,target.root)
    H_vrels_Pr2_fnu2[i_vrel] = calculate_Hvisc(sol,target.root)
    W_vrels_Pr2_fnu2[i_vrel] = calculate_Work(sol,target.root)
    j_vrels_Pr2_fnu2[i_vrel] = target.root
    print("vrel=%.3f    Q=%.3f    j=%.3f    (Q-H-W)/j=%.3f" %(vrel, Q_vrels_Pr2_fnu2[i_vrel], j_vrels_Pr2_fnu2[i_vrel], (Q_vrels_Pr2_fnu2[i_vrel]-H_vrels_Pr2_fnu2[i_vrel]-W_vrels_Pr2_fnu2[i_vrel])/j_vrels_Pr2_fnu2[i_vrel]))
    plot_solution(sol,j_vrels_Pr2_fnu2[i_vrel],"Pr1e-2_fnu1e-2_logvrel_"+str(np.log10(vrel)))


fig,ax = plt.subplots(1,1)
ax.loglog(vrels, 2.5*0.99*j_vrels_Pr2_fnu2, 'o-', label=r'$\frac{\gamma}{\gamma-1} \Delta T j$', color='k')
ax.loglog(vrels, Q_vrels_Pr2_fnu2, 'o-', label=r'$Q$')
ax.loglog(vrels, H_vrels_Pr2_fnu2, 'o-', label=r'$H$')
ax.loglog(vrels, W_vrels_Pr2_fnu2, 'o-', label=r'$W$')
ax.loglog(vrels, vrels**0.75, color='grey', ls=':', lw=0.5, label=r"$v_{\rm rel}^{3/4}$")
ax.loglog(vrels, 0.5 * j_vrels_Pr2_fnu2 * vrels**2, color='grey', ls='--', lw=0.5, label=r"$\frac{1}{2} j v_{\rm rel}^2$")
ax.legend(loc='best')
ax.set_xlabel(r"$v_{\rm rel}$")
ax.set_title(r"$Pr = "+str(np.round(Prandtl,2))+r" \quad f_\nu = "+str(np.round(f_nu,3))+r"$")
plt.savefig("integrated_quantities_versus_vrel_with_f_nu"+str(np.round(f_nu,3))+"_Pr"+str(np.round(Prandtl,2))+".pdf")
plt.close('all')



Prandtl = 1.0e-1
f_nu = 1.0e-1
vrels = np.logspace(-1.25,0.25,7)[::-1]
j_vrels_Pr1_fnu1 = np.zeros_like(vrels)
W_vrels_Pr1_fnu1 = np.zeros_like(vrels)
H_vrels_Pr1_fnu1 = np.zeros_like(vrels)
Q_vrels_Pr1_fnu1 = np.zeros_like(vrels)
for i_vrel,vrel in enumerate(vrels):
    factor = 0.9
    j = 1.0
    final_gradient = find_final_gradient(j)
    positive = 1
    while positive > 0:
        j *= factor
        next_final_gradient = find_final_gradient(j)
        print(next_final_gradient,end='\r')
        positive = next_final_gradient*final_gradient
        final_gradient = next_final_gradient

    target = scipy.optimize.root_scalar(find_final_gradient, bracket = [j,j/factor], rtol=1e-8)#, xtol=3e-12, rtol=3e-12)
    sol = calculate_solution(target.root)
    Q_vrels_Pr1_fnu1[i_vrel] = calculate_Q_cool(sol,target.root)
    H_vrels_Pr1_fnu1[i_vrel] = calculate_Hvisc(sol,target.root)
    W_vrels_Pr1_fnu1[i_vrel] = calculate_Work(sol,target.root)
    j_vrels_Pr1_fnu1[i_vrel] = target.root
    print("vrel=%.3f    Q=%.3f    j=%.3f    (Q-H-W)/j=%.3f" %(vrel, Q_vrels_Pr1_fnu1[i_vrel], j_vrels_Pr1_fnu1[i_vrel], (Q_vrels_Pr1_fnu1[i_vrel]-H_vrels_Pr1_fnu1[i_vrel]-W_vrels_Pr1_fnu1[i_vrel])/j_vrels_Pr1_fnu1[i_vrel]))
    plot_solution(sol,j_vrels_Pr1_fnu1[i_vrel],"Pr1e-1_fnu1e-1_logvrel_"+str(np.log10(vrel)))


fig,ax = plt.subplots(1,1)
ax.loglog(vrels, 2.5*0.99*j_vrels_Pr1_fnu1, 'o-', label=r'$\frac{\gamma}{\gamma-1} \Delta T j$', color='k')
ax.loglog(vrels, Q_vrels_Pr1_fnu1, 'o-', label=r'$Q$')
ax.loglog(vrels, H_vrels_Pr1_fnu1, 'o-', label=r'$H$')
ax.loglog(vrels, W_vrels_Pr1_fnu1, 'o-', label=r'$W$')
ax.loglog(vrels, vrels**0.75, color='grey', ls=':', lw=0.5, label=r"$v_{\rm rel}^{3/4}$")
ax.loglog(vrels, 0.5 * j_vrels_Pr1_fnu1 * vrels**2, color='grey', ls='--', lw=0.5, label=r"$\frac{1}{2} j v_{\rm rel}^2$")
ax.legend(loc='best')
ax.set_xlabel(r"$v_{\rm rel}$")
ax.set_title(r"$Pr = "+str(np.round(Prandtl,2))+r" \quad f_\nu = "+str(np.round(f_nu,3))+r"$")
plt.savefig("integrated_quantities_versus_vrel_with_f_nu"+str(np.round(f_nu,3))+"_Pr"+str(np.round(Prandtl,2))+".pdf")
plt.close('all')