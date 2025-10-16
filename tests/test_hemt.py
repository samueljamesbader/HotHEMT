from typing import Any
from hothemt.analytical import analytical_RthWEff2D, analytical_RthWEff3D
import numpy as np
from hothemt import specific_sim_work_dir, um, mm, K, W, m
from hothemt.hemt import HEMT

# some generic HEMT parameters to use for tests
# don't really matter since tests are single-finger and not allowing heat flow out metals
common: dict[str,Any]= dict(
    lgs=.5*um,
    lg=.2*um,
    lgd=.5*um,
    lc=1*um,
    L_ho=0,)
# All numbers in these tests are just made up to be reasonable values given the literature
# or to test specific analytically computable conditions

def test_analytical_2d3d():
    """ Check analytical 3D expression against simplier 2D expression.
    
    We set l_chip=w_f=L_p to get an effectively 2D problem,
    and k_GaN=k_Rel=k_Sub for the simplified expression.
    """
    with specific_sim_work_dir("output/output_test/test_myrtus_hemt_analytical_2d3d", clear=True):
        h=HEMT(
            n_f=1, **common, w_f=2*um,
            rows=1, row_pitch=7*um,
            L_h=0.2*um, t_GaN=0.5*um, t_Rel=.4*um, t_Sub=.1*um, l_chip=2*um,
            k300_GaN=130*W/(m*K), k300_Rel=130*W/(m*K), k300_Sub=130*W/(m*K),
            nthr_GaN=0, nthr_Rel=0, nthr_Sub=0, h_bot=1e9*W/(K*m**2)
        )
        r2d=analytical_RthWEff2D(h)
        r3d=analytical_RthWEff3D(h)

        assert np.isclose(r2d, 9.16*K*mm/W, rtol=1e-2)
        assert np.isclose(r2d, r3d, rtol=1e-4)

def test_hemt3d(show_gui:bool=False):
    """ Compare analytical 3D expression against 3D simulation.
    
    Only valid for n_f=1.
    """
    
    with specific_sim_work_dir("output/output_test/test_myrtus_hemt3d", clear=True):
        # Baseline case
        # Match t_Sub to t_Rel to use analytical
        # Analytical 9.83 Kmm/W
        h=HEMT(
            n_f=1, **common, w_f=2*um,
            rows=1, row_pitch=7*um,
            L_h=0.2*um, t_GaN=0.5*um, t_Rel=.4*um, t_Sub=.1*um, l_chip=10*um, h_bot=1e9*W/(K*m**2),
            k300_GaN=150*W/(m*K),k300_Rel=10*W/(m*K),k300_Sub=10*W/(m*K),
            nthr_GaN=0, nthr_Rel=0, nthr_Sub=0,
        )
        meshpath1=h.get_mesh_path(force_remesh=True, show_gui=show_gui)
        r3dsim=h.get_siminfo(force_resim=True)['RthWEff3D Heater Avg [K.mm/W]'] * (K*mm/W)
        r3dcal=analytical_RthWEff3D(h)
        assert np.isclose(r3dcal, 9.83*K*mm/W, rtol=1e-2)
        assert np.isclose(r3dsim, r3dcal, rtol=.05), f"Simulated {r3dsim}, calculated {r3dcal}"

        # Baseline but match t_GaN to t_Rel instead 
        # Analytical 9.83 Kmm/W
        h=HEMT(
            n_f=1, **common, w_f=2*um,
            rows=1, row_pitch=7*um,
            L_h=0.2*um, t_GaN=0.1*um, t_Rel=.4*um, t_Sub=.5*um, l_chip=10*um, h_bot=1e9*W/(K*m**2),
            k300_GaN=150*W/(m*K),k300_Rel=150*W/(m*K),k300_Sub=10*W/(m*K),
            nthr_GaN=0, nthr_Rel=0, nthr_Sub=0, dont_mesh_contacts=True,
        )
        h.get_mesh_path(force_remesh=True, show_gui=show_gui)
        r3dsim=h.get_siminfo(force_resim=True)['RthWEff3D Heater Avg [K.mm/W]'] * (K*mm/W)
        r3dcal=analytical_RthWEff3D(h)
        assert np.isclose(r3dcal, 9.83*K*mm/W, rtol=1e-2)
        assert np.isclose(r3dsim, r3dcal, rtol=.05), f"Simulated {r3dsim}, calculated {r3dcal}"

        # Baseline but re-use the mesh for different material properties
        # Make sure mesh is reused but analytical value changes
        # Analytical 13.61 Kmm/W
        h=HEMT(
            n_f=1, **common, w_f=2*um,
            rows=1, row_pitch=7*um,
            L_h=0.2*um, t_GaN=0.5*um, t_Rel=.4*um, t_Sub=.1*um, l_chip=10*um, h_bot=1e9*W/(K*m**2),
            k300_GaN=100*W/(m*K),k300_Rel=10*W/(m*K),k300_Sub=10*W/(m*K),
            nthr_GaN=0, nthr_Rel=0, nthr_Sub=0,
        )
        meshpath2=h.get_mesh_path(force_remesh=False, show_gui=show_gui)
        assert meshpath1==meshpath2, "Mesh should be reused!"
        r3dsim=h.get_siminfo(force_resim=True)['RthWEff3D Heater Avg [K.mm/W]'] * (K*mm/W)
        r3dcal=analytical_RthWEff3D(h)
        assert np.isclose(r3dcal, 13.61*K*mm/W, rtol=1e-2)
        assert np.isclose(r3dsim, r3dcal, rtol=.05)

        # Baseline but try a wider device
        # Analytical 13.99 Kmm/W
        h=HEMT(
            n_f=1, **common, w_f=4*um,
            rows=1, row_pitch=7*um,
            L_h=0.2*um, t_GaN=0.5*um, t_Rel=.4*um, t_Sub=.1*um, l_chip=6*um, h_bot=1e9*W/(K*m**2),
            k300_GaN=150*W/(m*K),k300_Rel=10*W/(m*K),k300_Sub=10*W/(m*K),
            nthr_GaN=0, nthr_Rel=0, nthr_Sub=0, dont_mesh_contacts=True
        )
        h.get_mesh_path(force_remesh=True, show_gui=show_gui)
        r3dsim=h.get_siminfo(force_resim=True)['RthWEff3D Heater Avg [K.mm/W]'] * (K*mm/W)
        r3dcal=analytical_RthWEff3D(h)
        assert np.isclose(r3dcal, 13.99*K*mm/W, rtol=1e-2)
        assert np.isclose(r3dsim, r3dcal, rtol=.05)

        # Baseline but try a thinner Gan layer and smaller l_chip
        # Analytical 22.85 Kmm/W
        h=HEMT(
            n_f=1, **common, w_f=2*um,
            rows=1, row_pitch=7*um,
            L_h=0.2*um, t_GaN=0.1*um, t_Rel=.4*um, t_Sub=.1*um, l_chip=3*um, h_bot=1e9*W/(K*m**2),
            k300_GaN=150*W/(m*K),k300_Rel=10*W/(m*K),k300_Sub=10*W/(m*K),
            nthr_GaN=0, nthr_Rel=0, nthr_Sub=0, dont_mesh_contacts=True,
        )
        h.get_mesh_path(force_remesh=True, show_gui=show_gui)
        r3dsim=h.get_siminfo(force_resim=True)['RthWEff3D Heater Avg [K.mm/W]'] * (K*mm/W)
        r3dcal=analytical_RthWEff3D(h)
        assert np.isclose(r3dcal, 22.85*K*mm/W, rtol=1e-2)
        assert np.isclose(r3dsim, r3dcal, rtol=.05)

        # Baseline but test smaller heat generation region down to 50nm
        # Analytical 12.73 Kmm/W
        h=HEMT(
            n_f=1, **common, w_f=2*um,
            rows=1, row_pitch=7*um,
            L_h=0.05*um, t_GaN=0.5*um, t_Rel=.4*um, t_Sub=.1*um, l_chip=10*um, h_bot=1e9*W/(K*m**2),
            k300_GaN=150*W/(m*K),k300_Rel=10*W/(m*K),k300_Sub=10*W/(m*K),
            nthr_GaN=0, nthr_Rel=0, nthr_Sub=0, dont_mesh_contacts=True
        )
        h.get_mesh_path(force_remesh=True, show_gui=show_gui)
        r3dsim=h.get_siminfo(force_resim=True)['RthWEff3D Heater Avg [K.mm/W]'] * (K*mm/W)
        r3dcal=analytical_RthWEff3D(h)
        assert np.isclose(r3dcal, 12.73*K*mm/W, rtol=1e-2)
        assert np.isclose(r3dsim, r3dcal, rtol=.05)

        
        # Baseline but test T-dependent thermal conductivity ### RERUN
        # Analytical 12.30 Kmm/W
        h=HEMT(
            n_f=1, **common, w_f=2*um,
            rows=1, row_pitch=7*um,
            L_h=0.2*um, t_GaN=.5*um, t_Rel=.4*um, t_Sub=.1*um, l_chip=10*um, h_bot=1e10*W/(K*m**2),
            k300_GaN=150*W/(m*K),k300_Rel=10*W/(m*K),k300_Sub=10*W/(m*K),
            nthr_GaN=1.3, nthr_Rel=1.3, nthr_Sub=1.3,
            P_per_W=10*W/mm, T_A=300*K, dont_mesh_contacts=True
        )
        h.get_mesh_path(force_remesh=True, show_gui=show_gui)
        r3dsim=h.get_siminfo(force_resim=True)['RthWEff3D Heater Avg [K.mm/W]'] * (K*mm/W)
        r3dcal=analytical_RthWEff3D(h)
        assert np.isclose(r3dcal, 12.30*K*mm/W, rtol=1e-2)
        assert np.isclose(r3dsim, r3dcal, rtol=.05), f"Simulated {r3dsim}, calculated {r3dcal}"

        
        # Baseline but test smaller heat generation region down to 5nm
        # Also reduce width and l_chip to keep mesh smaller for quick tests
        # Analytical 14.84Kmm/W
        h=HEMT(
            n_f=1, **common, w_f=.5*um,
            rows=1, row_pitch=7*um,
            L_h=0.005*um, t_GaN=0.5*um, t_Rel=.4*um, t_Sub=.1*um, l_chip=3*um, h_bot=1e9*W/(K*m**2),
            k300_GaN=150*W/(m*K),k300_Rel=10*W/(m*K),k300_Sub=10*W/(m*K),
            nthr_GaN=0, nthr_Rel=0, nthr_Sub=0, dont_mesh_contacts=True
        )
        h.get_mesh_path(force_remesh=True, show_gui=show_gui)
        r3dsim=h.get_siminfo(force_resim=True)['RthWEff3D Heater Avg [K.mm/W]'] * (K*mm/W)
        r3dcal=analytical_RthWEff3D(h)
        assert np.isclose(r3dcal, 14.84*K*mm/W, rtol=1e-2)
        assert np.isclose(r3dsim, r3dcal, rtol=.05), f"Simulated {r3dsim}, calculated {r3dcal}"

        # Try one with infinite half-space spreading...
        # Make heat area large enough and l_chip small enough that boundary matters
        # The boundary contribution should be ~1/2*pi*k*(l_chip/2), and this gives ~20% importance
        l_chip_um=20
        h=HEMT(
            n_f=1, **{k:v for k,v in common.items() if k!='lgd'}, w_f=3*um, lgd=3*um,
            rows=1, row_pitch=7*um,
            L_h=3*um, t_GaN=0.5*um, t_Rel=.5*um, t_Sub=(l_chip_um/2-1-.5)*um, l_chip=l_chip_um*um,
            #h_bot=1e10*W/(K*m**2),
            k300_GaN=150*W/(m*K),k300_Rel=150*W/(m*K),k300_Sub=150*W/(m*K),
            nthr_GaN=0, nthr_Rel=0, nthr_Sub=0, dont_mesh_contacts=True,
        )
        show_gui=True
        #h.get_mesh_path(force_remesh=True, show_gui=show_gui)
        r3dsim=h.get_siminfo(force_resim=True)['RthWEff3D Heater Avg [K.mm/W]'] * (K*mm/W)
        r3dcal=analytical_RthWEff3D(h)
        print(f"Calculated RthWEff3D: {r3dcal/(K*mm/W):.2f} K mm/W")
        assert np.isclose(r3dcal, 3.15*K*mm/W, rtol=1e-2)
        assert np.isclose(r3dsim, r3dcal, rtol=.05), f"Simulated {r3dsim}, calculated {r3dcal}"
        

if __name__ == "__main__":
    test_analytical_2d3d()
    test_hemt3d(show_gui=False)