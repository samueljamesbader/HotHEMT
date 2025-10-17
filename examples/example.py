from hothemt import specific_sim_work_dir, um, W, mm, K, m
from hothemt.hemt import HEMT

def run_example(do_plots=True):

    # Where should output go
    with specific_sim_work_dir("output/hemt3d_example"):

        # Define the device
        h=HEMT(

            # Device gate length, spacing, and contact size
            lg=.15*um, lgd=.5*um, lgs=.5*um, lc=.5*um,

            # Device row geometry (4 finger, 3um wide)
            n_f=4, w_f=3*um, rows=1, row_pitch=7*um,

            # Heater definition (200nm starting at drain edge of gate)
            L_h=.2*um, L_ho=0*um,

            # Layer thicknesses and simulation domain
            t_GaN=.5*um, t_Rel=.5*um, t_mesa=.2*um, l_chipx=10*um, l_chipy=10*um,

            # Conditions
            P_per_W=2*W/mm, T_A=300*K,

            # Mesh density (increase for more density... up to 1 is typical)
            alpha_mesh=.3

        )

        # Optional: force (re)mesh and show the mesh 
        h.get_mesh_path(force_remesh=True, show_gui=do_plots)

        # Forcibly (re)run the simulation
        r3dsim=h.get_siminfo(force_remesh=False,force_resim=True)['RthWEff3D Heater Avg [K.mm/W]'] * (K*mm/W)
        print(f"Simulated RthWEff3D: {r3dsim/(K*mm/W):.2f} Kmm/W")

        if do_plots:
            # Plot the results
            h.plot_temp_profile3d()
            h.visualize_solution_3d()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the HEMT 3D example')
    parser.add_argument('--no_plots', action='store_true', help="Don't generate plots")
    namespace = parser.parse_args()
    run_example(do_plots=(not namespace.no_plots))