from hothemt import specific_sim_work_dir
from hothemt.persistent_id import DBBackedIDMixin


def cli_vis():
    import argparse
    parser = argparse.ArgumentParser(description='Plot temperature profile of a HEMT 3D simulation')
    parser.add_argument('plot', type=str, help='type of plot: "prof" or "3D" or "mesh" or "regs" or "print')
    parser.add_argument('sim_id', type=int, help='Simulation ID')
    parser.add_argument('--sim_work_dir', type=str, default=None, help='Simulation work directory')
    args = parser.parse_args()
    with specific_sim_work_dir(args.sim_work_dir):
        h=DBBackedIDMixin.from_simid(args.sim_id)
        print(h)
        if args.plot.lower() in ['prof', 'profile']:
            h.plot_temp_profile3d()
        elif args.plot.lower() in ['3d']:
            h.visualize_solution_3d()
        elif args.plot.lower() in ['mesh']:
            h.visualize_mesh()
        elif args.plot.lower() in ['regs', 'regions']:
            h.visualize_regs_3d()
        elif args.plot.lower() in ['print']:
            info=h.get_siminfo()
            print(info['hemt_parameters'])