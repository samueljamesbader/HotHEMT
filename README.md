# HotHEMT
This package uses open-source tools to simulate thermal resistances for HEMT structures.  Specifically [gmsh](https://gmsh.info/) is used to create 3-D meshes and [SFEpy](https://sfepy.org) is used to solve the Finite-Element problem on the mesh.

The phyiscs is straightforward: self-heating is handled by simply assuming uniform heaters of specified size at the drain-edge of the gate, and Fourier's Law is solved (with potentially temperature-dependent thermal conductivities) on the user-specified geometry for the heat flows.

Have a look at [example.py](examples/example.py) in `examples/` to start producing 3D simulations like this and see how much your HEMT will heat up:

![Example3D](doc/Example3D.png)


## Installation Notes
To install the latest directly from GitHub, use
```
pip install "git+https://github.com/samueljamesbader/hothemt.git@main"
```

If on linux, `gmsh` may also require the following

```
sudo apt-get install -y libglu1-mesa
```

## That's it
This was a fun personal exploration, no publications or anything to cite at the moment.  But if you find this useful in your research, would appreciate a star or shout-out or just a thanks! Cheers, Sam.