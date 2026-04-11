from itertools import chain
import json
import os
import subprocess
import sysconfig
from dataclasses import asdict, dataclass

from pathlib import Path
from typing import Annotated, Any, Callable, Optional, Union

import numpy as np

from hothemt.persistent_id import DBBackedIDMixin
from hothemt import get_default_sim_work_dir, specific_sim_work_dir, um, m, mm, K, W, time_it, get_mesh_dir, get_info_dir, get_state_dir
from hothemt.sfepy_util import probe_cache_context


@dataclass(kw_only=True)
class HEMT(DBBackedIDMixin):

    # Define the geometry of the HEMT row
    n_f: int
    """number of fingers"""
    w_f: float
    """finger width"""
    lg: float
    """gate length"""
    lgs: float
    """gate to source length"""
    lgd: float
    """gate to drain length"""
    lc: float
    """contact metal length"""

    # Define the arrangement of multiple rows and columns
    rows: int
    """number of rows of fingers (tiling in y-direction)"""
    row_pitch: float
    """pitch of rows of fingers (perpendicular to current flow)"""
    columns: int = 1
    """number of columns of finger arrays (tiling in x-direction)"""
    column_pitch: float = 0
    """pitch of columns of finger arrays (along current-flow direction)"""

    # Define the heat source   
    L_h: float
    """heat source length"""
    L_ho: float
    """heat source offset (=0 means starts at drain edge of gate, =-L_h/2 means centered on drain edge of gate)"""

    # Define the layer thicknesses and chip size
    t_GaN: float
    """thickness of GaN layer"""
    t_Rel: float
    """thickness of relaxation layer"""
    l_chipx: float
    """dimension of the chiplet in x direction"""
    l_chipy: float
    """dimension of the chiplet in y direction"""
    t_mesa: float = 0
    """thickness of the mesa (must be < t_GaN)"""
    t_Sub: float = None # type: ignore
    """thickness of substrate layer (defaults to max(l_chipx,l_chipy)/2 - t_GaN - t_Rel) """

    # Define the material parameters and boundary conditions
    k300_GaN: Annotated[float,"NotMesh"] = 150 * W/(m*K)
    """thermal conductivity of GaN """
    nthr_GaN: Annotated[float,"NotMesh"] = 1.3
    """temperature exponent of GaN thermal conductivity """
    k300_Rel: Annotated[float,"NotMesh"] = 10 * W/(m*K)
    """thermal conductivity of relaxation layer """
    nthr_Rel: Annotated[float,"NotMesh"] = 0
    """temperature exponent of relaxation layer thermal conductivity """
    k300_Sub: Annotated[float,"NotMesh"] = 150 * W/(m*K)
    """thermal conductivity of substrate layer """
    nthr_Sub: Annotated[float,"NotMesh"] = 1.3
    """temperature exponent of substrate thermal conductivity """
    h_bot: Annotated[Optional[float],"NotMesh"] = None
    """if specified, heat transfer coefficient at the bottom, with zero contribution from Subs sidewalls.
        Otherwise Subs boundaries will be approximated with an infinite spreading TBR."""
    h_con: Annotated[float,"NotMesh"] = 0 # 1e8 * W/(m**2*K)
    """heat transfer coefficient at the contacts """
    h_gat: Annotated[float,"NotMesh"] = 0 # 1e8 * W/(m**2*K)
    """heat transfer coefficient at the gate """

    # Define the run conditions
    T_A: Annotated[float,"NotMesh"] = 300 * K
    """ambient temperature """
    P_per_W: Annotated[float,"NotMesh"] = 1 * W/mm
    """power dissipation per unit width """


    # Extra mesh options
    alpha_mesh: float = .7
    """mesh density factor (more alpha -> more dense) """
    aniso_mesh: bool = True
    """use anisotropic mesh sizing """
    dont_mesh_contacts: bool = False
    """if True, don't indicate the contact metals in the mesh (only allowed if h_con=0)"""
    dont_mesh_gates: bool = False
    """if True, don't indicate the gates in the mesh (only allowed if h_gat=0)"""
    half_x: bool = True
    """if True, only simulate x>0 and use mirror symmetry to capture the rest"""
    is_active: Annotated[Optional[Callable[[int, int], bool]], "NotMesh", "NotSim"] = None
    """optional callable(row, col) -> bool indicating whether the island at integer coordinate
    (row, col) is active (centered at (0,0) when a center exists). Only abs(row) and abs(col)
    are passed — negative coordinates are assumed symmetric with positive ones.
    If None (default), all islands are active."""

    # Initialization and checks
    def __post_init__(self):
        super().__post_init__()
        if self.t_Sub is None:
            self.t_Sub = max(self.l_chipx,self.l_chipy)/2 - self.t_GaN - self.t_Rel
        if self.dont_mesh_contacts and self.h_con!=0:
            raise ValueError("dont_mesh_contacts=True only allowed if h_con=0")
        if self.dont_mesh_gates and self.h_gat!=0:
            raise ValueError("dont_mesh_gates=True only allowed if h_gat=0")

    def _island_is_active(self, icolx: int, isrcy: int) -> bool:
        """Return whether the island at column index icolx and row index isrcy is active.

        Converts grid indices to centered (row, col) coordinates and delegates to
        self.is_active(abs(row), abs(col)) if provided, otherwise returns True.
        """
        if self.is_active is None:
            return True
        row_coord = isrcy - (self.rows - 1) // 2
        col_coord = icolx - (self.columns - 1) // 2
        return bool(self.is_active(abs(row_coord), abs(col_coord)))

    # Properties for access to geometry parameters, enabling subclasses to override parameterization
    @property
    def Lcs(self): return self.lc
    @property
    def Lcd(self): return self.lc
    @property
    def Lgs(self): return self.lgs
    @property
    def Lgd(self): return self.lgd

    # Geometry calculations
    @property
    def Lhsh(self):
        """length from one heater to next when a source contact is in-between"""
        return self.Lcs + 2*(self.Lgs + self.lg + self.L_ho + self.L_h/2 )
    
    @property
    def Lhdh(self):
        """length from one heater to next when a drain contact is in-between"""
        return self.Lcd + 2*(self.Lgd - self.L_ho - self.L_h/2)

    def iter_heaters(self, crop_symmetric: bool = True, active_only: bool = False):
        """Iterate over the the heaters.

        Args:
            crop_symmetric: if True, crop results and only return heaters in the symmetrically representative region
            active_only: if True, skip heaters belonging to inactive islands (see is_active)

        Yields:
            (xlef,xrit,yfore,yback) of each heater
        """
        Lhsh = self.Lhsh
        Lhdh = self.Lhdh
        if (self.n_f==1) and self.half_x:
            assert self.h_gat==0, "Asymmetric single finger only allowed with h_gat=0"
            if self.Lhsh!=self.Lhdh:
                assert self.t_mesa==0, "Asymmetric single finger only allowed with no mesa"
                assert self.h_con==0, "Asymmetric single finger only allowed with h_con=0"
        else:
            if (self.n_f%2!=0) and self.half_x:
                assert Lhsh==Lhdh, "Allowed cases for symmetric simplification: "\
                    "single finger, even number of fingers, or equal source and drain lengths with heat centered on gate"
                assert self.h_gat==0 or (self.L_ho+self.L_h/2)==-self.lg/2, "Allowed cases for symmetric simplification: "\
                    "single finger, even number of fingers, or equal source and drain lengths with heat centered on gate"

        offset = ((self.n_f//2)*Lhdh+((self.n_f-1)//2)*Lhsh)/2
        for icolx in range(self.columns):
            col_offset = self.column_pitch*icolx - (self.columns-1)*self.column_pitch/2
            for isrcx in range(self.n_f):
                xcen = ((isrcx+1)//2)*Lhdh+(isrcx//2)*Lhsh - offset + col_offset
                xlef = xcen - self.L_h/2
                xrit = xcen + self.L_h/2
                assert xrit<=self.l_chipx/2, "Chip too small for the number of fingers!"

                for isrcy in range(self.rows):
                    ycen = self.row_pitch*isrcy - (self.rows-1)*self.row_pitch/2
                    yfore = ycen - self.w_f/2
                    yback = ycen + self.w_f/2
                    assert yback<=self.l_chipy/2, "Chip too small for the number of rows!"

                    if active_only and not self._island_is_active(icolx, isrcy): continue

                    if crop_symmetric:
                        if self.half_x:
                            if xrit < 0: continue
                            xlef = max(xlef,0)
                        if yback < 0: continue
                        yfore = max(yfore,0)

                    yield (xlef, xrit, yfore, yback)

    def iter_contacts(self, crop_symmetric: bool, even_if_not_meshing: bool=False, active_only: bool = False):
        """Iterate over the the contacts.

        By default, will yield no results if dont_mesh_contacts=True,
        but this can be overridden with even_if_not_meshing=True.

        Args:
            crop_symmetric: if True, crop results and only return contacts in the symmetrically representative region
            even_if_not_meshing: if True, yield the contact positions even if dont_mesh_contacts=True
            active_only: if True, skip contacts belonging to inactive islands (see is_active)
        Yields:
            (xlef,xrit,yfore,yback) of each contact
        """
        if self.dont_mesh_contacts and (not even_if_not_meshing): return
        Lhsh = self.Lhsh
        Lhdh = self.Lhdh
        offset = ((self.n_f//2)*Lhdh+((self.n_f-1)//2)*Lhsh)/2
        for icolx in range(self.columns):
            col_offset = self.column_pitch*icolx - (self.columns-1)*self.column_pitch/2
            contact_to_left_is_source = True
            for isrcx in range(self.n_f):
                if isrcx > 0:
                    contact_to_left_is_source = not contact_to_left_is_source
                on_first_x = (isrcx == 0)
                xcen = ((isrcx+1)//2)*Lhdh+(isrcx//2)*Lhsh - offset + col_offset
                xlef_hs = xcen - self.L_h/2
                xrit_hs = xcen + self.L_h/2
                for isrcy in range(self.rows):
                    ycen = self.row_pitch*isrcy - (self.rows-1)*self.row_pitch/2
                    yfore_base = ycen - self.w_f/2; yback = ycen + self.w_f/2
                    if crop_symmetric and yback < 0: continue
                    if active_only and not self._island_is_active(icolx, isrcy): continue
                    for contact_side in (['left','right'] if on_first_x else ['right']):
                        if contact_side=='left':
                            if contact_to_left_is_source:
                                xrit=xlef_hs - self.L_ho - self.lg - self.Lgs
                                xlef=xrit - self.Lcs
                            else:
                                xrit=xrit_hs + self.L_ho - self.Lgd
                                xlef=xrit - self.Lcd
                        else:
                            if contact_to_left_is_source:
                                xlef=xlef_hs - self.L_ho + self.Lgd
                                xrit=xlef + self.Lcd
                            else:
                                xlef=xrit_hs + self.L_ho + self.lg + self.Lgs
                                xrit=xlef + self.Lcs
                        assert xrit<=self.l_chipx/2, "Chip too small for the number of fingers including contacts!"
                        xlef_out=xlef; yfore_out=yfore_base
                        if crop_symmetric:
                            if self.half_x:
                                if xrit < 0: continue
                                xlef_out = max(xlef_out,0)
                            yfore_out = max(yfore_out,0)
                        yield (xlef_out, xrit, yfore_out, yback)

    def iter_gates(self, crop_symmetric: bool, even_if_not_meshing: bool=False, active_only: bool = False):
        """Iterate over the the gates.

        By default, will yield no results if dont_mesh_gates=True,
        but this can be overridden with even_if_not_meshing=True.

        Args:
            crop_symmetric: if True, crop results and only return gates in the symmetrically representative region
            even_if_not_meshing: if True, yield the gate positions even if dont_mesh_gates=True
            active_only: if True, skip gates belonging to inactive islands (see is_active)
        Yields:
            (xlef,xrit,yfore,yback) of each gate
        """
        if self.dont_mesh_gates and (not even_if_not_meshing): return
        Lhsh = self.Lhsh
        Lhdh = self.Lhdh
        offset = ((self.n_f//2)*Lhdh+((self.n_f-1)//2)*Lhsh)/2
        for icolx in range(self.columns):
            col_offset = self.column_pitch*icolx - (self.columns-1)*self.column_pitch/2
            contact_to_left_is_source = True
            for isrcx in range(self.n_f):
                if isrcx > 0:
                    contact_to_left_is_source = not contact_to_left_is_source
                xcen = ((isrcx+1)//2)*Lhdh+(isrcx//2)*Lhsh - offset + col_offset
                xlef_hs = xcen - self.L_h/2
                xrit_hs = xcen + self.L_h/2
                if contact_to_left_is_source:
                    xrit=xlef_hs - self.L_ho
                    xlef=xrit - self.lg
                else:
                    xlef=xrit_hs + self.L_ho
                    xrit=xlef + self.lg
                assert xrit<=self.l_chipx/2, "Chip too small for the number of fingers including gates!"
                for isrcy in range(self.rows):
                    ycen = self.row_pitch*isrcy - (self.rows-1)*self.row_pitch/2
                    yfore = ycen - self.w_f/2; yback = ycen + self.w_f/2
                    xlef_out=xlef
                    if active_only and not self._island_is_active(icolx, isrcy): continue
                    if crop_symmetric:
                        if self.half_x:
                            if xrit < 0: continue
                            xlef_out = max(xlef_out,0)
                        if yback < 0: continue
                        yfore = max(yfore,0)
                    yield (xlef_out, xrit, yfore, yback)


    def iter_islands(self, crop_symmetric: bool, active_only: bool = False):
        """Iterate over the device areas (island top surfaces).
        Args:
            crop_symmetric: if True, crop results and only return islands in the symmetrically representative region
            active_only: if True, skip inactive islands (see is_active)
        Yields:
            (xlef,xrit,yfore,yback) of each island area
        """
        Lhsh = self.Lhsh
        Lhdh = self.Lhdh
        offset = ((self.n_f//2)*Lhdh+((self.n_f-1)//2)*Lhsh)/2
        # Compute per-column island x-range from geometry
        xcen_base_first = -offset
        xlef_island_base = xcen_base_first - self.L_h/2 - self.L_ho - self.lg - self.Lgs - self.Lcs
        last = self.n_f - 1
        xcen_base_last = ((last+1)//2)*Lhdh + (last//2)*Lhsh - offset
        last_ctlis = (self.n_f % 2 == 1)  # contact_to_left_is_source for the last finger
        if last_ctlis:
            xrit_island_base = xcen_base_last - self.L_h/2 - self.L_ho + self.Lgd + self.Lcd
        else:
            xrit_island_base = xcen_base_last + self.L_h/2 + self.L_ho + self.lg + self.Lgs + self.Lcs
        for icolx in range(self.columns):
            col_offset = self.column_pitch*icolx - (self.columns-1)*self.column_pitch/2
            xlef = xlef_island_base + col_offset
            xrit = xrit_island_base + col_offset
            if crop_symmetric and self.half_x:
                if xrit < 0: continue
                xlef = max(xlef, 0)
            for isrcy in range(self.rows):
                ycen = self.row_pitch*isrcy - (self.rows-1)*self.row_pitch/2
                yfore = ycen - self.w_f/2; yback = ycen + self.w_f/2
                if active_only and not self._island_is_active(icolx, isrcy): continue
                if crop_symmetric:
                    if yback < 0: continue
                    yfore = max(yfore, 0)
                yield xlef, xrit, yfore, yback

    def get_mesh_path(self, force_remesh:bool=False, show_gui:bool=False) -> Path:
        """Get the path to the 3D mesh file, generating it if necessary.
        Args:
            force_remesh: if True, remesh even if a mesh file already exists
            show_gui: if True, show the gmsh GUI while meshing
        Returns:
            Path to the mesh file
        """

        assert self.sim_work_dir == get_default_sim_work_dir(),\
                "Simulation context has changed since creation of HEMT object.  "\
                "If using run_task/run_hemts, make sure to supply sim_work_dir explicitly."

        gmsh_filepath = get_mesh_dir() / f'hemt3d_{self.meshid}.mesh'
        vtk_filepath = get_mesh_dir() / f'hemt3d_{self.meshid}.vtk'
        get_mesh_dir().mkdir(exist_ok=True, parents=True)

        if force_remesh: gmsh_filepath.unlink(missing_ok=True)
        if not gmsh_filepath.exists():
            vtk_filepath.unlink(missing_ok=True)
            self.make_mesh(gmsh_filepath,show_gui=show_gui)
            assert gmsh_filepath.exists()
        elif show_gui: self.visualize_mesh()
        return gmsh_filepath
        
    
    def make_mesh(self, filepath:Path, show_gui:bool=True):
        """Make the 3D mesh file (raw output from gmsh, typically .mesh) at the specified path.
        Args:
            filepath: path to the mesh file to create
            show_gui: if True, show the gmsh GUI after meshing
        """
        assert self.sim_work_dir == get_default_sim_work_dir(), "Simulation context has changed since creation of HEMT object"
        tol=1e-6*um
        import gmsh
        gmsh.initialize()
        try:
            gmsh.model.add("hemt3d")
            kernel=gmsh.model.occ

            # Make the overall chip
            if self.half_x:
                top_surface=kernel.addRectangle(0, 0, -self.t_mesa, self.l_chipx/2, self.l_chipy/2)
            else:
                top_surface=kernel.addRectangle(-self.l_chipx/2, 0, -self.t_mesa, self.l_chipx, self.l_chipy/2)
            extrusion=kernel.extrude([(2, top_surface)], 0, 0, -self.t_GaN+self.t_mesa-self.t_Rel-self.t_Sub)

            # Make the GaN/Rel interface surface
            if self.half_x:
                GaNRel_surface=kernel.addRectangle(0, 0, -self.t_GaN, self.l_chipx/2, self.l_chipy/2)
                RelSub_surface=kernel.addRectangle(0, 0, -self.t_GaN-self.t_Rel, self.l_chipx/2, self.l_chipy/2)
            else:
                GaNRel_surface=kernel.addRectangle(-self.l_chipx/2, 0, -self.t_GaN, self.l_chipx, self.l_chipy/2)
                RelSub_surface=kernel.addRectangle(-self.l_chipx/2, 0, -self.t_GaN-self.t_Rel, self.l_chipx, self.l_chipy/2)
            kernel.synchronize()
            kernel.fragment(extrusion,[(2, GaNRel_surface),(2, RelSub_surface)])
            kernel.synchronize()

            # Make the mesas
            if self.t_mesa>0:
                island_surfaces=[
                    kernel.addRectangle(xlef, yfore , -self.t_mesa, (xrit-xlef), (yback-yfore))
                        for (xlef, xrit, yfore, yback) in self.iter_islands(crop_symmetric=True)]
                kernel.synchronize()
                outDimTags,_=kernel.fragment([(dim,num) for dim,num  in kernel.getEntities()
                                        if (dim==2 and np.isclose(kernel.getCenterOfMass(dim,num)[2],-self.t_mesa))
                                            or dim==3],
                                [(2, s) for s in island_surfaces])
                kernel.synchronize()
                island_surfaces=[dn[1] for dn in outDimTags
                                     if dn[0]==2 and np.isclose(kernel.getCenterOfMass(*dn)[2],-self.t_mesa)
                                         and len(gmsh.model.getAdjacencies(*dn)[1])==4]
                extrusions=kernel.extrude([(2, asurf) for asurf in island_surfaces],0, 0, self.t_mesa)
                kernel.synchronize()

            # Make the heat sources and contacts and gates
            heater_surfaces=[
                kernel.addRectangle(xlef, yfore , 0, (xrit-xlef), (yback-yfore))
                    for (xlef, xrit, yfore, yback) in self.iter_heaters(crop_symmetric=True, active_only=True)]
            contact_surfaces=[
                kernel.addRectangle(xlef, yfore , 0, (xrit-xlef), (yback-yfore))
                    for (xlef, xrit, yfore, yback) in self.iter_contacts(crop_symmetric=True)]
            gate_surfaces=[
                kernel.addRectangle(xlef, yfore , 0, (xrit-xlef), (yback-yfore))
                    for (xlef, xrit, yfore, yback) in self.iter_gates(crop_symmetric=True)]
            kernel.synchronize()
            outDimTags,_=kernel.fragment([(dim,num) for dim,num  in kernel.getEntities()
                                    if (dim==2 and kernel.getCenterOfMass(dim,num)[2]==0)
                                        or dim==3],
                            [(2, s) for s in heater_surfaces+contact_surfaces+gate_surfaces])
            kernel.synchronize()
            heater_and_gate_and_contact_surfaces=[dn[1] for dn in outDimTags if dn[0]==2
                    and kernel.getCenterOfMass(*dn)[2]==0 and len(gmsh.model.getAdjacencies(*dn)[1])==4]
            heater_surfaces=[dn[1] 
                    for (xlef, xrit, yfore, yback) in self.iter_heaters(crop_symmetric=True, active_only=True)
                        for dn in kernel.getEntitiesInBoundingBox(xlef-tol, yfore-tol, -tol, xrit+tol, yback+tol, tol, 2)]


            kernel.removeAllDuplicates()
            kernel.synchronize()

            # If we want an anisotropic mesh
            if self.aniso_mesh:
                # First make loose 3D mesh on which we'll calculate the mesh sizes before doing a proper tight mesh
                ialpha=5
            # Otherwise, just do an isotropic mesh directly
            else:
                # with the specified density
                ialpha=1/self.alpha_mesh

            # Set the mesh size field based on distance from the source surfaces with a couple layers of thresholding
            dist_from_source=gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_from_source, "SurfacesList", heater_surfaces)
            gmsh.model.mesh.field.setNumber(dist_from_source, "Sampling", 1000)
            threshold_source=gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_source, "InField", dist_from_source)
            gmsh.model.mesh.field.setNumber(threshold_source, "SizeMin", ialpha*min(0.015*um,self.L_h/10))
            gmsh.model.mesh.field.setNumber(threshold_source, "SizeMax", ialpha*min(0.3*um,self.L_h*25))
            gmsh.model.mesh.field.setNumber(threshold_source, "DistMin", min(0.01*um,self.L_h/2))
            gmsh.model.mesh.field.setNumber(threshold_source, "DistMax", 0.08*um)
            gmsh.model.mesh.field.setNumber(threshold_source, "StopAtDistMax", 1)
            threshold_source2=gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_source2, "InField", dist_from_source)
            gmsh.model.mesh.field.setNumber(threshold_source2, "SizeMin", ialpha*0.1*um)
            gmsh.model.mesh.field.setNumber(threshold_source2, "SizeMax", ialpha*0.3*um)
            gmsh.model.mesh.field.setNumber(threshold_source2, "DistMin", 1.5*um)
            gmsh.model.mesh.field.setNumber(threshold_source2, "DistMax", 2.0*um)
            gmsh.model.mesh.field.setNumber(threshold_source2, "StopAtDistMax", 1)
            threshold_source3=gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_source3, "InField", dist_from_source)
            gmsh.model.mesh.field.setNumber(threshold_source3, "SizeMin", ialpha*0.3*um)
            gmsh.model.mesh.field.setNumber(threshold_source3, "SizeMax", ialpha*5*um)
            gmsh.model.mesh.field.setNumber(threshold_source3, "DistMin", 2*um)
            gmsh.model.mesh.field.setNumber(threshold_source3, "DistMax", 40*um)
            gmsh.model.mesh.field.setNumber(threshold_source3, "StopAtDistMax", 1)

            # Generate
            minfield=gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(minfield, "FieldsList", [threshold_source, threshold_source2,threshold_source3])
            gmsh.model.mesh.field.setAsBackgroundMesh(minfield)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.model.mesh.generate(3)
            
            # Now if requested, do anisotropic mesh sizing, using mesh size formulas evaluated on that pre-generated mesh
            if self.aniso_mesh:

                # Conveniences
                alpha=self.alpha_mesh
                Lh=self.L_h; wf=self.w_f

                # This group will be the source surfaces
                source_face_group=gmsh.model.addPhysicalGroup(2, heater_surfaces)

                # This group will be the edges of the sources
                source_edge_group=gmsh.model.addPhysicalGroup(1,[tag for src in heater_surfaces
                                                                 for tag in gmsh.model.getAdjacencies(2, src)[1]])
                
                # This group will be edges of the sources oriented along y (ie fixed x), excluding those at x=0 
                source_line_ys=[y for _,_,yfore,yback in self.iter_heaters(crop_symmetric=True, active_only=True) for y in [yfore,yback]]
                source_far_edge_group=gmsh.model.addPhysicalGroup(1,[tag for src in heater_surfaces
                                                                     for tag in gmsh.model.getAdjacencies(2, src)[1]
                                                                     if any(np.isclose(kernel.getCenterOfMass(1,tag)[1], y) for y in source_line_ys if y>0)])
                
                # Make the anisotropic mesh sizing based on distance from these groups
                gmsh.plugin.setNumber("Distance","PhysicalLine",source_edge_group)
                distview=gmsh.plugin.run("Distance")
                gmsh.plugin.setNumber("Distance","PhysicalLine",source_far_edge_group)
                fardistview=gmsh.plugin.run("Distance")
                gmsh.plugin.setNumber("Distance","PhysicalSurface",source_face_group)
                surfdistview=gmsh.plugin.run("Distance")

                # A smooth step function that is 0 for x<dmin, 1 for x>dmax, and smoothly varies in between
                thresh=lambda x,dmin,dmax: f".5*(1+Tanh(8/({dmax}-{dmin})*({x}-({dmax}+{dmin})/2)))"

                # X is sized based on disance from any source edge,
                # with an additional large-element threshold based on distance from source surface
                assert um==1, "the mesh size expressions below all assume um=1"
                gmsh.plugin.setNumber("MathEval","View",gmsh.view.get_index(distview))
                gmsh.plugin.setNumber("MathEval","OtherView",gmsh.view.get_index(surfdistview))
                gmsh.plugin.setString("MathEval","Expression0",
                    f"{alpha}^2/min(10,max(1/12*{Lh}*(v0/{Lh})+5*{thresh('w0',Lh,20)},{Lh/12}))^2")
                mev_x=gmsh.plugin.run("MathEval")

                # Y is sized based on distance from the y-oriented source edges,
                # with an additional large-element threshold based on distance from source surface
                gmsh.plugin.setNumber("MathEval","View",gmsh.view.get_index(fardistview))
                gmsh.plugin.setNumber("MathEval","OtherView",gmsh.view.get_index(surfdistview))
                gmsh.plugin.setString("MathEval","Expression0",
                    f"{alpha}^2/min(10,max(1/4*{wf}*(v0/{wf})+5*{thresh('w0',Lh,20)},{Lh/8}))^2")
                mev_y=gmsh.plugin.run("MathEval")


                # Z is sized based on distance from source surface
                gmsh.plugin.setNumber("MathEval","View",gmsh.view.get_index(surfdistview))
                gmsh.plugin.setString("MathEval","Expression0",
                    f"{alpha}^2/min(10,max(1/8*{Lh}*(v0/{Lh})+5*{thresh('v0',Lh,20)},{Lh/12}))^2")
                mev_z=gmsh.plugin.run("MathEval")

                # Compile to a tensor field and set as background mesh
                gmsh.plugin.setString("MathEval","Expression0",f"0")
                mev_0=gmsh.plugin.run("MathEval")
                gmsh.plugin.setNumber("Scal2Tens","NumberOfComponents",9)
                gmsh.plugin.setNumber(    "Scal2Tens","View0",gmsh.view.get_index(mev_x));\
                    gmsh.plugin.setNumber("Scal2Tens","View1",gmsh.view.get_index(mev_0));\
                    gmsh.plugin.setNumber("Scal2Tens","View2",gmsh.view.get_index(mev_0))
                gmsh.plugin.setNumber(    "Scal2Tens","View3",gmsh.view.get_index(mev_0));\
                    gmsh.plugin.setNumber("Scal2Tens","View4",gmsh.view.get_index(mev_y));\
                    gmsh.plugin.setNumber("Scal2Tens","View5",gmsh.view.get_index(mev_0))
                gmsh.plugin.setNumber(    "Scal2Tens","View6",gmsh.view.get_index(mev_0));\
                    gmsh.plugin.setNumber("Scal2Tens","View7",gmsh.view.get_index(mev_0));\
                    gmsh.plugin.setNumber("Scal2Tens","View8",gmsh.view.get_index(mev_z))
                tens=gmsh.plugin.run("Scal2Tens")
                pv=gmsh.model.mesh.field.add("PostView")
                gmsh.model.mesh.field.setNumber(pv, "ViewIndex", gmsh.view.get_index(tens))
                gmsh.model.mesh.field.setAsBackgroundMesh(pv)

                # Clear the loose mesh and do 2-D anisotropic meshes on each surface with BAMG
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
                gmsh.option.setNumber("Mesh.SmoothRatio", 3)
                gmsh.option.setNumber("Mesh.AnisoMax", 10)
                gmsh.option.setNumber("Mesh.Algorithm", 7)
                gmsh.model.mesh.clear()
                kernel.synchronize()
                gmsh.model.mesh.generate(2)

                # Now make the 3-D anisotropic mesh
                # MMG3D fails if you try to mesh multiple volumes at once, so we 
                # set only one visible at a time and mesh them one by one
                gmsh.option.setNumber("Mesh.Algorithm3D", 7)
                all_vols = gmsh.model.getEntities(3)
                gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
                for ivol,(_,vol) in enumerate(all_vols):
                    gmsh.model.setVisibility([dn for dn in all_vols if dn[1]!=vol], False)
                    gmsh.model.setVisibility([(3,vol)], True)
                    kernel.synchronize()
                    print(f"Generating only for volume {ivol+1}/{len(all_vols)}")
                    gmsh.model.mesh.generate(3)
                kernel.synchronize()
                
                # Remove the views created for the mesh size fields
                # to make the meash easier to view in gmsh.
                # Can be handy to leave these in when debugging mesh size issues,
                # in which case just comment out this loop
                for view_tag in list(gmsh.view.get_tags()):
                    gmsh.view.remove(view_tag)


            # Also I like to hide the 3-D volumes when viewing the mesh
            # Can easily make them visible in gmsh GUI if needed
            # or comment this out
            all_vols = gmsh.model.getEntities(3)
            for ivol,(_,vol) in enumerate(all_vols):
                gmsh.model.setVisibility([(3,vol)], False)

            # Save the mesh
            gmsh.option.setNumber("Mesh.SaveAll", 1)
            gmsh.write(str(filepath))

            # Show if desired
            if show_gui: gmsh.fltk.run()
        finally:
            gmsh.finalize()

    def get_siminfo(self, force_remesh:bool=False, force_resim:bool=False) -> dict[str,Any]:
        """Get the simulation info dictionary, running the simulation if necessary.
        Args:
            force_remesh: if True, remesh even if a mesh file already exists
            force_resim: if True, rerun the simulation even if results already exist
        Returns:
            The simulation info dictionary
        """
        assert self.sim_work_dir == get_default_sim_work_dir(), "Simulation context has changed since creation of HEMT object"
        
        info_filepath = get_info_dir() / f'hemt3d_info_{self.simid}.json'
        if force_remesh: assert force_resim, "force_remesh=True only makes sense if also force_resim=True"
        if force_resim: info_filepath.unlink(missing_ok=True)
        if not info_filepath.exists():
            print(f"Need to run simulation to create {info_filepath}")
            return self.run_sim3d(force_remesh=force_remesh)
        else:
            print("Will use pre-existing results")
            with open(info_filepath,'r') as f: return json.load(f)

    def run_sim3d(self, force_remesh:bool=False) -> dict[str,Any]:
        """Run the thermal simulation, saving results and returning the simulation info dictionary.
        Args:
            force_remesh: if True, remesh even if a mesh file already exists
        Returns:
            The simulation info dictionary
        """
        assert self.sim_work_dir == get_default_sim_work_dir(), "Simulation context has changed since creation of HEMT object"
        
        from sfepy.discrete.fem import Mesh, FEDomain, Field
        from sfepy.discrete import (FieldVariable, Material, Integral,
                                     Function, Equation, Equations, Problem)
        from sfepy.solvers.ls import ScipyDirect
        from sfepy.solvers.nls import Newton
        from sfepy.base.base import IndexedStruct, Struct
        from sfepy.terms import Term
        from sfepy.postprocess.probes_vtk import Probe
        tol=1e-6*um
        mesh=Mesh.from_file(str(self.get_mesh_path(force_remesh=force_remesh)))
        domain=FEDomain('domain', mesh)
        omega = domain.create_region('Omega', 'all')
        #gan = domain.create_region('GaN', 'cells of group 1')
        #rel = domain.create_region('Rel', 'cells of group 2')
        gan = domain.create_region('GaN',  f'vertices in (z > {-self.t_GaN-tol:.8f})', 'cell')
        rel = domain.create_region('Rel', f'vertices in (z < {-self.t_GaN+tol:.8f}) *v vertices in (z > {-self.t_GaN-self.t_Rel-tol})', 'cell')
        sub = domain.create_region('Sub',  f'vertices in (z < {-self.t_GaN-self.t_Rel+tol:.8f})', 'cell')
        symm = 4 if self.half_x else 2
        
        sources=[
            domain.create_region(f'SingleSource{isrc}',
                  (f' vertices in (x > {xlef -tol:.8f}) *v'\
                   f' vertices in (x < {xrit +tol:.8f}) *v'\
                   f' vertices in (y > {yfore-tol:.8f}) *v'\
                   f' vertices in (y < {yback+tol:.8f}) *v'\
                   f' vertices in (z > {-tol:.8f})'),'facet')
                for isrc,(xlef, xrit, yfore, yback) \
                    in enumerate(self.iter_heaters(crop_symmetric=True, active_only=True))]
        source = domain.create_region('TotalSource',
                   ' +s '.join('r.'+r.name for r in sources), 'facet') # type: ignore
        actives=[
            domain.create_region(f'SingleActive{iact}',
                  (f' vertices in (x > {xlef -tol:.8f}) *v'\
                   f' vertices in (x < {xrit +tol:.8f}) *v'\
                   f' vertices in (y > {yfore-tol:.8f}) *v'\
                   f' vertices in (y < {yback+tol:.8f}) *v'\
                   f' vertices in (z > {     -tol:.8f})'),'facet')
                for iact,(xlef, xrit, yfore, yback) \
                    in enumerate(self.iter_islands(crop_symmetric=True, active_only=True))
        ]
        active = domain.create_region('TotalActive',
                   ' +s '.join('r.'+r.name for r in actives), 'facet') # type: ignore
        if not self.dont_mesh_contacts:
            contacts=[
                domain.create_region(f'SingleContact{icont}',
                      (f' vertices in (x > {xlef -tol:.8f}) *v'\
                       f' vertices in (x < {xrit +tol:.8f}) *v'\
                       f' vertices in (y > {yfore-tol:.8f}) *v'\
                       f' vertices in (y < {yback+tol:.8f}) *v'
                       f' vertices in (z > {-tol:.8f})'),'facet')
                    for icont,(xlef, xrit, yfore, yback) \
                        in enumerate(self.iter_contacts(crop_symmetric=True))]
            contact = domain.create_region('TotalContact',
                       ' +s '.join('r.'+r.name for r in contacts), 'facet') # type: ignore
        else:
            assert self.h_con==0, "dont_mesh_contacts=True only allowed if h_con=0"
            topsurface = domain.create_region('TopSurface', f'vertices in (z < {-tol:.8f})'.format(tol=tol), 'facet')
            contact = domain.create_region('TotalContact', f'r.TopSurface -s r.TotalSource'.format(tol=tol), 'facet', allow_empty=True)
        if (not self.dont_mesh_gates) and len(list(self.iter_gates(crop_symmetric=True)))>0:
            gates=[
                domain.create_region(f'SingleGate{icont}',
                      (f' vertices in (x > {xlef -tol:.8f}) *v'\
                       f' vertices in (x < {xrit +tol:.8f}) *v'\
                       f' vertices in (y > {yfore-tol:.8f}) *v'\
                       f' vertices in (y < {yback+tol:.8f}) *v'
                       f' vertices in (z > {-tol:.8f})'),'facet')
                    for icont,(xlef, xrit, yfore, yback) \
                        in enumerate(self.iter_gates(crop_symmetric=True))]
            gate = domain.create_region('TotalGate',
                       ' +s '.join('r.'+r.name for r in gates), 'facet') # type: ignore
        else:
            assert self.h_con==0, "dont_mesh_contacts=True only allowed if h_con=0"
            topsurface = domain.create_region('TopSurface2', f'vertices in (z < {-tol:.8f})'.format(tol=tol), 'facet')
            gate = domain.create_region('TotalGate', f'r.TopSurface2 -s r.TotalSource'.format(tol=tol), 'facet', allow_empty=True)

        noncontact_active = domain.create_region('NonContactActive', f'r.TotalActive -s r.TotalContact', 'facet')

        if self.h_bot is not None:
            sink = domain.create_region('Sink',
                       f'vertices in (z < {(-self.t_GaN-self.t_Rel-self.t_Sub+tol):.8f})','facet')
        else:
            if self.half_x:
                outr = domain.create_region('Outer',
                              f' vertices in (x > {( self.l_chipx/2-tol):.8f}) +s'\
                              f' vertices in (y > {( self.l_chipy/2-tol):.8f})', 'facet')
            else:
                outr = domain.create_region('Outer',
                              f' vertices in (x > {( self.l_chipx/2-tol):.8f}) +s'\
                              f' vertices in (x < {(-self.l_chipx/2+tol):.8f}) +s'\
                              f' vertices in (y > {( self.l_chipy/2-tol):.8f})', 'facet')
            sioutr = domain.create_region('SiOuter',
                          f' vertices in (z < {(-self.t_GaN-self.t_Rel+tol):.8f}) *s'\
                          f' r.{outr.name}' # type: ignore
                          ,'facet')
            sibot = domain.create_region('SiBottom',
                          f' vertices in (z < {(-self.t_GaN-self.t_Rel-self.t_Sub+tol):.8f})'
                          ,'facet')
            sink = domain.create_region('Sink', f' r.{sioutr.name} +s r.{sibot.name}','facet') # type: ignore
                                         
        field = Field.from_args('temperature', np.float64, 'scalar', omega, approx_order=1)
        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')

        Tref=300*K
        def get_k_GaN(T: np.ndarray): # type: ignore
            return self.k300_GaN*(Tref/T)**(self.nthr_GaN)
        def get_k_Rel(T: np.ndarray): # type: ignore
            return self.k300_Rel*(Tref/T)**(self.nthr_Rel)
        def get_k_Sub(T: np.ndarray): # type: ignore
            return self.k300_Sub*(Tref/T)**(self.nthr_Sub)
        def get_dkdT_GaN(T: np.ndarray): # type: ignore
            return -self.nthr_GaN*self.k300_GaN/Tref*(Tref/T)**(self.nthr_GaN+1)
        def get_dkdT_Rel(T: np.ndarray): # type: ignore
            return -self.nthr_Rel*self.k300_Rel/Tref*(Tref/T)**(self.nthr_Rel+1)
        def get_dkdT_Sub(T: np.ndarray): # type: ignore
            return -self.nthr_Sub*self.k300_Sub/Tref*(Tref/T)**(self.nthr_Sub+1)
        
        def get_heatsink(ts, coor, mode=None, problem:Problem=None, term:Term=None, **kwargs): # type: ignore
            if mode == 'qp':
                if self.h_bot is None:
                    r=np.sqrt((coor**2).sum(axis=1))
                    x=coor[:,0]; y=coor[:,1]; z=coor[:,2]

                    rhat_dot_nhat_where_nhat_along_x = (np.isclose(np.abs(x),self.l_chipx/2,atol=tol) * (np.abs(x)/r))
                    rhat_dot_nhat_where_nhat_along_y = (np.isclose(       y ,self.l_chipy/2,atol=tol) * (       y /r))
                    rhat_dot_nhat_where_nhat_along_z = (np.isclose(z,-self.t_GaN-self.t_Rel-self.t_Sub,atol=tol) * (-z/r))
                    rhat_dot_nhat = rhat_dot_nhat_where_nhat_along_x + rhat_dot_nhat_where_nhat_along_y + rhat_dot_nhat_where_nhat_along_z
                    rhat_dot_nhat_times_one_over_r = rhat_dot_nhat / r
                    that_reshaped=np.rollaxis(np.atleast_3d(rhat_dot_nhat_times_one_over_r),1)

                    return {'h': self.k300_Sub*that_reshaped, 'T0': that_reshaped*0+self.T_A, 'unit':that_reshaped*0+1}
                else:
                    r=np.zeros_like(coor[:,0])
                    r=np.rollaxis(np.atleast_3d(r),1)
                    return {'h': r*0+self.h_bot, 'T0': r*0+self.T_A, 'unit':r*0+1}

        heat_flux_mat = Material('heatflux', values={'Q': -self.P_per_W/self.L_h,'unit':1})
        heat_sink_mat = Material('heatsink', function=get_heatsink)
        contact_mat = Material('contact', values={'h':self.h_con, 'T0': self.T_A,'unit':1})
        gate_mat = Material('gate', values={'h':self.h_gat, 'T0': self.T_A,'unit':1})
        intgl = Integral('i', order=2)
        term_flow_GaN = Term.new('dw_nl_diffusion(get_k_GaN, get_dkdT_GaN, v, u)', intgl, gan, v=v, u=u, get_k_GaN=get_k_GaN, get_dkdT_GaN=get_dkdT_GaN)
        term_flow_Rel = Term.new('dw_nl_diffusion(get_k_Rel, get_dkdT_Rel, v, u)', intgl, rel, v=v, u=u, get_k_Rel=get_k_Rel, get_dkdT_Rel=get_dkdT_Rel)
        term_flow_Sub = Term.new('dw_nl_diffusion(get_k_Sub, get_dkdT_Sub, v, u)', intgl, sub, v=v, u=u, get_k_Sub=get_k_Sub, get_dkdT_Sub=get_dkdT_Sub)
        term_hgen = Term.new('dw_integrate(heatflux.Q, v)', intgl, source, heatflux=heat_flux_mat, v=v)
        term_hsnk = Term.new('dw_bc_newton(heatsink.h, heatsink.T0, v, u)', intgl, sink, heatsink=heat_sink_mat, v=v, u=u)
        term_cont = Term.new('dw_bc_newton(contact.h, contact.T0, v, u)', intgl, contact, contact=contact_mat, v=v, u=u)
        term_gate = Term.new('dw_bc_newton(gate.h, gate.T0, v, u)', intgl, gate, gate=gate_mat, v=v, u=u)

        eq = Equation('balance', term_flow_GaN+term_flow_Rel+term_flow_Sub+term_hgen+term_hsnk+term_cont+term_gate)
        eqs = Equations([eq])
        pb = Problem('heat', equations=eqs)

        ls = ScipyDirect({})
        nls_status = IndexedStruct()
        nls = Newton({}, lin_solver=ls, status=nls_status, i_max=50, eps_a=1e-10, )
        pb.set_solver(nls)
        with time_it("Saving regions"):
            get_state_dir().mkdir(exist_ok=True, parents=True)
            pb.save_regions_as_groups(get_state_dir()/f'hemt3d_regs_{self.simid}',region_names=['GaN','Rel','Sub','Sink','TotalSource','TotalGate','TotalContact','TotalActive'])
        with time_it("Solving"):
            coor_shape=pb.get_mesh_coors().shape
            state0=np.zeros(coor_shape[0],dtype=u.field.dtype)+self.T_A
            variables=pb.solve(status=nls_status,save_results=False, state0=state0)
        with time_it("Saving state"):
            state_file=get_state_dir()/f'hemt3d_{self.simid}.vtk'
            pb.save_state(str(state_file), variables)

        info_to_save={}
        info_to_save['sim_hash']=self.simid
        info_to_save['state_file']=str(state_file)

        
        def get_thermal_conductivity(ts, coor, mode=None, problem:Problem=None, term:Term=None, **kwargs): # type: ignore
            if mode == 'qp':
                T: np.array = problem.evaluate(f'ev_integrate.{term.integral.name}.{term.region.name}(u)', # type: ignore
                               u=u, mode='qp', integrals={'i':intgl}, **kwargs)
                T.shape=(T.shape[0]*T.shape[1],) 
                print(f"########## Evaluating thermal conductivity in {term.region.name} when max temp is {T.max()/K:.2f} K")
                kGaN = get_k_GaN(T)
                kRel = get_k_Rel(T)
                kSub = get_k_Sub(T)
                val = np.select([coor[:,2]>-self.t_GaN+tol,
                                 np.isclose(coor[:,2],self.t_GaN,atol=tol),
                                 coor[:,2]>-self.t_GaN-self.t_Rel+tol,
                                 np.isclose(coor[:,2],-self.t_GaN-self.t_Rel,atol=tol),],
                                [kGaN, 0.5*(kGaN+kRel), kRel, 0.5*(kRel+kSub)], default=kSub)
                return {'k':    val.reshape(-1,1,1),
                        'kmat': np.array([np.diag([ki]*3) for ki in val])}
        mat = Material('mymaterial', function=get_thermal_conductivity)
        # Check heat flux in and out boundary conditions
        heat_flux_in:float= -symm*pb.evaluate('ev_integrate_mat.i.TotalSource(heatflux.Q,u)', # type: ignore
                                   heatflux=heat_flux_mat, u=u, integrals={'i':intgl}, mode='eval')
        heat_flow_out_bottom_by_surf_flux:float= -symm*pb.evaluate('ev_surface_flux.i.Sink(mymaterial.kmat,u)', # type: ignore
                                   mymaterial=mat, u=u, integrals={'i':intgl}, mode='eval')
        heat_flow_out_contacts_by_surf_flux:float= -symm*pb.evaluate('ev_surface_flux.i.TotalContact(mymaterial.kmat,u)', # type: ignore
                                   mymaterial=mat, u=u, integrals={'i':intgl}, mode='eval')
        heat_flow_out_gates_by_surf_flux:float= -symm*pb.evaluate('ev_surface_flux.i.TotalGate(mymaterial.kmat,u)', # type: ignore
                                   mymaterial=mat, u=u, integrals={'i':intgl}, mode='eval')
        heat_flow_out_bottom_by_rel_flux:float= symm*pb.evaluate('ev_grad.i.Rel(mymaterial.k,u)', # type: ignore
                                   mymaterial=mat, u=u, integrals={'i':intgl}, mode='eval')[2] / self.t_Rel
        heat_flow_out_bottom_by_hbot:float=   symm*pb.evaluate('ev_integrate.i.Sink(heatsink.h,u)', # type: ignore
                                                         u=u, heatsink=heat_sink_mat, integrals={'i':intgl}, mode='eval')\
                                             -symm*pb.evaluate('ev_integrate_mat.i.Sink(heatsink.h,u)',  # type: ignore
                                                         u=u, heatsink=heat_sink_mat, integrals={'i':intgl}, mode='eval')*self.T_A
        heat_flow_out_contacts_by_hcon:float= symm*pb.evaluate('ev_integrate.i.TotalContact(contact.h,u)', # type: ignore
                                                         u=u, contact=contact_mat, integrals={'i':intgl}, mode='eval')\
                                             -symm*pb.evaluate('ev_integrate_mat.i.TotalContact(contact.h,u)',  # type: ignore
                                                         u=u, contact=contact_mat, integrals={'i':intgl}, mode='eval')*self.T_A
        heat_flow_out_gates_by_hcon:float=    symm*pb.evaluate('ev_integrate.i.TotalGate(gate.h,u)', # type: ignore
                                                         u=u, gate=gate_mat, integrals={'i':intgl}, mode='eval')\
                                             -symm*pb.evaluate('ev_integrate_mat.i.TotalGate(gate.h,u)',  # type: ignore
                                                         u=u, gate=gate_mat, integrals={'i':intgl}, mode='eval')*self.T_A
        heat_flow_out_by_surf_flux=heat_flow_out_bottom_by_surf_flux+heat_flow_out_contacts_by_surf_flux+heat_flow_out_gates_by_surf_flux
        heat_flow_out_by_hbot=heat_flow_out_bottom_by_hbot+heat_flow_out_contacts_by_hcon+heat_flow_out_gates_by_hcon
        Wtot = self.w_f * len(list(self.iter_heaters(crop_symmetric=False,active_only=True)))
        print(f"Heat flux in: {heat_flux_in/Wtot/(W/mm):.2f} W/mm")
        print(f"Heat flux out bottom by mean flux in Rel: {heat_flow_out_bottom_by_rel_flux/Wtot/(W/mm):.2f} W/mm")
        print(f"Heat flux out by surf flux: {heat_flow_out_by_surf_flux/Wtot/(W/mm):.2f} W/mm (can be off if k is T-dependent, or boundary mesh is loose)")
        print(f"Heat flux out by hbot: {heat_flow_out_by_hbot/Wtot/(W/mm):.2f} W/mm")
        assert np.isclose(heat_flux_in/Wtot,self.P_per_W, rtol=1e-2), f"Heat flux in does not match expectation!"
        assert np.isclose(heat_flux_in, heat_flow_out_by_hbot, rtol=1e-2), "Heat flux in and out do not match!"
        assert np.isclose(heat_flow_out_bottom_by_rel_flux, heat_flow_out_bottom_by_hbot, rtol=1e-2), "Heat flux out bottom by flux and by hbot do not match!"
        assert np.isclose(heat_flow_out_bottom_by_rel_flux, heat_flow_out_bottom_by_surf_flux, rtol=.2), "Heat flux out bottom by flux and by surface integral do not match!"
        info_to_save['heat_flux_in']=heat_flux_in
        info_to_save['heat_flow_out_bottom']=heat_flow_out_bottom_by_rel_flux

        bot_temps_elavg:np.ndarray=pb.evaluate('ev_integrate.i.Sink(u)', # type: ignore
                                           mymaterial=mat, u=u, integrals={'i':intgl}, mode='el_avg')
        max_bot_temp:float=bot_temps_elavg.max() # type: ignore
        info_to_save['max_bot_temp']=max_bot_temp

        source_temp_int: float=pb.evaluate('ev_integrate.i.TotalSource(u)', u=u, integrals={'i':intgl}, mode='eval') # type: ignore
        source_flux_simarea: float=pb.evaluate('ev_integrate_mat.i.TotalSource(heatflux.unit,u)', # type: ignore
                                   heatflux=heat_flux_mat, u=u, integrals={'i':intgl}, mode='eval')

        mean_source_temp=source_temp_int/source_flux_simarea
        rthweff_heater=(mean_source_temp-self.T_A)/(heat_flux_in/Wtot)
        info_to_save['mean_source_temp']=mean_source_temp
        info_to_save['RthWEff3D Heater Avg [K.mm/W]']=rthweff_heater/(K*mm/W)

        nca_temp_int: float=pb.evaluate('ev_integrate.i.NonContactActive(u)', u=u, integrals={'i':intgl}, mode='eval') # type: ignore
        nca_simarea: float=pb.evaluate('ev_integrate_mat.i.NonContactActive(heatflux.unit,u)', # type: ignore
                                   heatflux=heat_flux_mat, u=u, integrals={'i':intgl}, mode='eval')
        mean_nca_temp=nca_temp_int/nca_simarea
        rthweff_nca=(mean_nca_temp-self.T_A)/(heat_flux_in/Wtot)
        info_to_save['RthWEff3D NCA Avg [K.mm/W]']=rthweff_nca/(K*mm/W)


        act_temp_int: float=pb.evaluate('ev_integrate.i.TotalActive(u)', u=u, integrals={'i':intgl}, mode='eval') # type: ignore
        act_simarea: float=pb.evaluate('ev_integrate_mat.i.TotalActive(heatflux.unit,u)', # type: ignore
                                   heatflux=heat_flux_mat, u=u, integrals={'i':intgl}, mode='eval')
        mean_act_temp=act_temp_int/act_simarea
        rthweff_act=(mean_act_temp-self.T_A)/(heat_flux_in/Wtot)
        info_to_save['RthWEff3D ACT Avg [K.mm/W]']=rthweff_act/(K*mm/W)

        print(f"Max bot temperature: {max_bot_temp/K:.2f} K")
        print(f"Top temperature: {mean_source_temp/K:.2f} K")
        print(f"RthWEff3D Heater Avg: { rthweff_heater/ (K*mm/W):.2f} K mm/W")
        print(f"RthWEff3D NCAct  Avg: { rthweff_nca/ (K*mm/W):.2f} K mm/W")
        print(f"RthWEff3D Active Avg: { rthweff_act/ (K*mm/W):.2f} K mm/W")
        
        with time_it("Probing"):
            with probe_cache_context():
                from sfepy.discrete.probes import LineProbe
                profiles=info_to_save['profiles']={}
                x_for_cut= 0 if self.n_f%2==1 else (self.Lhdh/2 if self.n_f%4==2 else self.Lhsh/2)
                y_for_cut= 0 if self.rows%2==1 else self.row_pitch/2
                xcuts=profiles['xcuts_by_depth']={}
                subs_depths=[d for d in np.linspace(0,self.t_GaN+self.t_Rel+self.t_Sub,5) if d>self.t_GaN+self.t_Rel+tol]
                for depth in chain(np.linspace(0,self.t_GaN+self.t_Rel,7),subs_depths):
                    pars,vals=LineProbe([0,y_for_cut,-depth], [self.l_chipx/2,y_for_cut,-depth], int(self.l_chipx/2/(.01*um))).probe(u) # type: ignore
                    xcuts[depth]={'x [um]': (pars/um).tolist(),
                                  'T [K]': (np.ravel(vals)/K).tolist()}
                ycuts=profiles['ycuts_by_depth']={}
                for depth in chain(np.linspace(0,self.t_GaN+self.t_Rel,7,endpoint=True),subs_depths):
                    pars,vals=LineProbe([x_for_cut,0,-depth], [x_for_cut,self.l_chipy/2,-depth], int(self.l_chipy/2/(.01*um))).probe(u) # type: ignore
                    ycuts[depth]={'y [um]': (pars/um).tolist(),
                                  'T [K]': (np.ravel(vals)/K).tolist()}
                pars,vals=LineProbe([x_for_cut,y_for_cut,0], [x_for_cut,y_for_cut,-(self.t_GaN+self.t_Rel+self.t_Sub)], # type: ignore
                                    int((self.t_GaN+self.t_Rel+self.t_Sub)/((self.t_GaN+self.t_Rel)/100))).probe(u)
                profiles['zcut']={'-z [um]': (pars/um).tolist(),
                      'T [K]': (np.ravel(vals)/K).tolist()}
        info_to_save['hemt_parameters']={k: v for k, v in asdict(self).items() if not callable(v)}
        info_to_save['units']={'m':m, 'K':K, 'W':W}
        info_to_save['TA [K]']=self.T_A/K
        
        get_info_dir().mkdir(exist_ok=True, parents=True)
        import json
        info_file=get_info_dir() / f'hemt3d_info_{self.simid}.json'
        print(f"Saving info to {info_file}")
        with open(info_file, 'w') as f:
            json.dump(info_to_save, f, indent=4)
        print(f" -> Note: mesh (with {len(omega.cells)} 3D elts) is in {self.get_mesh_path()}")
        print("Done")
        return info_to_save
    
    def plot_temp_profile3d(self, t_max:Optional[float]=None):
        """Plot the 1-D temperature profiles from the 3D simulation. """
        info_to_save=self.get_siminfo()
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        profiles=info_to_save['profiles']
        xcuts=profiles['xcuts_by_depth']
        for depth,cut in xcuts.items():
            x_um,T_K = np.array(cut['x [um]']), cut['T [K]']
            c=plt.plot(x_um, T_K, label=f'{float(depth)/um:.2f}')[0].get_color()
            plt.plot(-x_um, T_K,color=c)
        for xlef,xrit in set((xlef,xrit) for (xlef, xrit, yfore, yback)
                                            in self.iter_heaters(crop_symmetric=False)):
            plt.axvspan(xlef/um, xrit/um, color='gray', alpha=0.3)
        #plt.axhline(info_to_save['mean_source_temp']/K, color='k', ls='--')#, label='Mean source T')
        plt.xlabel(r'Position (along current-flow direction) [μm]')
        plt.ylabel(r'T [K]')
        plt.xlim(-self.l_chipx/2/um, self.l_chipx/2/um)
        plt.ylim(ymin=info_to_save['TA [K]'], ymax=t_max)
        plt.legend(title=r'Depth [μm]')
        plt.subplot(1,3,2)
        ycuts=profiles['ycuts_by_depth']
        for depth,cut in ycuts.items():
            y_um,T_K = np.array(cut['y [um]']), cut['T [K]']
            c=plt.plot(y_um, T_K, label=f'{float(depth)/um:.2f}')[0].get_color()
            plt.plot(-y_um, T_K,color=c)
        for yfore,yback in set((yfore,yback) for (xlef, xrit, yfore, yback)
                                            in self.iter_heaters(crop_symmetric=False)):
            plt.axvspan(yfore/um,yback/um, color='gray', alpha=0.3)
        #plt.axhline(info_to_save['mean_source_temp']/K, color='k', ls='--')#, label='Mean source T')
        plt.legend(title=r'Depth [μm]')
        plt.xlabel(r'Position (along gate line direction) [μm]')
        plt.ylabel(r'T [K]')
        plt.xlim(-self.l_chipy/2/um, self.l_chipy/2/um)
        plt.ylim(ymin=info_to_save['TA [K]'], ymax=t_max)
        plt.legend(title=r'Depth [μm]')
        plt.subplot(1,3,3)
        zcut=profiles['zcut']
        depth_um,T_K = zcut['-z [um]'], zcut['T [K]']
        plt.plot(depth_um, T_K)
        #plt.axhline(info_to_save['mean_source_temp']/K, color='k', ls='--', label='Mean source T')
        hp=info_to_save['hemt_parameters']
        plt.xlim(0, min(hp['t_GaN']/um + hp['t_Rel']/um + hp['t_Sub']/um, (hp['t_GaN']/um + hp['t_Rel']/um)*5))
        plt.axvline(hp['t_GaN']/um, color='gray', ls='--')
        plt.axvline((hp['t_GaN']+hp['t_Rel'])/um, color='gray', ls='--')
        plt.ylim(ymin=info_to_save['TA [K]'], ymax=t_max)
        plt.xlabel(r'Depth below gate [μm]')
        plt.ylabel(r'T [K]')
        plt.tight_layout()
        plt.show()

    def visualize_mesh(self):
        """Visualize the 3D mesh with gmsh."""
        env = {**os.environ, 'PATH': sysconfig.get_path('scripts') + os.pathsep + os.environ.get('PATH', '')}
        subprocess.run(['gmsh', str(self.get_mesh_path())], env=env, shell=True)
   
    def visualize_solution_3d(self):
        """Visualize the 3D solution with sfepy-view."""
        env = {**os.environ, 'PATH': sysconfig.get_path('scripts') + os.pathsep + os.environ.get('PATH', '')}
        subprocess.Popen(['sfepy-view', str(get_state_dir() / f'hemt3d_{self.simid}.vtk')], env=env, shell=True)
    
    def visualize_regs_3d(self):
        """Visualize the 3D regions with sfepy-view."""
        env = {**os.environ, 'PATH': sysconfig.get_path('scripts') + os.pathsep + os.environ.get('PATH', '')}
        subprocess.Popen(['sfepy-view', str(get_state_dir() / f'hemt3d_regs_{self.simid}.vtk')], env=env, shell=True)
    
    def visualize_elements(self, crop_symmetric:bool=False, active_only:bool=False):
        """Visualize the 2D topsurface elements (heaters, contacts, gates) with matplotlib."""
        from matplotlib import pyplot as plt
        for (xlef,xrit,yfore,yback) in self.iter_heaters(crop_symmetric=crop_symmetric, active_only=active_only):
            plt.plot([xlef/um,xrit/um,xrit/um,xlef/um,xlef/um],
                     [yfore/um,yfore/um,yback/um,yback/um,yfore/um], 'r-')
        for (xlef,xrit,yfore,yback) in self.iter_contacts(crop_symmetric=crop_symmetric,even_if_not_meshing=True, active_only=active_only):
            plt.plot([xlef/um,xrit/um,xrit/um,xlef/um,xlef/um],
                     [yfore/um,yfore/um,yback/um,yback/um,yfore/um], '--',color='gold')
        for (xlef,xrit,yfore,yback) in self.iter_gates(crop_symmetric=crop_symmetric,even_if_not_meshing=True, active_only=active_only):
            plt.plot([xlef/um,xrit/um,xrit/um,xlef/um,xlef/um],
                     [yfore/um,yfore/um,yback/um,yback/um,yfore/um], '-.',color='grey')
        for (xlef,xrit,yfore,yback) in self.iter_islands(crop_symmetric=crop_symmetric, active_only=active_only):
            plt.plot([xlef/um,xrit/um,xrit/um,xlef/um,xlef/um],
                     [yfore/um,yfore/um,yback/um,yback/um,yfore/um], ':',color='green')
        plt.xlabel('x [um]')
        plt.ylabel('y [um]')
        plt.show()
