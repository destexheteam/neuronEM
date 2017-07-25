#!/usr/bin/env python
#coding=utf-8

import neuron
import numpy as np

h = neuron.h

def integrate(tstop, i_axial=False, neuron_cells = None):
    """Run Neuron simulation and return time and 2d array of 
    transmembrane currents (n_pts x n_segs). If i_axial is true
    return in addition axial currents.
    For particular cell in the network, pass the cell as neuron_cell,
    if for all the cells in the network leave it None"""

    i_membrane_all = []
    i_axial_all = [] 
    
    if neuron_cells != None:      
        for idx_cell in range(len(neuron_cells)):
            i_membrane_all.append([]) 
            i_axial_all.append([])  
    
    li = []
    #for i in xrange(numcells):
    #    li.append( np.array( tstop/h.dt, neuron_cells[i].all ) 
        
    while h.t < tstop:
        h.fadvance()
        if neuron_cells ==None:
            v = get_for_all(get_i_membrane)
            i_membrane_all.append(v)
        else:
            v = []
            for idx_cell, next_cell in enumerate(neuron_cells):
                v_next = get_bycell(get_i_membrane, next_cell)
                i_membrane_all[idx_cell].append(v_next)
                
        if i_axial:
            if neuron_cells ==None:
                iax = get_for_all(get_i_axial)
                i_axial_all.append(iax)
            else:
                iax = []
                for idx_cell, next_cell in enumerate(neuron_cells): 
                    iax_next = get_bycell(get_i_axial, next_cell)
                    i_axial_all[idx_cell].append(iax_next) 
                       
    if neuron_cells == None:
        t = np.arange(0, len(i_membrane_all))*h.dt
    else:
        t = np.arange(0, len(i_membrane_all[0]))*h.dt
        
    if i_axial:
        
        return t, np.array(i_membrane_all), np.array(i_axial_all)
    else:
    
        return t, np.array(i_membrane_all)

def insert_extracellular():
    for sec in h.allsec():
        sec.insert("extracellular")

def get_v():
    v = []
    for sec in h.allsec():
        for seg in sec:
            v.append(seg.v)
    return v


def get_bycell(func, neuron_cell):
    """loops through all the segments of given cell
    and calculates the given function for each segment"""
    variable = []

    for sec in neuron_cell.all:
        variable = func(variable, sec)
    return variable
    
def get_for_all(func):
    """loops through all the segments of all the cells
    and calculates the given function for each segment"""
    variable = []
    for sec in h.allsec():
        variable = func(variable, sec)
    return variable
    
def get_i_membrane(v, sec):
    i_sec = [seg.i_membrane for seg in sec]
    x = [seg.x for seg in sec]
    #add currents from point processes at the beginning and end of section
    c_factor = 100 #from  [i/area]=nA/um2 to [i_membrane]=mA/cm2
    area0 = h.area(x[0], sec=sec)
    area1 = h.area(x[-1], sec=sec)
    i_sec[0] += sum(pp.i for pp in sec(0).point_processes())/area0*c_factor
    i_sec[-1] += sum(pp.i for pp in sec(1).point_processes())/area1*c_factor
    v += i_sec
    return v    

def get_i_axial(currents, sec):
    """return axial current density in mA/cm2"""
    #for sec in h.allsec():
    v0 = sec(0).v
    for seg in sec:
       v1 = seg.v
       l = sec.L/sec.nseg #length in um
       r = sec.Ra #resistance in ohm
       iax = (v1-v0)/(r*l*1e-4)
       currents.append(iax)
       v0 = v1
    return currents

def get_nsegs():
    nsegs = 0
    for sec in h.allsec():
        nsegs += sec.nseg
    return nsegs


def get_coords():
    total_segs = get_nsegs()
    coords = np.zeros(total_segs,
                      dtype=[("x", np.float32),
                             ("y", np.float32),
                             ("z", np.float32),
                             ("L", np.float32),
                             ("diam", np.float32)
                            ])
    j = 0
    for sec in h.allsec():
        n3d = int(h.n3d(sec))
        x = np.array([h.x3d(i,sec) for i in range(n3d)])
        y = np.array([h.y3d(i,sec) for i in range(n3d)])
        z = np.array([h.z3d(i,sec) for i in range(n3d)])
        nseg = sec.nseg
        pt3d_x = np.arange(n3d)
        seg_x = np.arange(nseg)+0.5

        if len(pt3d_x)<1:
            x_coord = y_coord = z_coord =np.ones(nseg)*np.nan
        else:
            x_coord = np.interp(seg_x, pt3d_x, x)
            y_coord = np.interp(seg_x, pt3d_x, y)
            z_coord = np.interp(seg_x, pt3d_x, z)
      
        lengths = np.zeros(nseg)
        diams = np.zeros(nseg)
        
        lengths = np.ones(nseg)*sec.L*1./nseg
        diams = np.ones(nseg)*sec.diam

        coords['x'][j:j+nseg]=x_coord
        coords['y'][j:j+nseg]=y_coord
        coords['z'][j:j+nseg]=z_coord
        coords['L'][j:j+nseg]=lengths
        coords['diam'][j:j+nseg]=diams

        j+=nseg

    return coords

def get_seg_coords():
    nchars = 40
    total_segs = get_nsegs()
   
    coords = np.zeros(total_segs,
                      dtype=[("x0", np.float32),
                             ("y0", np.float32),
                             ("z0", np.float32),
                             ("x1", np.float32),
                             ("y1", np.float32),
                             ("z1", np.float32),
                             ("L", np.float32),
                             ("diam", np.float32),
                             ("name", "|S%d" % nchars)
                            ])
                            
    j = 0
    for sec in h.allsec():
        nseg = sec.nseg
        
        diams = np.ones(nseg)*sec.diam
        lengths = np.ones(nseg)*sec.L*1./nseg
        names = np.repeat(sec.name()[:nchars], nseg).astype("|S%d"%nchars)

        seg_x = np.arange(nseg+1)*1./nseg
        
        x_coord, y_coord, z_coord = get_locs_coord(sec, seg_x)
      

        coords['x0'][j:j+nseg] = x_coord[:-1]
        coords['y0'][j:j+nseg] = y_coord[:-1]
        coords['z0'][j:j+nseg] = z_coord[:-1]
        
        coords['x1'][j:j+nseg] = x_coord[1:]
        coords['y1'][j:j+nseg] = y_coord[1:]
        coords['z1'][j:j+nseg] = z_coord[1:]

        coords['diam'][j:j+nseg] = diams
        coords['L'][j:j+nseg] = lengths
        coords['name'][j:j+nseg] = names 
        j+=nseg

    return coords

def get_locs_coord(sec, loc):
    """get 3d coordinates of section locations"""
    n3d = int(h.n3d(sec=sec))

    x = np.array([h.x3d(i, sec=sec) for i in range(n3d)])
    y = np.array([h.y3d(i, sec=sec) for i in range(n3d)])
    z = np.array([h.z3d(i, sec=sec) for i in range(n3d)])

    arcl = np.sqrt(np.diff(x)**2+np.diff(y)**2+np.diff(z)**2)
    arcl = np.cumsum(np.concatenate(([0], arcl)))
    nseg = sec.nseg
    pt3d_x = arcl/arcl[-1]
    x_coord = np.interp(loc, pt3d_x, x)
    y_coord = np.interp(loc, pt3d_x, y)
    z_coord = np.interp(loc, pt3d_x, z)

    return x_coord, y_coord, z_coord

def get_pp_coord(point_process):
    """get 3d coordinates of a point process such as synapse:
        point_process -- neuron object"""

    loc = point_process.get_loc()
    sec = h.cas()
    coord = get_locs_coord(sec, loc)
    h.pop_section()
    return coord

def get_point_processes():
    """Returns record array with all point processes and their
    coordinates as tuple.
    """
    point_processes = []
    for sec in h.allsec():
        pp_in_sec = []
        for seg in sec.allseg():
            pp_in_sec += seg.point_processes()
        locs = [pp.get_loc() for pp in pp_in_sec]
        #remove section from stack to avoid overflow
        [h.pop_section() for pp in pp_in_sec]
        x, y, z = get_locs_coord(sec, locs)
        point_processes += zip(pp_in_sec, x, y, z)
    return point_processes

def initialize(dt=0.025):
    insert_extracellular()
    h.finitialize()
    h.dt = dt
    h.fcurrent()
    h.frecord_init()

def load_model(hoc_name, dll_name=None):
    if dll_name:
        h.nrn_load_dll(dll_name)
    h.load_file(hoc_name)

def load_model_swc(filename_swc):
    '''loads morphology of the neuron from the swc file'''
    h.load_file("celbild.hoc")
    h.load_file('import3d.hoc')

    Import = h.Import3d_SWC_read()
    Import.input(filename_swc)
    imprt = h.Import3d_GUI(Import, 0)
    imprt.instantiate(None)

