# vim: set fileencoding=utf-8 :

"""Spatial distribution of the field potential due to synaptic
input to two neurons. 
"""

import numpy as np
import matplotlib.pyplot as plt
from eap import field, cell, graph

import platform
ARCH = platform.uname()[4]

# Parameteres
dt = 0.025
tstop = 50
xyz_cell1 = [0,0,0] #microm
xyz_cell2 = [200,0,0]
cell_list = []

syn_e = -72.0
con_weight=.45e-3
tau1 = 0.25
tau2 = 8
t_length=5
time_to_plot = 4

# SETUP CELLS
# load your model cell
cell.load_model('models/amaral.hoc')

cells = cell.h.List()
cells.append(cell.h.AmaralCell(xyz_cell1[0], xyz_cell1[1], xyz_cell1[2],0))
cells.append(cell.h.AmaralCell(xyz_cell2[0], xyz_cell2[1], xyz_cell2[2],0))

nclist = []
stimlist = []

# init synapses
def init_synapse(cell_no, syn_start):
    syn = cell.h.Exp2Syn(0.5, sec=cells[cell_no].soma[0])
    syn.e = syn_e
    syn.tau1 = tau1 #.1
    syn.tau2 = tau2 #5  

    stimlist.append(cell.h.NetStim())
    stimlist[-1].start=syn_start
    stimlist[-1].number=1
    stimlist[-1].interval=1
    stimlist[-1].noise=0
    nclist.append(cell.h.NetCon(stimlist[-1],syn,con_thresh=0,
                        con_delay=1,con_weight=.25e-3))

init_synapse(cell_no=0,syn_start=0.1)
init_synapse(cell_no=1,syn_start=0.5)

# initialise and run neuron simulations
cell.initialize(dt=dt)
t, I_cell = cell.integrate(t_length, i_axial=False, neuron_cells=cells)

# CALCULATION OF THE FIELD
seg_coords = cell.get_seg_coords()
n_samp = 40
x_range = [-210, 430]
y_range = [-255, 725]

# define grid
xx, yy = field.calc_grid(x_range, y_range, n_samp)

# join the currents together
I = I_cell.swapaxes(0,1).reshape(I_cell.shape[1], -1)

v_ext = field.estimate_on_grid(seg_coords, I, xx, yy)

# PLOTS
synapses = [s for s in cell.get_point_processes()]

# plot neurons shape
graph.plot_neuron(seg_coords, colors='0.4')

# plot field potential
plt.imshow((v_ext[time_to_plot,:,:]), interpolation="nearest", extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
	       origin='lower')
    
cbar = plt.colorbar()
plt.xlabel(r'$\mu m$')
plt.ylabel(r'$\mu m$')
cbar.ax.set_ylabel('nV')
plt.show()

