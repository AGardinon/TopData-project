#!

import numpy as np
import gudhi as gd
from gudhi.representations import DiagramSelector, DiagramScaler,\
	Clamping, Landscape, Silhouette, BottleneckDistance

from sklearn.preprocessing import MinMaxScaler

import networkx as nx
import itertools
import matplotlib.pyplot as plt


def get_skeleton(data,max_edge_length):
    return gd.RipsComplex(points = data,
                          max_edge_length = max_edge_length)


def get_VRcomplex(skeleton,max_dim):
    return skeleton.create_simplex_tree(max_dimension = max_dim)


def get_barCode(VRcomplex):
    return VRcomplex.persistence()


def get_filtration(data,VRcomplex,r):
    filtration_ = list()
    for i in VRcomplex.get_filtration():
        if i[1] != 0.0 and i[1] <= r:
            filtration_.append(i[0])
            
    return [tuple(i) for i in filtration_]


def plt_filtration(data,filtration,r,circle=False,axes=None):
    Npoints = len(data)
    G = nx.Graph()
    G.add_nodes_from(np.arange(Npoints))
    for node_tuple in filtration:
        G.add_edges_from(itertools.combinations_with_replacement(node_tuple, 2))
    
    if circle:
	    for point in data:
	    	cir = plt.Circle(point, r, color='gold', 
	    					 fill=True, alpha=0.3)
	    	axes.add_patch(cir)
	    	axes.set_aspect('equal', adjustable='datalim')
    
    nx.draw(G, pos=data,
            node_size=50, node_color='tab:green',
            width=1.0, edge_color='tab:blue',
            alpha=1, ax=axes)

    return axes


########## OLD FUNCTIONS #
def get_landscape(VRcomplex, res, dim=1):
	ls = Landscape(resolution=res)
	return ls.fit_transform([VRcomplex.persistence_intervals_in_dimension(dim)])


def get_silhouette(VRcomplex, res, wgt="power", dim=1):
	if wgt == "power":
		weight_ = lambda x: np.power(x[1] - x[0], 1)
	else:
		weight_ = wgt
	sh = gd.representations.Silhouette(resolution=res, weight=weight_)
	return sh.fit_transform([VRcomplex.persistence_intervals_in_dimension(dim)])
# OLD FUNCTIONS ##########


def get_landscape_new(VRcomplex, res, dim=1):
	ls = Landscape(resolution=res)
	diags = [VRcomplex.persistence_intervals_in_dimension(dim)]
	diags = DiagramSelector(use=True, point_type="finite").fit_transform(diags)
	diags = DiagramScaler(use=True, scalers=[([0,1], MinMaxScaler())]).fit_transform(diags)
	return ls.fit_transform(diags)


def pow(n):
	return lambda x: np.power(x[1]-x[0],n)


def get_silhouette_new(VRcomplex, res, wgt="power", dim=1):
	if wgt == "power":
		weight_ = pow(2)
	else:
		weight_ = wgt
	diags = [VRcomplex.persistence_intervals_in_dimension(dim)]
	diags = DiagramSelector(use=True, point_type="finite").fit_transform(diags)
	diags = DiagramScaler(use=True, scalers=[([0,1], MinMaxScaler())]).fit_transform(diags)
	sh = Silhouette(resolution=res, weight=weight_)
	return sh.fit_transform(diags)
