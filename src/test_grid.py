import sys
import awkward as ak
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import os.path as osp
from glob import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import vector
import math
import collections
from ele_block import ele_var
from grid import CellIndex, CellGrid

data_ZH = ak.from_parquet('hps/HPS/QCD/reco_p8_ee_qcd_ecm380_128.parquet')

def build_p4(data, part):
    return vector.awk(
        ak.zip({
            "px":data[part].x,
            "py":data[part].y,
            "pz":data[part].z,
            "mass":data[part].tau
        })
    )

def get_part_p4(pdgid, data):
    part = data.event_reco_cand_p4s[data.event_reco_cand_pdg==pdgid]
    return vector.awk(
        ak.zip({'mass': part.tau, 
                'px': part.x, 
                'py': part.y,
                'pz': part.z}
           )
    )

ele_p4 = get_part_p4(11, data_ZH)

def signalCone(pt):
    minpt = 30
    minr=0.05
    cone = 3
    return np.maximum(3/np.maximum(pt, minpt), 0.05)

n_inner_cells = 11
inner_cell_size = 0.02

def process_onejet(etas, phis, parts, part_type):
    if part_type == 'ele':
        part_var = ele_var
    list_part_var = np.zeros((121,len(part_var)))
    for eta, phi, part in zip(etas, phis, parts): 
        if eta == None or phi ==None: # it means eta, phi are not within inner or outer grid
            continue
        cellIndex = CellIndex(eta, phi)
        flatcellIndex = grid.GetFlatIndex(cellIndex)
        if part_type == 'ele':
            list_part_var[flatcellIndex][part_var['elept']] += part.pt
            list_part_var[flatcellIndex][part_var['eleeta']] += part.eta
            list_part_var[flatcellIndex][part_var['elephi']] += part.phi
            list_part_var[flatcellIndex][part_var['elemass']] += part.mass
    return list_part_var

grid = CellGrid(n_inner_cells, n_inner_cells, inner_cell_size, inner_cell_size)
reco_jet_p4 = build_p4(data_ZH, 'reco_jet_p4s')
signalcone = signalCone(reco_jet_p4.pt)
dr = reco_jet_p4.deltaR(ele_p4)
deta = reco_jet_p4.eta - ele_p4.eta
dphi = reco_jet_p4.phi - ele_p4.phi
ele_inside_signalcone = ak.mask(ele_p4, dr < signalcone)
maskdeta = ak.mask(deta, dr < signalcone)
maskdphi = ak.mask(dphi, dr < signalcone)
etacellindex, phicellindex = grid.getcellIndex(maskdeta, maskdphi)
for idx, eta in enumerate(etacellindex): #this loops over all jet?
    list_part_info_perjet = process_onejet(eta, phicellindex[idx], ele_inside_signalcone[idx], 'ele')
    if idx == 0:
        list_all_part_info = list_part_info_perjet
    else:
        list_all_part_info = np.concatenate((list_part_info_perjet, list_all_part_info), axis=-1)

list_ak = ak.from_numpy(list_all_part_info)
print('type of list_ak ', type(list_ak))
write_info = {field: data_ZH[field] for field in data_ZH.fields}
write_info.update({'inner_grid_ele_block': list_ak})
ak.to_parquet(ak.Record(write_info), 'test_grid.parquet')
print(list_all_part_info.shape)
assert(list_all_part_info.shape[0] == n_inner_cells * n_inner_cells)
assert(list_all_part_info.shape[1] == len(ele_var) * len(reco_jet_p4))
