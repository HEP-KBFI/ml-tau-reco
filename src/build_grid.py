import awkward as ak
import numpy as np
import vector
import json

from part_var import part_var_list
from grid import CellIndex, CellGrid
from basicTauBuilder import BasicTauBuilder


class GridBuilder(BasicTauBuilder):
    def __init__(self, cfgFileName="./config/grid_builder.json", verbosity=0):
        super(BasicTauBuilder, self).__init__()
        cfgFile = open(cfgFileName, "r")
        cfg = json.load(cfgFile)
        self._builderConfig = cfg["GridAlgo"]
        cfgFile.close()
        self.parttype = ["ele", "gamma", "mu", "charged_cand", "neutral_cand"]

    def build_p4(self, data, part_p4):
        return vector.awk(
            ak.zip({"px": data[part_p4].x, "py": data[part_p4].y, "pz": data[part_p4].z, "mass": data[part_p4].tau})
        )

    def get_part_p4(self, parttype, data):
        if parttype == "ele":
            partp4 = data.event_reco_cand_p4s[data.event_reco_cand_pdg == 11]
        elif parttype == "gamma":
            partp4 = data.event_reco_cand_p4s[data.event_reco_cand_pdg == 21]
        elif parttype == "mu":
            partp4 = data.event_reco_cand_p4s[data.event_reco_cand_pdg == 13]
        elif parttype == "charged_cand":
            partp4 = data.event_reco_cand_p4s[data.event_reco_cand_pdg == 211]
        elif parttype == "neutral_cand":
            total_mask = ak.zeros_like(data.reco_cand_pdg)
            for neutral_cand in [2112, 130]:
                mask_addition = ak.where(
                    data.reco_cand_pdg == neutral_cand, ak.ones_like(data.reco_cand_pdg), ak.zeros_like(data.reco_cand_pdg)
                )
                total_mask = total_mask + mask_addition
            partp4 = data.event_reco_cand_p4s[total_mask > 0]
        else:
            assert 0
        return vector.awk(ak.zip({"mass": partp4.tau, "px": partp4.x, "py": partp4.y, "pz": partp4.z}))

    def signalCone(self, pt):
        minpt = 30
        minr = 0.05
        cone = 3
        return np.maximum(cone / np.maximum(pt, minpt), minr)

    def process_onejet(self, etas, phis, parts_p4, part_type, grid, tau_p4):
        part_var = part_var_list[part_type]
        list_part_var = np.zeros((len(part_var), grid.nCellsEta * grid.nCellsPhi))
        for eta, phi, part_p4 in zip(etas, phis, parts_p4):
            if eta is None or phi is None or tau_p4.pt == 0:  # it means eta, phi are not within inner or outer grid
                continue
            cellIndex = CellIndex(eta, phi)
            flatcellIndex = grid.GetFlatIndex(cellIndex)

            ##### fill electron variable ####
            if part_type == "ele":
                list_part_var[part_var["elept"]][flatcellIndex] += part_p4.pt / tau_p4.pt
                list_part_var[part_var["eleeta"]][flatcellIndex] += part_p4.eta
                list_part_var[part_var["elephi"]][flatcellIndex] += part_p4.phi
                list_part_var[part_var["elemass"]][flatcellIndex] += part_p4.mass
            #### fill gamma variable #####
            elif part_type == "gamma":
                list_part_var[part_var["gammapt"]][flatcellIndex] += part_p4.pt / tau_p4.pt
                list_part_var[part_var["gammaeta"]][flatcellIndex] += part_p4.eta
                list_part_var[part_var["gammaphi"]][flatcellIndex] += part_p4.phi
                list_part_var[part_var["gammamass"]][flatcellIndex] += part_p4.mass
            #### fill muon candidate #####
            elif part_type == "mu":
                list_part_var[part_var["mupt"]][flatcellIndex] += part_p4.pt / tau_p4.pt
                list_part_var[part_var["mueta"]][flatcellIndex] += part_p4.eta
                list_part_var[part_var["muphi"]][flatcellIndex] += part_p4.phi
                list_part_var[part_var["mumass"]][flatcellIndex] += part_p4.mass
            #### fill charged candidate variable ####
            elif part_type == "charge_candidate":
                list_part_var[part_var["chargedpt"]][flatcellIndex] += part_p4.pt / tau_p4.pt
                list_part_var[part_var["chargedeta"]][flatcellIndex] += part_p4.eta
                list_part_var[part_var["chargedphi"]][flatcellIndex] += part_p4.phi
                list_part_var[part_var["chargedmass"]][flatcellIndex] += part_p4.mass
            ### fill neutral candidate #####
            elif part_type == "neutral_candidate":
                list_part_var[part_var["neutralpt"]][flatcellIndex] += part_p4.pt / tau_p4.pt
                list_part_var[part_var["neutraleta"]][flatcellIndex] += part_p4.eta
                list_part_var[part_var["neutralphi"]][flatcellIndex] += part_p4.phi
                list_part_var[part_var["neutralmass"]][flatcellIndex] += part_p4.mass
        return list_part_var.reshape(-1)

    def processJets(self, data):
        write_info = {field: data[field] for field in data.fields}
        reco_tau_p4 = self.build_p4(data, "tau_p4s")
        signalcone = self.signalCone(reco_tau_p4.pt)
        for part in self.parttype:
            part_p4 = self.get_part_p4(part, data)
            dr = reco_tau_p4.deltaR(part_p4)
            deta = reco_tau_p4.eta - part_p4.eta
            dphi = reco_tau_p4.phi - part_p4.phi

            for cone in ["inner_grid", "outer_grid"]:
                grid = CellGrid(
                    self._builderConfig[cone]["n_cells"],
                    self._builderConfig[cone]["n_cells"],
                    self._builderConfig[cone]["cell_size"],
                    self._builderConfig[cone]["cell_size"],
                )
                if cone == "inner_grid":
                    part_inside_cone = ak.mask(part_p4, dr < signalcone)
                    maskdeta = ak.mask(deta, dr < signalcone)
                    maskdphi = ak.mask(dphi, dr < signalcone)
                else:
                    part_inside_cone = ak.mask(part_p4, dr > signalcone)
                    maskdeta = ak.mask(deta, dr > signalcone)
                    maskdphi = ak.mask(dphi, dr > signalcone)
                    part_inside_cone = ak.mask(part_inside_cone, dr < 0.5)
                    maskdeta = ak.mask(maskdeta, dr < 0.5)
                    maskdphi = ak.mask(maskdphi, dr < 0.5)
                etacellindex, phicellindex = grid.getcellIndex(maskdeta, maskdphi)
                assert(len(reco_tau_p4) == len(etacellindex))
                for idx, eta in enumerate(etacellindex):  # this loops over all jet?
                    list_part_info_perjet = self.process_onejet(eta, phicellindex[idx], part_inside_cone[idx], part, grid, reco_tau_p4[idx])
                    if idx == 0:
                        list_part_info_alljet = list_part_info_perjet
                    else:
                        list_part_info_alljet = np.concatenate((list_part_info_perjet, list_part_info_alljet), axis=-1)
                list_ak = ak.from_numpy(list_part_info_alljet)
                write_info.update({f"{cone}_{part}_block": list_ak})
                assert list_part_info_alljet.shape[0] == pow(self._builderConfig[cone]["n_cells"], 2) * len(
                    part_var_list[part]
                ) * len(reco_tau_p4)
        return write_info
