import awkward as ak
import numpy as np
import vector
import json
import os
import time
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from math import pi

from part_var import Var
from grid import CellGrid
from basicTauBuilder import BasicTauBuilder


class GridBuilder(BasicTauBuilder):
    def __init__(self, cfgFileName="./config/grid_builder.json", verbosity=0):
        super(BasicTauBuilder, self).__init__()
        cfgFile = open(cfgFileName, "r")
        cfg = json.load(cfgFile)
        self._builderConfig = cfg["GridAlgo"]
        cfgFile.close()
        self.num_particles_in_grid = cfg["num_particles_in_grid"]
        self.do_plot = cfg["plot"]
        self.max_var = Var.max_value()

    def build_p4(self, part=""):
        if part:
            return vector.awk(
                ak.zip(
                    {"px": self.data[part].x, "py": self.data[part].y, "pz": self.data[part].z, "mass": self.data[part].tau}
                )
            )
        return vector.awk(
            ak.zip(
                {
                    "px": self.data.event_reco_cand_p4s.x,
                    "py": self.data.event_reco_cand_p4s.y,
                    "pz": self.data.event_reco_cand_p4s.z,
                    "mass": self.data.event_reco_cand_p4s.tau,
                }
            )
        )

    def signalCone(self):
        minpt = 30
        minr = 0.05
        cone = 3
        return np.maximum(cone / np.maximum(self.reco_tau_p4.pt, minpt), minr)

    def process_onejet(self, etas, phis, inner):
        list_part_var = np.zeros((self.max_var * self.num_particles_in_grid, self.grid.nCellsEta, self.grid.nCellsPhi))
        if self.reco_tau_p4.pt[self.jetidx] == 0:
            return list_part_var
        self.filledgrid = np.zeros((self.grid.nCellsEta, self.grid.nCellsPhi), dtype=int)
        for idx, (eta, phi) in enumerate(zip(etas, phis)):
            etaidx, phiidx = self.bin_idx_eta[self.jetidx][idx], self.bin_idx_phi[self.jetidx][idx]
            self.images_count[self.jetidx][etaidx, phiidx] += 1
            if (self.filledgrid[etaidx, phiidx]) >= self.num_particles_in_grid:
                continue
            filledpartidx = self.filledgrid[etaidx, phiidx]
            offset = filledpartidx * self.max_var - 1
            list_part_var[offset + Var.rel_pt.value][etaidx, phiidx] = (
                self.pt_sorted_cand_pt[self.pt_sorted_cone_mask > 0][self.jetidx][idx]
            ) / self.reco_tau_p4.pt[self.jetidx]
            list_part_var[offset + Var.dtheta.value][etaidx, phiidx] = self.pt_sorted_cand_dtheta[
                self.pt_sorted_cone_mask > 0
            ][self.jetidx][idx]
            list_part_var[offset + Var.dphi.value][etaidx, phiidx] = self.pt_sorted_cand_dphi[self.pt_sorted_cone_mask > 0][
                self.jetidx
            ][idx]
            list_part_var[offset + Var.mass.value][etaidx, phiidx] = self.pt_sorted_cand_mass[self.pt_sorted_cone_mask > 0][
                self.jetidx
            ][idx]
            list_part_var[offset + Var.charge.value][etaidx, phiidx] = self.pt_sorted_cand_charge[
                self.pt_sorted_cone_mask > 0
            ][self.jetidx][idx]
            list_part_var[offset + Var.dxy.value][etaidx, phiidx] = self.pt_sorted_cand_dxy[self.pt_sorted_cone_mask > 0][
                self.jetidx
            ][idx]
            list_part_var[offset + Var.dxy_sig.value][etaidx, phiidx] = self.pt_sorted_cand_dxy_sig[
                self.pt_sorted_cone_mask > 0
            ][self.jetidx][idx]
            list_part_var[offset + Var.dz.value][etaidx, phiidx] = self.pt_sorted_cand_dz[self.pt_sorted_cone_mask > 0][
                self.jetidx
            ][idx]
            list_part_var[offset + Var.dz_sig.value][etaidx, phiidx] = self.pt_sorted_cand_dz_sig[
                self.pt_sorted_cone_mask > 0
            ][self.jetidx][idx]
            list_part_var[offset + Var.isele.value][etaidx, phiidx] = self.pt_sorted_cand_isele[
                self.pt_sorted_cone_mask > 0
            ][self.jetidx][idx]
            list_part_var[offset + Var.ismu.value][etaidx, phiidx] = self.pt_sorted_cand_ismu[self.pt_sorted_cone_mask > 0][
                self.jetidx
            ][idx]
            list_part_var[offset + Var.isch.value][etaidx, phiidx] = self.pt_sorted_cand_isch[self.pt_sorted_cone_mask > 0][
                self.jetidx
            ][idx]
            list_part_var[offset + Var.isnh.value][etaidx, phiidx] = self.pt_sorted_cand_isnh[self.pt_sorted_cone_mask > 0][
                self.jetidx
            ][idx]
            list_part_var[offset + Var.isgamma.value][etaidx, phiidx] = self.pt_sorted_cand_isgamma[
                self.pt_sorted_cone_mask > 0
            ][self.jetidx][idx]
            self.filledgrid[etaidx, phiidx] += 1
        return list_part_var

    def calculate_dangle(self):
        return np.sqrt(
            self.pt_sorted_cand_dtheta * self.pt_sorted_cand_dtheta + self.pt_sorted_cand_dphi * self.pt_sorted_cand_dphi
        )

    def deltaphi(self, phi1, phi2):
        diff = phi1 - phi2
        return np.arctan2(np.sin(diff), np.cos(diff))

    def mask_dphi(self):
        mask_greater = 2 * pi - self.pt_sorted_cand_dphi
        self.pt_sorted_cand_dphi = ak.where(self.pt_sorted_cand_dphi > pi, mask_greater, self.pt_sorted_cand_dphi)
        mask_less = 2 * pi + self.pt_sorted_cand_dphi
        self.pt_sorted_cand_dphi = ak.where(self.pt_sorted_cand_dphi <= -pi, mask_less, self.pt_sorted_cand_dphi)

    def mask_pdgid(self, pdgid):
        return ak.where(
            self.pt_sorted_cand_pdgid == pdgid,
            ak.ones_like(self.pt_sorted_cand_pdgid),
            ak.zeros_like(self.pt_sorted_cand_pdgid),
        )

    def calcuclate_sig(self, neu, deno):
        sig = neu / deno
        return ak.where(deno == 0, ak.zeros_like(deno), sig)

    def pt_sorted_ftrs(self):
        self.part_p4 = self.build_p4("event_reco_cand_p4s")
        self.pt_sorted_idx = ak.argsort(self.part_p4.pt, ascending=False)
        self.pt_sorted_cand_p4 = self.part_p4[self.pt_sorted_idx]
        self.pt_sorted_cand_pt = self.pt_sorted_cand_p4.pt
        self.pt_sorted_cand_theta = self.pt_sorted_cand_p4.theta
        self.pt_sorted_cand_phi = self.pt_sorted_cand_p4.phi
        self.pt_sorted_cand_mass = self.pt_sorted_cand_p4.mass
        self.pt_sorted_cand_charge = self.data.event_reco_cand_charge[self.pt_sorted_idx]
        self.pt_sorted_cand_dxy = self.data.event_reco_cand_dxy[self.pt_sorted_idx]
        self.pt_sorted_cand_dxy_err = self.data.event_reco_cand_dxy_err[self.pt_sorted_idx]
        self.pt_sorted_cand_dxy_sig = self.calcuclate_sig(self.pt_sorted_cand_dxy, self.pt_sorted_cand_dxy_err)
        self.pt_sorted_cand_dz = self.data.event_reco_cand_dz[self.pt_sorted_idx]
        self.pt_sorted_cand_dz_err = self.data.event_reco_cand_dz_err[self.pt_sorted_idx]
        self.pt_sorted_cand_dz_sig = self.calcuclate_sig(self.pt_sorted_cand_dz, self.pt_sorted_cand_dz_err)
        self.pt_sorted_cand_pdgid = self.data.event_reco_cand_pdg[self.pt_sorted_idx]
        self.pt_sorted_cand_isele = np.abs(self.pt_sorted_cand_pdgid) == 13
        self.pt_sorted_cand_ismu = np.abs(self.pt_sorted_cand_pdgid) == 11
        self.pt_sorted_cand_isch = self.pt_sorted_cand_pdgid == 211
        self.pt_sorted_cand_isnh = self.pt_sorted_cand_pdgid == 130
        self.pt_sorted_cand_isgamma = self.pt_sorted_cand_pdgid == 22
        self.pt_sorted_cand_dtheta = self.reco_tau_p4.theta - self.pt_sorted_cand_theta
        self.pt_sorted_cand_dphi = self.reco_tau_p4.phi - self.pt_sorted_cand_p4.phi
        self.mask_dphi()
        self.pt_sorted_cand_dangle = self.calculate_dangle()

    def get_cone_mask(self, inner_grid):
        if inner_grid:
            return ak.where(
                self.pt_sorted_cand_dangle < self.signalcone,
                ak.ones_like(self.pt_sorted_cand_dangle),
                ak.zeros_like(self.pt_sorted_cand_dangle),
            )
        else:
            mask_outside_signal = ak.where(
                self.pt_sorted_cand_dangle > self.signalcone,
                ak.ones_like(self.pt_sorted_cand_dangle),
                ak.zeros_like(self.pt_sorted_cand_dangle),
            )
            mask_inside_isolation = ak.where(
                self.pt_sorted_cand_dangle < 0.5,
                ak.ones_like(self.pt_sorted_cand_dangle),
                ak.zeros_like(self.pt_sorted_cand_dangle),
            )
            total_mask = mask_outside_signal + mask_inside_isolation
            total_mask = ak.where(
                total_mask == 2, ak.ones_like(self.pt_sorted_cand_dangle), ak.zeros_like(self.pt_sorted_cand_dangle)
            )
            return total_mask

    def plot(self, cone):
        plt.hist(ak.flatten(self.maskdphi), bins=self.bins, range=(-self.bin, self.bin))
        plt.savefig(f"dphi_{cone}.png")
        plt.close("all")
        plt.hist(ak.flatten(self.maskdeta), bins=self.bins, range=(-self.bin, self.bin))
        plt.savefig(f"dtheta_{cone}.png")
        plt.close("all")
        plt.imshow(np.average(self.images_count, axis=0))
        plt.colorbar()
        plt.savefig(f"test_{cone}.png")
        plt.close("all")

    def processJets(self, data):
        self.data = data
        write_info = {field: data[field] for field in data.fields}
        self.reco_tau_p4 = self.build_p4("tau_p4s")
        self.signalcone = self.signalCone()
        self.pt_sorted_ftrs()
        for cone in ["inner_grid", "outer_grid"]:
            self.grid = CellGrid(
                self._builderConfig[cone]["n_cells"],
                self._builderConfig[cone]["n_cells"],
                self._builderConfig[cone]["cell_size"],
                self._builderConfig[cone]["cell_size"],
            )
            self.bin = self.grid.MaxDeltaEta()
            self.bins = self.grid.nCellsEta
            self.bins_eta = np.linspace(-self.bin, self.bin, self.bins)
            self.bins_phi = np.linspace(-self.bin, self.bin, self.bins)
            self.images_count = np.zeros((len(self.reco_tau_p4), len(self.bins_eta) + 1, len(self.bins_phi) + 1))
            self.pt_sorted_cone_mask = self.get_cone_mask(cone == "inner_grid")
            self.maskdeta, self.maskdphi = (
                self.pt_sorted_cand_dtheta[self.pt_sorted_cone_mask > 0],
                self.pt_sorted_cand_dphi[self.pt_sorted_cone_mask > 0],
            )
            self.bin_idx_eta = ak.unflatten(
                np.searchsorted(self.bins_eta, ak.flatten(self.maskdeta)), ak.count(self.maskdeta, axis=-1), axis=-1
            )
            self.bin_idx_phi = ak.unflatten(
                np.searchsorted(self.bins_phi, ak.flatten(self.maskdphi)), ak.count(self.maskdphi, axis=-1), axis=-1
            )
            grid_all_jets = []
            for idx, eta in enumerate(self.bin_idx_eta):  # this loops over all jet?
                self.jetidx = idx
                list_part_info_perjet = self.process_onejet(self.maskdeta[idx], self.maskdphi[idx], cone)
                grid_all_jets.append(list_part_info_perjet)

            list_part_info_alljet = np.stack(grid_all_jets, axis=0)
            list_ak = ak.from_numpy(list_part_info_alljet)
            write_info.update({f"{cone}": list_ak})
            if self.do_plot:
                self.plot(cone)
        print("building grid is finished on # of jets: ", len(self.reco_tau_p4))
        return write_info


if __name__ == "__main__":
    grid = GridBuilder()
    inputfile = "/scratch/persistent/veelken/CLIC_tau_ntuples/2023Mar18_woPtCuts/\
    HPS/ZH_Htautau/reco_p8_ee_ZH_Htautau_ecm380_200001.parquet"
    data = ak.from_parquet(inputfile)
    print("Time: ", time.strftime("%H:%M"))
    data = grid.processJets(data)
    outputdir = "grid/ZH_Htautau"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print("Time: ", time.strftime("%H:%M"))
    ak.to_parquet(ak.Record(data), os.path.join(outputdir, os.path.basename(inputfile)))
