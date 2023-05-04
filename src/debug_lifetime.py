import os
import json
import hydra
import numpy as np
import general as g
import mplhep as hep
import awkward as ak
import matplotlib.pyplot as plt
from omegaconf import DictConfig

hep.style.use(hep.styles.CMS)

c = 299792458000  # mm/s
tau_lifetime = 2.903e-13  # s
tau_mass = 1.77  # GeV


def plot(values, title, output_path, xmin, xmax, print_rms=True, nbins=30):
    fig, ax = plt.subplots()
    rms = np.sqrt(ak.sum(values**2) * (1 / (len(values))))
    bins = np.linspace(xmin, xmax, num=30)
    hist, bin_edges = np.histogram(values, bins=bins)
    hist = hist / ak.sum(hist)
    hep.histplot(hist, bin_edges)
    plt.xlabel("mm", loc="center")
    plt.title(title)
    if print_rms:
        textstr = f"RMS={'{:0.3e}'.format(rms)}"
        props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.5}
        ax.text(0.6, 0.8, textstr, transform=ax.transAxes, fontsize=16, verticalalignment="top", bbox=props)
    plt.xlim((xmin, xmax))
    plt.savefig(output_path)
    plt.close("all")


def process_data(data, output_dir):
    tau_mask = data["gen_jet_tau_decaymode"] > 0
    data = data[tau_mask]
    gen_jet_tau_p4s = g.reinitialize_p4(data["gen_jet_full_tau_p4s"])
    gen_jet_tau_gamma = np.sqrt(1 + (gen_jet_tau_p4s.p / tau_mass) ** 2)
    expected_traveldistance = gen_jet_tau_gamma * c * tau_lifetime
    track_mask = abs(data["reco_cand_charge"]) > 0
    tau_descendant_mask = data["reco_cand_matched_gen_energy"] / g.reinitialize_p4(data["reco_cand_p4s"]).energy > 0.1
    suitable_cands_mask = track_mask * tau_descendant_mask
    gen_DV_dist = (
        np.sqrt(
            data["gen_jet_tau_decay_vertex_x"] ** 2
            + data["gen_jet_tau_decay_vertex_y"] ** 2
            + data["gen_jet_tau_decay_vertex_z"] ** 2
        )
        / expected_traveldistance
    )
    reco_pca_dist_from_reco_PV = np.sqrt(
        (data["reco_cand_PV_x"][suitable_cands_mask] - data["reco_cand_PCA_x"][suitable_cands_mask]) ** 2
        + (data["reco_cand_PV_y"][suitable_cands_mask] - data["reco_cand_PCA_y"][suitable_cands_mask]) ** 2
        + (data["reco_cand_PV_z"][suitable_cands_mask] - data["reco_cand_PCA_z"][suitable_cands_mask]) ** 2
    )
    reco_PV_from_gen_PV = np.sqrt(
        data["reco_cand_PV_x"][suitable_cands_mask] ** 2
        + data["reco_cand_PV_y"][suitable_cands_mask] ** 2
        + data["reco_cand_PV_z"][suitable_cands_mask] ** 2
    )
    reshaped_gen_DV_x = ak.ones_like((data["reco_cand_PCA_x"][suitable_cands_mask])) * data["gen_jet_tau_decay_vertex_x"]
    reshaped_gen_DV_y = ak.ones_like((data["reco_cand_PCA_y"][suitable_cands_mask])) * data["gen_jet_tau_decay_vertex_y"]
    reshaped_gen_DV_z = ak.ones_like((data["reco_cand_PCA_z"][suitable_cands_mask])) * data["gen_jet_tau_decay_vertex_z"]
    reco_PCA_from_gen_DV = np.sqrt(
        (data["reco_cand_PCA_x"][suitable_cands_mask] - reshaped_gen_DV_x) ** 2
        + (data["reco_cand_PCA_y"][suitable_cands_mask] - reshaped_gen_DV_y) ** 2
        + (data["reco_cand_PCA_z"][suitable_cands_mask] - reshaped_gen_DV_z) ** 2
    )
    reco_cand_pt = g.reinitialize_p4(data["reco_cand_p4s"][suitable_cands_mask]).pt
    high_pt_mask = reco_cand_pt > 20
    plot(
        values=ak.flatten(data["reco_cand_z0"][suitable_cands_mask], axis=-1),
        title="Reco Cand z0",
        output_path=os.path.join(output_dir, "reco_cand_z0.png"),
        xmin=-2e-1,
        xmax=2e-1,
    )
    plot(
        values=ak.flatten(data["reco_cand_d0"][suitable_cands_mask], axis=-1),
        title="Reco Cand d0",
        output_path=os.path.join(output_dir, "reco_cand_d0.png"),
        xmin=-2e-1,
        xmax=2e-1,
    )
    plot(
        values=ak.flatten(data["reco_cand_PV_x"][suitable_cands_mask], axis=-1),
        title="Reco Cand PV_x",
        output_path=os.path.join(output_dir, "PV_x.png"),
        xmin=-5e-5,
        xmax=5e-5,
    )
    plot(
        values=ak.flatten(data["reco_cand_PV_y"][suitable_cands_mask], axis=-1),
        title="Reco Cand PV_y",
        output_path=os.path.join(output_dir, "PV_y.png"),
        xmin=-5e-7,
        xmax=5e-7,
    )
    plot(
        values=ak.flatten(data["reco_cand_PV_z"][suitable_cands_mask], axis=-1),
        title="Reco Cand PV_z",
        output_path=os.path.join(output_dir, "PV_z.png"),
        xmin=-1e-2,
        xmax=1e-2,
    )
    plot(
        values=ak.flatten(gen_DV_dist, axis=-1),
        title="Gen_DV_from_PV_wrt_expected",
        output_path=os.path.join(output_dir, "gen_DV_dist.png"),
        xmin=-0,
        xmax=1,
        print_rms=True,
    )
    plot(
        values=ak.flatten(reco_PV_from_gen_PV, axis=-1),
        title="Reco_PV_from_gen_PV",
        output_path=os.path.join(output_dir, "reco_PV_from_gen_PV.png"),
        xmin=-1e-1,
        xmax=1e-1,
        print_rms=True,
    )

    plot(
        values=ak.flatten(reco_PCA_from_gen_DV, axis=-1),
        title="reco_PCA_from_gen_DV",
        output_path=os.path.join(output_dir, "reco_PCA_from_gen_DV.png"),
        xmin=0,
        xmax=1e1,
        print_rms=True,
    )

    plot(
        values=ak.flatten(reco_PCA_from_gen_DV[high_pt_mask], axis=-1),
        title="reco_PCA_from_gen_DV high_pt",
        output_path=os.path.join(output_dir, "reco_PCA_from_gen_DV_high_pT.png"),
        xmin=0,
        xmax=1e1,
        print_rms=True,
    )
    plot(
        values=ak.flatten(reco_PCA_from_gen_DV[~high_pt_mask], axis=-1),
        title="reco_PCA_from_gen_DV low pT",
        output_path=os.path.join(output_dir, "reco_PCA_from_gen_DV_low_pT.png"),
        xmin=0,
        xmax=1e1,
        print_rms=True,
    )

    plot(
        values=ak.flatten(reco_pca_dist_from_reco_PV / expected_traveldistance, axis=-1),
        title="Reco PCA from reco PV wrt. expectation",
        output_path=os.path.join(output_dir, "reco_PCA_from_PV.png"),
        xmin=0,
        xmax=0.3,
    )
    plot(
        values=ak.flatten(data["gen_jet_tau_decay_vertex_x"], axis=-1),
        title="Gen_DV_x",
        output_path=os.path.join(output_dir, "Gen_DV_x.png"),
        xmin=-1,
        xmax=1,
        print_rms=False,
    )
    plot(
        values=ak.flatten(data["gen_jet_tau_decay_vertex_y"], axis=-1),
        title=" Gen_DV_y",
        output_path=os.path.join(output_dir, "Gen_DV_y.png"),
        xmin=-1,
        xmax=1,
        print_rms=False,
    )
    plot(
        values=ak.flatten(data["gen_jet_tau_decay_vertex_z"], axis=-1),
        title="Gen_DV_z",
        output_path=os.path.join(output_dir, "Gen_DV_z.png"),
        xmin=-1,
        xmax=1,
        print_rms=False,
    )
    distribution_in_tau_direction(gen_jet_tau_p4s, data, suitable_cands_mask, output_dir)


def distribution_in_tau_direction(gen_jet_tau_p4s, data, suitable_cands_mask, output_dir):
    reshaped_tau_vec_x = ak.ones_like((data["reco_cand_PCA_x"][suitable_cands_mask])) * gen_jet_tau_p4s.px
    reshaped_tau_vec_y = ak.ones_like((data["reco_cand_PCA_y"][suitable_cands_mask])) * gen_jet_tau_p4s.py
    reshaped_tau_vec_z = ak.ones_like((data["reco_cand_PCA_z"][suitable_cands_mask])) * gen_jet_tau_p4s.pz
    tau_vec = np.array(
        (
            ak.flatten(reshaped_tau_vec_x, axis=-1),
            ak.flatten(reshaped_tau_vec_y, axis=-1),
            ak.flatten(reshaped_tau_vec_z, axis=-1),
        )
    ).T
    reshaped_gen_DV_x = ak.ones_like((data["reco_cand_PCA_x"][suitable_cands_mask])) * data["gen_jet_tau_decay_vertex_x"]
    reshaped_gen_DV_y = ak.ones_like((data["reco_cand_PCA_y"][suitable_cands_mask])) * data["gen_jet_tau_decay_vertex_y"]
    reshaped_gen_DV_z = ak.ones_like((data["reco_cand_PCA_z"][suitable_cands_mask])) * data["gen_jet_tau_decay_vertex_z"]
    pca_diff_vec = np.array(
        (
            ak.flatten((data["reco_cand_PCA_x"][suitable_cands_mask] - reshaped_gen_DV_x), axis=-1),
            ak.flatten((data["reco_cand_PCA_y"][suitable_cands_mask] - reshaped_gen_DV_y), axis=-1),
            ak.flatten((data["reco_cand_PCA_z"][suitable_cands_mask] - reshaped_gen_DV_z), axis=-1),
        )
    ).T
    pv_diff_vec = np.array(
        (
            ak.flatten(data["reco_cand_PCA_x"][suitable_cands_mask], axis=-1),
            ak.flatten(data["reco_cand_PCA_y"][suitable_cands_mask], axis=-1),
            ak.flatten(data["reco_cand_PCA_z"][suitable_cands_mask], axis=-1),
        )
    ).T

    # In tau direction
    PCA_diff_tau_direction = []
    for pcadv, tv in zip(pca_diff_vec, tau_vec):
        PCA_diff_tau_direction.append(np.dot(pcadv, tv) / np.linalg.norm(tv))

    PV_diff_tau_direction = []
    for PVdtd, tv in zip(pv_diff_vec, tau_vec):
        PV_diff_tau_direction.append(np.dot(PVdtd, tv) / np.linalg.norm(tv))
    plot(
        values=np.array(PCA_diff_tau_direction),
        title="PCA_diff_tau_direction",
        output_path=os.path.join(output_dir, "PCA_diff_tau_direction.png"),
        xmin=-1e1,
        xmax=1e1,
        print_rms=False,
    )
    plot(
        values=np.array(PV_diff_tau_direction),
        title="PV_diff_tau_direction",
        output_path=os.path.join(output_dir, "PV_diff_tau_direction.png"),
        xmin=-1e-1,
        xmax=1e-1,
        print_rms=False,
    )

    PV_diff_perp_tau_x = []
    PV_diff_perp_tau_y = []
    PCA_diff_perp_tau_x = []
    PCA_diff_perp_tau_y = []
    for tv, pvdv, pcadv in zip(tau_vec, pv_diff_vec, pca_diff_vec):
        perp_x = np.random.randn(3)
        # Perpendicular to tau direction
        perp_x -= perp_x.dot(tv) * tv / np.linalg.norm(tv) ** 2  # First
        perp_y = np.cross(tv, perp_x)  # Second
        PV_diff_perp_tau_x.append(np.dot(pvdv, perp_x) / np.linalg.norm(perp_x))
        PV_diff_perp_tau_y.append(np.dot(pvdv, perp_y) / np.linalg.norm(perp_y))
        PCA_diff_perp_tau_x.append(np.dot(pcadv, perp_x) / np.linalg.norm(perp_x))
        PCA_diff_perp_tau_y.append(np.dot(pcadv, perp_y) / np.linalg.norm(perp_y))

    plot(
        values=np.array(PV_diff_perp_tau_x),
        title="PV_diff_perp_tau_x",
        output_path=os.path.join(output_dir, "PV_diff_perp_tau_x.png"),
        xmin=-1e-1,
        xmax=1e-1,
        print_rms=True,
    )
    # print("PV_x <= 0.02", np.sum(abs(np.array(PV_diff_perp_tau_x)) <= 0.02))
    plot(
        values=np.array(PV_diff_perp_tau_y),
        title="PV_diff_perp_tau_y",
        output_path=os.path.join(output_dir, "PV_diff_perp_tau_y.png"),
        xmin=-1e-1,
        xmax=1e-1,
        print_rms=True,
    )
    # print("PV_y <= 0.02", np.sum(abs(np.array(PV_diff_perp_tau_y)) <= 0.02))
    plot(
        values=np.array(PCA_diff_perp_tau_x),
        title="PCA_diff_perp_tau_x",
        output_path=os.path.join(output_dir, "PCA_diff_perp_tau_x.png"),
        xmin=-5e-1,
        xmax=5e-1,
        print_rms=True,
    )
    print("PCA_x <= 0.02", np.sum(abs(np.array(PCA_diff_perp_tau_x)) <= 0.02) / len(PCA_diff_perp_tau_x))
    plot(
        values=np.array(PCA_diff_perp_tau_y),
        title="PCA_diff_perp_tau_y",
        output_path=os.path.join(output_dir, "PCA_diff_perp_tau_y.png"),
        xmin=-5e-1,
        xmax=5e-1,
        print_rms=True,
    )
    print("PCA_y <= 0.02", np.sum(abs(np.array(PCA_diff_perp_tau_y)) <= 0.02) / len(PCA_diff_perp_tau_y))

    for i in range(len(PCA_diff_perp_tau_y)):
        dict_ = {
            "L_PCA_DV_diff": PCA_diff_tau_direction[i],
            "P_PCA_DV_diff_x": PCA_diff_perp_tau_x[i],
            "P_PCA_DV_diff_y": PCA_diff_perp_tau_y[i],
            "L_PV_diff": PV_diff_tau_direction[i],
            "P_PV_diff_x": PV_diff_perp_tau_x[i],
            "P_PV_diff_y": PV_diff_perp_tau_y[i],
        }
        print(json.dumps(dict_, indent=4))
        print("----------------------------------------------------")
        if i > 5:
            break

    outlier_mask = np.array(PCA_diff_perp_tau_y) > 2
    cand_p4s = ak.flatten(g.reinitialize_p4(data["reco_cand_p4s"][suitable_cands_mask]))[outlier_mask]
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    for i in range(len(cand_p4s)):
        print(f"pT: {cand_p4s.pt[i]} \t eta: {cand_p4s.eta[i]}")
        if i > 5:
            break


@hydra.main(config_path="../config", config_name="lifetime_verification", version_base=None)
def study_lifetime_resolution(cfg: DictConfig) -> None:
    for sample in cfg.samples.keys():
        data = g.load_all_data(cfg.samples[sample].input_dir, n_files=cfg.n_files)
        output_dir = os.path.join(cfg.samples[sample].input_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        process_data(data, output_dir)


# x = ROOT.RooRealVar("x", "x", 1, 0, 10)
# mean = ROOT.RooRealVar("mean", "Mean of Gaussian", 1, 0, 10)
# width = ROOT.RooRealVar("width", "With of Gaussian", 1, 0, 10)
# c = ROOT.RooRealVar("c", "Constant for exponential", -1.54, -10, -0.01)
# g = ROOT.RooGaussian("g", "Gaussian", x, mean, width)
# e = ROOT.RooExponential("e", "Exponential", x, c)
# conv = ROOT.RooFFTConvPdf("eXg", "Exponential (X) Gauss", x, e, g)

# hist = ROOT.TH1D("data", "data", 35, 0, 10)
# for d in scaled_by_lifetime:
#     hist.Fill(d)
# data_ = ROOT.RooDataHist("d2", "d2", x, hist)
# conv.fitTo(data_)


if __name__ == "__main__":
    study_lifetime_resolution()
