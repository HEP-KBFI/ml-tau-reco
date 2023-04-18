import numpy as np
import awkward as ak
import vector
import torch
from torch_geometric.data.batch import Batch
from basicTauBuilder import BasicTauBuilder
from taujetdataset_withgrid import TauJetDatasetWithGrid


class DeepTauBuilder(BasicTauBuilder):
    def __init__(
        self,
        model,
        config={},
    ):
        self.model = model
        model.eval()
        self._builderConfig = dict()

    def processJets(self, data):
        ds = TauJetDatasetWithGrid()
        data_obj = Batch.from_data_list(ds.process_file_data(data), exclude_keys=["gen_tau_decaymode", "gen_tau_p4"])
        pred_istau = self.model(data_obj)
        pred_istau = list(torch.softmax(pred_istau, axis=-1)[:, 1].contiguous().detach().numpy())
        njets = len(data["tau_p4s"])
        # dummy placeholders for now
        tauCharges = np.zeros(njets)
        dmode = np.zeros(njets)
        # as a dummy placeholder, just return the first PFCand for each jet
        tau_cand_p4s = data["reco_cand_p4s"][:, 0:1]
        jet_p4s = vector.awk(
            ak.zip(
                {
                    "px": pred_p4[:, 0],
                    "py": pred_p4[:, 1],
                    "pz": pred_p4[:, 2],
                    "mass": pred_p4[:, 3],
                }
            )
        )
        return {
            "jet_p4s": jet_p4s,
            "tau_p4s": tau_p4s,
            "tauSigCand_p4s": tau_cand_p4s,
            "tauClassifier": pred_istau,
            "tau_charge": tauCharges,
            "tau_decaymode": dmode,
        }
