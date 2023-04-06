from endtoend_simple import TauEndToEndSimple
import torch
import hls4ml
import torch_geometric

from torch_geometric.data.batch import Batch
from taujetdataset import TauJetDataset, get_split_files


if __name__ == "__main__":
    print(hls4ml.__version__)

    pytorch_model = torch.load("data/model.pt", map_location=torch.device("cpu"))

    # temporary workaround to disable aggregation layers,
    # which otherwise gives aten::scatter_reduce not supported

    assert pytorch_model.__class__ == TauEndToEndSimple
    pytorch_model.agg1 = None
    pytorch_model.agg2 = None
    pytorch_model.agg3 = None
    pytorch_model.sparse_mode = False
    print(pytorch_model)

    test_files = "config/datasets/test.yaml"

    # prepare the input data from one file
    files_test = get_split_files(test_files, "test")[:1]
    files_test = [f.replace("/scratch/persistent/laurits", "data") for f in files_test]
    ds = TauJetDataset(files_test)
    data_obj = Batch.from_data_list(ds.all_data, follow_batch=["jet_pf_features"])

    # pad to 3D: [njet, n_max_pf_per_jet, nfeat]
    pfs_padded, mask = torch_geometric.utils.to_dense_batch(data_obj.jet_pf_features, data_obj.jet_pf_features_batch, 0.0)

    # pad jet features to the same dim as PF candidate features
    jet_feat_pad = torch.nn.functional.pad(data_obj.jet_features, [0, 36 - 8])
    # stack the jet features to the PF candidate feature matrix
    jet_and_pf = torch.concat([jet_feat_pad.unsqueeze(dim=1), pfs_padded], axis=1)  # [njet, n_max_pf_per_jet + 1, nfeat]

    # run the model in forward mode
    ret = pytorch_model(jet_and_pf)
    print(ret)

    # this at least does not crash
    torch.onnx.export(
        pytorch_model,  # model being run
        jet_and_pf,  # model input (or a tuple for multiple inputs)
        "model.onnx",
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
    )

    # test if this works, and what needs to change in the model for this to work
    config = hls4ml.utils.config_from_pytorch_model(pytorch_model)
    print(config)

    hls_model = hls4ml.converters.convert_from_pytorch_model(pytorch_model, jet_and_pf.shape, hls_config=config)
    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="test.png")

    hls_model.compile()
    hls4ml_pred, hls4ml_trace = hls_model.trace(jet_and_pf.numpy())
    print(hls4ml_pred)
