from endtoend_simple import TauEndToEndSimple
import torch
import hls4ml
import yaml

from torch_geometric.data.batch import Batch
from taujetdataset import TauJetDataset


def get_split_files(config_path, split):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]
        return paths


if __name__ == "__main__":
    print(hls4ml.__version__)

    pytorch_model = torch.load("data/model.pt", map_location=torch.device("cpu"))

    # temporary workaround to disable aggregation layers,
    # which otherwise gives aten::scatter_reduce not supported
    pytorch_model.onnx_workaround_agg = True
    pytorch_model.agg1 = None
    pytorch_model.agg2 = None
    pytorch_model.agg3 = None

    assert pytorch_model.__class__ == TauEndToEndSimple
    print(pytorch_model)

    test_files = "config/datasets/test.yaml"

    # prepare the input data from one file
    files_test = get_split_files(test_files, "test")[:1]
    ds = TauJetDataset(files_test)
    data_obj = Batch.from_data_list(ds.all_data, follow_batch=["jet_pf_features"])
    x = (data_obj.jet_features, data_obj.jet_pf_features, data_obj.jet_pf_features_batch)

    # run the model in forward mode
    tau_id, tau_p4 = pytorch_model(*x)
    print(tau_id)

    # this at least does not crash
    torch.onnx.export(
        pytorch_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "model.onnx",
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
    )

    # test if this works, and what needs to change in the model for this to work
    hls4ml.converters.convert_from_pytorch_model(pytorch_model, ((None, 8), (None, 36), (None, 1)))
