from endtoend_simple import TauEndToEndSimple
import torch
import hls4ml

if __name__ == "__main__":
    pytorch_model = torch.load(
       "/scratch-persistent/joosep/ml-tau-reco/CLIC_data/2023_03_16/SimpleDNN/model.pt",
       map_location=torch.device("cpu")
    )
    assert pytorch_model.__class__ == TauEndToEndSimple
    print(pytorch_model)
    
