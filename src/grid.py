import numpy as np
import awkward as ak

class CellIndex():

    def __init__(self, e, p):
        self.eta = e
        self.phi = p

    def etaval(self):
        return self.eta

    def phival(self):
        return self.phi

class CellGrid():
    def __init__(self, nCellsEta, nCellsPhi, cellSizeEta, cellSizePhi):
        self.nCellsEta = nCellsEta
        self.nCellsPhi = nCellsPhi
        self.cellSizeEta = cellSizeEta
        self.cellSizePhi = cellSizePhi
        self.nTotal = nCellsEta * nCellsPhi

    def MaxEtaIndex(self):
        return int(self.nCellsEta/2.)

    def MaxPhiIndex(self):
        return int(self.nCellsPhi/2.)

    def MaxDeltaEta(self):
        return self.cellSizeEta * (0.5 * self.nCellsEta)

    def MaxDeltaPhi(self):
        return self.cellSizePhi * (0.5 * self.nCellsPhi)

    def GetFlatIndex(self, cellIndex):
        if abs(cellIndex.etaval()) > self.MaxEtaIndex() or abs(cellIndex.phival()) > self.MaxPhiIndex():
            print('Cell index is out of range')
            assert(0)
        shiftedEta = int(cellIndex.etaval()) + self.MaxEtaIndex()
        shiftedPhi = int(cellIndex.phival()) + self.MaxPhiIndex()
        return shiftedEta * self.nCellsPhi + shiftedPhi

    def GetnTotal(self):
        return self.nTotal

    def at(self, cellIndex):
        return self.cells[self.GetFlatIndex(cellIndex)]
        
    def TryCellIndex(self, vars, maxval, size):
        absvars = np.absolute(vars)
        maskabsvars = ak.mask(absvars, absvars < maxval)
        return np.copysign(ak.values_astype(maskabsvars/size +0.5, np.int32), vars)

    def getcellIndex(self, eta, phi):
        return self.TryCellIndex(eta, self.MaxDeltaEta(), self.cellSizeEta), self.TryCellIndex(phi, self.MaxDeltaPhi(), self.cellSizePhi)
