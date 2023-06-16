from enum import Enum, auto


class Var(Enum):
    rel_pt = auto()
    dtheta = auto()
    dphi = auto()
    mass = auto()
    charge = auto()
    dxy = auto()
    dxy_sig = auto()
    dz = auto()
    dz_sig = auto()
    d0 = auto()
    d0_sig = auto()
    isele = auto()
    ismu = auto()
    isch = auto()
    isnh = auto()
    ##### it should be last variable#####
    isgamma = auto()

    @classmethod
    def max_value(cls):
        return cls.isgamma.value
