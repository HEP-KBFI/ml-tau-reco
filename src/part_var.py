from enum import Enum, auto


class Var(Enum):
    rel_pt = auto()
    dtheta = auto()
    dphi = auto()
    mass = auto()
    charge = auto()
    dz_f2D = auto()
    dz_f2D_err = auto()
    dxy_f2D = auto()
    dxy_f2D_err = auto()
    isele = auto()
    ismu = auto()
    isch = auto()
    isnh = auto()
    ##### it should be last variable#####
    isgamma = auto()

    @classmethod
    def max_value(cls):
        return cls.isgamma.value
