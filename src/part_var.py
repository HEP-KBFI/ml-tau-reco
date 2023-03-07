from enum import Enum, auto


class Var(Enum):
    rel_pt = auto()
    dtheta = auto()
    dphi = auto()
    mass = auto()

    ##### it should be last variable#####
    pdgid = auto()

    @classmethod
    def max_value(cls):
        return cls.pdgid.value
