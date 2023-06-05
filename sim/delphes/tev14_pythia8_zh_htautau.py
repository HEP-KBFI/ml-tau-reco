# based on Pythia8_A14_NNPDF23LO_Common.py
# and https://atlaswww.hep.anl.gov/hepsim/info.php?item=281
# HepSim Pythia setting
# J. Duarte
# apply particle slim?
ApplyParticleSlim=off
#
# Collision settings
EventsNumber=5000
Random:setSeed = on
Random:seed = 0
Beams:idA = 2212
Beams:idB = 2212
Beams:eCM = 14000.

HardQCD:all = off
HiggsSM:ffbar2HZ = on
25:m0        = 125.0
25:onMode    = off
25:onIfAny   = 15
