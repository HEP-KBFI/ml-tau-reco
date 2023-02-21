"""
Helper functions tom get additional training inputs such as impact parameters
and calometric energy deposits. For the impact parameters a linear extrapolation
of the helix at the refference point is used. The definitions of track parameters
are taken from:
[1] https://flc.desy.de/lcnotes/notes/localfsExplorer_read?currentPath=/afs/desy.de/group/flc/lcnotes/LC-DET-2006-004.pdf
Math for finding the PCA from:
[2] https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

Author: Torben Lange (KBFI)
Date: 20.02.2022
"""

import numpy as np
import math

"""
Helper calulating the PCA to the refference point, see [1] for mor details.
"""


def calcX0(xr, phi0, d0):
    alpha = 0.5 * math.pi - phi0
    return xr + math.cos(alpha) * d0


"""
Helper calulating the PCA to the refference point, see [1] for mor details.
"""


def calcY0(yr, phi0, d0):
    alpha = 0.5 * math.pi - phi0
    return yr - math.sin(alpha) * d0


"""
Helper calulating the Y component corresponding to a given X for a linear interpolation of the track
around the refference point.
"""


def calcY(xr, yr, phi0, d0, x):
    x0 = calcX0(xr, phi0, d0)
    y0 = calcY0(yr, phi0, d0)
    return math.tan(phi0) * (x - x0) + y0


"""
Helper calculating the arc length of the track in the xy plane, needed to calculate the z component.
"""


def calcS(x, x0, y, y0, omega):
    return math.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)) * np.sign(omega)


"""
Helper calulating the Z component corresponding to a given X and Y for a linear interpolation of the track
around the refference point. See Eq. 10 in [1].
"""


def calcZ(x, y, pr, d0, z0, phi0, tanL, omega):
    xr, yr, zr = pr
    x0 = calcX0(xr, phi0, d0)
    y0 = calcY0(yr, phi0, d0)
    s = calcS(x, x0, y, y0, omega)
    return (z0 + zr) + s * tanL


"""
Helper, calculating the full coordinates of a point on the linear extrapolation of a track around its
refference point for a given X using the previous helper functions.
"""


def calcP(x, pr, d0, z0, phi0, tanL, omega):
    xr, yr, zr = pr
    y = calcY(xr, yr, phi0, d0, x)
    z = calcZ(x, y, pr, d0, z0, phi0, tanL, omega)
    return [x, y, z]


"""
Helper for the linear extrapolation of a track using an interpolation between two points
based on a parameter t. Used to find the PCA to the PV. See [2] for more details.
"""


def calcV(p1, p2, t):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    xn = x1 + (x2 - x1) * t
    yn = y1 + (y2 - y1) * t
    zn = z1 + (z2 - z1) * t
    return [xn, yn, zn]


"""
Helper, calculating the parameter value of t, for the PCA to the PV using a linear
extrapolation for two points on the assumed track parametrized in t (see calcV).
Formula taken from Eq. 3 in [2].
"""


def calcTPCA(p0, p1, p2):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    num = (x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)
    den = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1)
    return -num / den


"""
Helper calculating the PCA of the linear extrapolated track to the PV.
For more details see [2].
"""


def calcPCA(pr, d0, z0, phi0, tanL, omega, vertex):
    p1 = calcP(pr[0] + 0.01, pr, d0, z0, phi0, tanL, omega)
    p2 = calcP(pr[0] - 0.01, pr, d0, z0, phi0, tanL, omega)
    tPCA = calcTPCA(vertex, p1, p2)
    return calcV(p1, p2, tPCA)


"""
Finds the PCAs w.r.t the primary vertex for the tracks corresponding to a RecoParticle collection
for a given event (ev) and returns basic impact parameters for them, aswell as the PCAs. The parameter
frame should be an awkward record containing the required collections specified for trackCollection,
recoParticleCollection and vertexCollection.
"""


def findTrackPCAs(
    frame,
    ev,
    recoParticleCollection="MergedRecoParticles",
    trackCollection="SiTracks_Refitted_1",
    vertexCollection="PrimaryVertices",
    debug=-1,
):
    vertex = [
        frame[vertexCollection][ev][vertexCollection + ".position.x"][0],
        frame[vertexCollection][ev][vertexCollection + ".position.y"][0],
        frame[vertexCollection][ev][vertexCollection + ".position.z"][0],
    ]
    partTickleTrackLink_b = frame[recoParticleCollection][ev][recoParticleCollection + ".tracks_begin"]
    partTickleTrackLink_e = frame[recoParticleCollection][ev][recoParticleCollection + ".tracks_end"]
    partTickleTrackLink = []
    for ili, part_trkidx in enumerate(partTickleTrackLink_b):
        if part_trkidx == partTickleTrackLink_e[ili]:
            partTickleTrackLink.append(-1)  # no track / neutral
    else:
        partTickleTrackLink.append(part_trkidx)
    impacts = [-1, -1000, -1, -1000, -1000] * np.ones((len(partTickleTrackLink), 5))
    for ili, part_trkidx in enumerate(partTickleTrackLink):
        # each track exists 4 times, go to copy for trackstate at IP as interpolation works best here
        # i.e track 0 is present at idx 0-3 for different track states, 1 at 4-7, and so on -> multiply by 4
        si_trkidx = part_trkidx * 4
        if part_trkidx < 0:
            if debug >= 0:
                print("Found no SiTrack for particle! Maybe neutral?")
        elif si_trkidx < 0 or si_trkidx >= len(frame[trackCollection][ev][trackCollection + ".location"]):
            print("Warning, invalid track indx, please check!")
        else:
            if debug >= 0:
                print("Found SiTrack for particle!")
            pr = [
                frame[trackCollection][ev][trackCollection + ".referencePoint.x"][si_trkidx],
                frame[trackCollection][ev][trackCollection + ".referencePoint.y"][si_trkidx],
                frame[trackCollection][ev][trackCollection + ".referencePoint.z"][si_trkidx],
            ]
            d0 = frame[trackCollection][ev][trackCollection + ".D0"][si_trkidx]
            z0 = frame[trackCollection][ev][trackCollection + ".Z0"][si_trkidx]
            phi0 = frame[trackCollection][ev][trackCollection + ".phi"][si_trkidx]
            tanL = frame[trackCollection][ev][trackCollection + ".tanLambda"][si_trkidx]
            omega = frame[trackCollection][ev][trackCollection + ".omega"][si_trkidx]
            pca = calcPCA(pr, d0, z0, phi0, tanL, omega, vertex)
            dz = vertex[2] - pca[2]
            dxy = math.sqrt((vertex[0] - pca[0]) * (vertex[0] - pca[0]) + (vertex[1] - pca[1]) * (vertex[1] - pca[1]))
            d3 = math.sqrt(
                (vertex[0] - pca[0]) * (vertex[0] - pca[0])
                + (vertex[1] - pca[1]) * (vertex[1] - pca[1])
                + (vertex[2] - pca[2]) * (vertex[2] - pca[2])
            )
            impacts[ili] = np.array([dxy, dz, d3, d0, z0, pca])
    return impacts


def trimmed_track_info_z0_d0(
    frame,
    ev,
    recoParticleCollection="MergedRecoParticles",
    trackCollection="SiTracks_Refitted_1",
    debug=-1
):
    partTickleTrackLink_b = frame[recoParticleCollection][ev][recoParticleCollection + ".tracks_begin"]
    partTickleTrackLink_e = frame[recoParticleCollection][ev][recoParticleCollection + ".tracks_end"]
    partTickleTrackLink = []
    for ili, part_trkidx in enumerate(partTickleTrackLink_b):
        if part_trkidx == partTickleTrackLink_e[ili]:
            partTickleTrackLink.append(-1)  # no track / neutral
    else:
        partTickleTrackLink.append(part_trkidx)
    impacts = [-1000, -1000] * np.ones((len(partTickleTrackLink), 2))
    for ili, part_trkidx in enumerate(partTickleTrackLink):
        # each track exists 4 times, go to copy for trackstate at IP as interpolation works best here
        # i.e track 0 is present at idx 0-3 for different track states, 1 at 4-7, and so on -> multiply by 4
        si_trkidx = part_trkidx * 4
        if part_trkidx < 0:
            if debug >= 0:
                print("Found no SiTrack for particle! Maybe neutral?")
        elif si_trkidx < 0 or si_trkidx >= len(frame[trackCollection][ev][trackCollection + ".location"]):
            print("Warning, invalid track indx, please check!")
        else:
            if debug >= 0:
                print("Found SiTrack for particle!")
            d0 = frame[trackCollection][ev][trackCollection + ".D0"][si_trkidx]
            z0 = frame[trackCollection][ev][trackCollection + ".Z0"][si_trkidx]
            impacts[ili] = np.array([d0, z0])
    return impacts
