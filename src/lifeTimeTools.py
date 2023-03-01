"""
Helper functions tom get additional training inputs such as impact parameters
and calometric energy deposits. For the impact parameters a linear extrapolation
of the helix at the refference point is used. The definitions of track parameters
are taken from:
[1] https://flc.desy.de/lcnotes/notes/localfsExplorer_read?currentPath=/afs/desy.de/group/flc/lcnotes/LC-DET-2006-004.pdf
Math for finding the PCA from:
[2] https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

Author: Torben Lange (KBFI), Laurits Tani (KBFI)
Date: 22.02.2022
"""

import numpy as np
import math
import awkward as ak

"""
Helper calulating the PCA to the refference point, see [1] for mor details.
"""


def calcX0(xr, phi0, d0):
    alpha = 0.5 * math.pi - phi0
    return xr + math.cos(alpha) * d0


"""
Helper calulating the error of the PCA to the refference point, see [1] for mor details.
"""


def calcX0_error(xr, phi0, d0, phi0_error, d0_error):
    alpha = 0.5 * math.pi - phi0
    return math.sqrt(
        math.cos(alpha) * math.cos(alpha) * d0_error * d0_error
        + math.sin(alpha) * d0 * math.sin(alpha) * d0 * phi0_error * phi0_error
    )


"""
Helper calulating the PCA to the refference point, see [1] for mor details.
"""


def calcY0(yr, phi0, d0):
    alpha = 0.5 * math.pi - phi0
    return yr - math.sin(alpha) * d0


"""
Helper calulating the error of the PCA to the refference point, see [1] for mor details.
"""


def calcY0_error(yr, phi0, d0, phi0_error, d0_error):
    alpha = 0.5 * math.pi - phi0
    return math.sqrt(
        math.sin(alpha) * math.sin(alpha) * d0_error * d0_error
        + math.cos(alpha) * d0 * math.cos(alpha) * d0 * phi0_error * phi0_error
    )


"""
Helper calulating the Y component corresponding to a given X for a linear interpolation of the track
around the refference point.
"""


def calcY(xr, yr, phi0, d0, x):
    x0 = calcX0(xr, phi0, d0)
    y0 = calcY0(yr, phi0, d0)
    return math.tan(phi0) * (x - x0) + y0


"""
Helper calulating the error of the Y component corresponding to a given X for a linear interpolation of the track
around the refference point.
"""


def calcY_error(xr, yr, phi0, d0, x, phi0_error, d0_error):
    x0 = calcX0(xr, phi0, d0)
    x0_error = calcX0_error(xr, phi0, d0, phi0_error, d0_error)
    y0_error = calcY0_error(yr, phi0, d0, phi0_error, d0_error)
    return math.sqrt(
        y0_error * y0_error
        + math.tan(phi0) * math.tan(phi0) * x0_error * x0_error
        + (x - x0)
        / (math.cos(phi0) * math.cos(phi0))
        * (x - x0)
        / (math.cos(phi0) * math.cos(phi0))
        * phi0_error
        * phi0_error
    )


"""
Helper calculating the arc length of the track in the xy plane, needed to calculate the z component.
"""


def calcS(x, x0, y, y0, omega):
    return math.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)) * np.sign(omega)


"""
Helper calculating the error of the arc length of the track in the xy plane, needed to calculate the z component.
"""


def calcS_error(x, x0, y, y0, omega, x0_error, y_error, y0_error):
    return math.sqrt(
        ((x - x0) / calcS(x, x0, y, y0, omega)) * ((x - x0) / calcS(x, x0, y, y0, omega)) * x0_error * x0_error
        + ((y - y0) / calcS(x, x0, y, y0, omega))
        * ((y - y0) / calcS(x, x0, y, y0, omega))
        * (y_error * y_error + y0_error * y0_error)
    )


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
Helper calulating the error on the Z component corresponding to a given X and Y for a linear interpolation of the track
around the refference point. See Eq. 10 in [1].
"""


def calcZ_error(x, y, pr, d0, z0, phi0, tanL, omega, y_error, d0_error, z0_error, phi0_error, tanL_error):
    xr, yr, zr = pr
    x0 = calcX0(xr, phi0, d0)
    y0 = calcY0(yr, phi0, d0)
    s = calcS(x, x0, y, y0, omega)
    x0_error = calcX0_error(xr, phi0, d0, phi0_error, d0_error)
    y0_error = calcY0_error(yr, phi0, d0, phi0_error, d0_error)
    s_error = calcS_error(x, x0, y, y0, omega, x0_error, y_error, y0_error)
    return math.sqrt(s * s * tanL_error * tanL_error + tanL * tanL * s_error * s_error + z0_error * z0_error)


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
Helper, calculating the error on thefull coordinates of a point on the linear extrapolation of a track around its
refference point for a given X using the previous helper functions. As we parametrize in X, the error on X is 0.
"""


def calcP_error(x, pr, d0, z0, phi0, tanL, omega, d0_error, z0_error, phi0_error, tanL_error):
    xr, yr, zr = pr
    y = calcY(xr, yr, phi0, d0, x)
    y_error = calcY_error(xr, yr, phi0, d0, x, phi0_error, d0_error)
    z_error = calcZ_error(x, y, pr, d0, z0, phi0, tanL, omega, y_error, d0_error, z0_error, phi0_error, tanL_error)
    return [0, y_error, z_error]


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
Helper for the error on linear extrapolation of a track using an interpolation between two points
based on a parameter t. Used to find the PCA to the PV. See [2] for more details.
"""


def calcV_error(p1, p2, t, p1_error, p2_error, t_error):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x1_error, y1_error, z1_error = p1_error
    x2_error, y2_error, z2_error = p2_error
    xn_error = math.sqrt(
        x1_error * x1_error + t * t * (x1_error * x1_error + x2_error * x2_error) + t_error * t_error * (x2 - x1) * (x2 - x1)
    )
    yn_error = math.sqrt(
        y1_error * y1_error + t * t * (y1_error * y1_error + y2_error * y2_error) + t_error * t_error * (y2 - y1) * (y2 - y1)
    )
    zn_error = math.sqrt(
        z1_error * z1_error + t * t * (z1_error * z1_error + z2_error * z2_error) + t_error * t_error * (z2 - z1) * (z2 - z1)
    )
    return [xn_error, yn_error, zn_error]


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
Helper, calculating the error on the parameter value of t, for the PCA to the PV using a linear
extrapolation for two points on the assumed track parametrized in t (see calcV).
Formula taken from Eq. 3 in [2].
"""


def calcTPCA_error(p0, p1, p2, p0_error, p1_error, p2_error):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x0_error, y0_error, z0_error = p0_error
    x1_error, y1_error, z1_error = p1_error
    x2_error, y2_error, z2_error = p2_error
    # -x1*x1 +x1(x2+x0) -x0x2 + ...
    num = (x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)
    den = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1)
    num_error = math.sqrt(
        (x1 - x2) * (x1 - x2) * x0_error * x0_error
        + (-2 * x1 + x2 + x0) * (-2 * x1 + x2 + x0) * x1_error * x1_error
        + (x1 - x0) * (x1 - x0) * x2_error * x2_error
        + (y1 - y2) * (y1 - y2) * y0_error * y0_error
        + (-2 * y1 + y2 + y0) * (-2 * y1 + y2 + y0) * y1_error * y1_error
        + (y1 - y0) * (y1 - y0) * y2_error * y2_error
        + (z1 - z2) * (z1 - z2) * z0_error * z0_error
        + (-2 * z1 + z2 + z0) * (-2 * z1 + z2 + z0) * z1_error * z1_error
        + (z1 - z0) * (z1 - z0) * z2_error * z2_error
    )
    den_error = math.sqrt(
        (2 * x2 - 2 * x1) * (2 * x2 - 2 * x1) * (x1_error * x1_error + x2_error * x2_error)
        + (2 * y2 - 2 * y1) * (2 * y2 - 2 * y1) * (y1_error * y1_error + y2_error * y2_error)
        + (2 * z2 - 2 * z1) * (2 * z2 - 2 * z1) * (z1_error * z1_error + z2_error * z2_error)
    )
    return math.sqrt(num_error * num_error / (den * den) + num * num * den_error * den_error / (den * den * den * den))


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
Helper calculating the error on the PCA of the linear extrapolated track to the PV.
For more details see [2].
"""


def calcPCA_error(pr, d0, z0, phi0, tanL, omega, vertex, d0_error, z0_error, phi0_error, tanL_error):
    p1 = calcP(pr[0] + 0.01, pr, d0, z0, phi0, tanL, omega)
    p2 = calcP(pr[0] - 0.01, pr, d0, z0, phi0, tanL, omega)
    tPCA = calcTPCA(vertex, p1, p2)
    p1_error = calcP_error(pr[0] + 0.01, pr, d0, z0, phi0, tanL, omega, d0_error, z0_error, phi0_error, tanL_error)
    p2_error = calcP_error(pr[0] - 0.01, pr, d0, z0, phi0, tanL, omega, d0_error, z0_error, phi0_error, tanL_error)
    tPCA_error = calcTPCA_error(vertex, p1, p2, [0.0, 0.0, 0.0], p1_error, p2_error)
    return calcV_error(p1, p2, tPCA, p1_error, p2_error, tPCA_error)


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
    particles_ = frame[recoParticleCollection]
    particles = ak.Record({k.replace(f"{recoParticleCollection}.", ""): particles_[k] for k in particles_.fields})
    reco_particle_mask = particles["type"] != 0
    partTickleTrackLink_b = particles["tracks_begin"][reco_particle_mask][ev]
    partTickleTrackLink_e = particles["tracks_end"][reco_particle_mask][ev]
    partTickleTrackLink = []
    for ili, part_trkidx in enumerate(partTickleTrackLink_b):
        if part_trkidx == partTickleTrackLink_e[ili]:
            partTickleTrackLink.append(-1)  # no track / neutral
        else:
            partTickleTrackLink.append(part_trkidx)
    impacts_pcas_pvs = [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000] * np.ones(
        (len(partTickleTrackLink), 11)
    )
    impacts_pcas_errors = [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000] * np.ones((len(partTickleTrackLink), 8))
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
            # use omega to determine the track "direction", as omega is curvature correct for charge
            omega = (
                frame[trackCollection][ev][trackCollection + ".omega"][si_trkidx]
                * frame[recoParticleCollection][ev][recoParticleCollection + ".charge"][ili]
            )
            # declared with 21 entries but only has 15 as expected
            cov = frame[trackCollection][ev][trackCollection + ".covMatrix[21]"][si_trkidx]
            """ following:
                https://github.com/iLCSoft/ILDPerformance/blob/master/tracking/src/DDDiagnostics.cc#L777-L803
                We only use the sign of omega, error not needed
            """
            d0_error = cov[0]
            z0_error = cov[9]
            phi0_error = 0.0  # cov[2] approx 0
            tanL_error = 0.0  # approx 0. cov[14]
            pca = calcPCA(pr, d0, z0, phi0, tanL, omega, vertex)
            pca_error = calcPCA_error(pr, d0, z0, phi0, tanL, omega, vertex, d0_error, z0_error, phi0_error, tanL_error)
            # print(pca, pca_error)
            dz = vertex[2] - pca[2]
            dz_error = math.sqrt(pca_error[2] * pca_error[2])
            dxy = math.sqrt((vertex[0] - pca[0]) * (vertex[0] - pca[0]) + (vertex[1] - pca[1]) * (vertex[1] - pca[1]))
            dxy_error = math.sqrt(
                4 * (vertex[0] - pca[0]) * (vertex[0] - pca[0]) * pca_error[0] * pca_error[0]
                + 4 * (vertex[1] - pca[1]) * (vertex[1] - pca[1]) * pca_error[1] * pca_error[1]
            )
            d3 = math.sqrt(
                (vertex[0] - pca[0]) * (vertex[0] - pca[0])
                + (vertex[1] - pca[1]) * (vertex[1] - pca[1])
                + (vertex[2] - pca[2]) * (vertex[2] - pca[2])
            )
            d3_error = math.sqrt(
                4 * (vertex[0] - pca[0]) * (vertex[0] - pca[0]) * pca_error[0] * pca_error[0]
                + 4 * (vertex[1] - pca[1]) * (vertex[1] - pca[1]) * pca_error[1] * pca_error[1]
                + 4 * (vertex[2] - pca[2]) * (vertex[2] - pca[2]) * pca_error[2] * pca_error[2]
            )
            # store positions in mym as numbers are very small
            pca_store = [pca[i] * 1000000.0 for i in range(3)]
            pca_error_store = [pca_error[i] * 1000000.0 for i in range(3)]
            vertex_store = [vertex[i] * 1000.0 for i in range(3)]
            impacts_pcas_pvs[ili] = [dxy, dz, d3, d0, z0] + pca_store + vertex_store
            impacts_pcas_errors[ili] = [dxy_error, dz_error, d3_error, d0_error, z0_error] + pca_error_store
    return [impacts_pcas_pvs, impacts_pcas_errors]


"""
Helper to project the impact parameters of the track on the direction jet axis.
(Following BTV-11-002)
"""


def calculateImpactParameterSigns(ips, pca, pv, jetp4):
    # jet_direction = [jetp4["x"], jetp4["y"], jetp4["z"]]
    # jet_norm = math.sqrt(
    #     jet_direction[0] * jet_direction[0] + jet_direction[1] * jet_direction[1] + jet_direction[2] * jet_direction[2]
    # )
    # jet_direction = [(1.0 / jet_norm) * jet_direction[i] for i in range(len(jet_direction))]
    newips = []
    for iip, ip in enumerate(ips):
        if ip == -1000.0:
            newips.append(-1000.0)
            continue
        # pca_direction = [pca[0][iip] - pv[0][iip], pca[1][iip] - pv[1][iip], pca[2][iip] - pv[2][iip]]
        # pca_norm = math.sqrt(
        #     pca_direction[0] * pca_direction[0] + pca_direction[1] * pca_direction[1] + pca_direction[2] * pca_direction[2]
        # )
        # if pca_norm > 0:
        #     pca_direction = [(1.0 / pca_norm) * pca_direction[i] for i in range(len(pca_direction))]
        # else:
        #     pca_direction = jet_direction
        # sign = np.sign(
        #     pca_direction[0] * jet_direction[0] + pca_direction[1] * jet_direction[1] + pca_direction[2] * jet_direction[2]
        # )
        # newips.append(sign * abs(ip))
        newips.append(abs(ip))
    return newips
