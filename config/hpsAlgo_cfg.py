{
    "HPSAlgo": {
        "signalCands": {"minChargedHadronPt": 0.5, "minElectronPt": 0.5, "minMuonPt": 1.0e6},
        "isolationCands": {
            "minChargedHadronPt": 0.5,
            "minElectronPt": 0.5,
            "minGammaPt": 1.0,
            "minMuonPt": 0.5,
            "minNeutralHadronPt": 10.0,
        },
        "matchingConeSize": 1.0e-1,
        "isolationConeSize": 5.0e-1,
        "StripAlgo": {
            "useGammas": true,
            "minGammaPtSeed": 1.0,
            "minGammaPtAdd": 1.0,
            "useElectrons": true,
            "minElectronPtSeed": 0.5,
            "minElectronPtAdd": 0.5,
            "minStripPt": 1.0,
            "updateStripAfterEachCand": false,
            "maxStripBuildIterations": -1,
            "maxStripSizeEta": 0.05,
            "maxStripSizePhi": 0.20,
        },
    }
}
