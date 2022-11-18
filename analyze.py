import itertools
import bz2
import json
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import glob
import fastjet
import awkward as ak
import vector
import math
import awkward
from operator import mul

#Create the genparticle -> track/cluster -> particle flow event graph
#Graph means. 
def event_to_graph(df_gen, df_cl, df_tr, df_pfs):
    g = nx.DiGraph()
    
    #Add genparticles
    for igen in range(len(df_gen)):
        g.add_node("gen{}".format(igen), typ=int(df_gen[igen]["pdgid"]), e=df_gen[igen]["energy"])

    #Add links to parents
    for igen in range(len(df_gen)):
        
        #Add links to parents
        idx_parent0 = int(df_gen[igen]["idx_parent0"])
        if idx_parent0 != -1:
            g.add_edge("gen{}".format(idx_parent0), "gen{}".format(igen), w=0)
            
        idx_parent1 = int(df_gen[igen]["idx_parent1"])
        if idx_parent1 != -1:
            g.add_edge("gen{}".format(idx_parent1), "gen{}".format(igen), w=0)
            
    #Add calorimeter clusters
    for icl in range(len(df_cl)):
        g.add_node(
            "clu{}".format(icl),
            typ=df_cl[icl]["type"],
            e=df_cl[icl]["energy"]
        )
        
        #Add links from genparticles to cluster
        #The weight is the energy contribution from the genparticle
        for gp, gp_w in zip(df_cl[icl]["gp_contributions"]["0"], df_cl[icl]["gp_contributions"]["1"]):
            gp = int(gp)
            if gp_w/df_cl[icl]["energy"]>0.2:
                g.add_edge("gen{}".format(gp), "clu{}".format(icl), w=gp_w)

    #Add tracks
    for itr in range(len(df_tr)):
        g.add_node("tra{}".format(itr), typ=0, e=df_tr[itr]["pt"])
        
        #Add links from genparticles to track.
        #The weight is the number of hits in the track that came from this genparticle
        for gp, gp_w in zip(df_tr[itr]["gp_contributions"]["0"], df_tr[itr]["gp_contributions"]["1"]):
            gp = int(gp)
            if gp_w/df_tr[itr]["nhits"]>0.2:
                g.add_edge("gen{}".format(gp), "tra{}".format(itr), w=gp_w)

    #Add PF objects
    for ipf in range(len(df_pfs)):
        g.add_node(
            "pfo{}".format(ipf),
            typ=int(df_pfs[ipf]["type"]),
            e=df_pfs[ipf]["energy"]
        )
        
        #Add link from cluster to PF object if available
        cl_idx = int(df_pfs[ipf]["cluster_idx"])
        if cl_idx!=-1:
            g.add_edge("clu{}".format(cl_idx), "pfo{}".format(ipf), w=0)

        #Add link from track to PF object if available
        tr_idx = int(df_pfs[ipf]["track_idx"])
        if tr_idx!=-1:
            g.add_edge("tra{}".format(tr_idx), "pfo{}".format(ipf), w=0)
    return g

def compute_track_properties(df_tr):
    df_tr["pt"] = track_pt(df_tr["omega"])
    df_tr["px"] = np.cos(df_tr["phi"])*df_tr["pt"]
    df_tr["py"] = np.sin(df_tr["phi"])*df_tr["pt"]
    df_tr["pz"] = df_tr["tan_lambda"]*df_tr["pt"]
    
def process_one_event(data, iev):
    
    #Get the dataframes corresponding to this event
    df_gen = data[iev]["genparticles"]
    df_cl = data[iev]["clusters"]
    df_tr = data[iev]["tracks"]
    df_pfs = data[iev]["pfs"]
    compute_track_properties(df_tr)#Might be useful

    #Get the generator taus with status==2
    #PDG= number assigned to generator particles
    #Status=2 mean this is the last tau in the decay chain. 
    idx_taus = awkward.where((np.abs(df_gen["pdgid"])==15) & (df_gen["status"]==2))[0]
    
    #cluster the PF particles to jets, reorder by pt descending.
    cluster = fastjet.ClusterSequence(ak.Array({
        "px": df_pfs["px"],
        "py": df_pfs["py"],
        "pz": df_pfs["pz"],
        "E": df_pfs["energy"],
    }), jetdef)
    jets_constituents = cluster.constituent_index(min_pt=5)[::-1]
    #Usually anti k_t jets. 
    
    #Get the tau contributions in each PF object
    graph = event_to_graph(df_gen, df_cl, df_tr, df_pfs)
    df_pfs_taufrac = get_tau_fractions(idx_taus, df_pfs, graph)
    
    #Now get the list of PF objects in each jet
    pfs_by_jet = []
    jets = []
    for jet_constituents in jets_constituents:
        
        #Get the PF objects corresponding to this jet
        pfs_jet = df_pfs.iloc[jet_constituents]
        pfs_jet_additional = df_pfs_taufrac.iloc[jet_constituents]
        pfs_this_jet = pandas.concat([pfs_jet, pfs_jet_additional], axis=1)
        pfs_by_jet.append(pfs_this_jet)
        jet = computeJet(pfs_this_jet)
        jets.append(jet)

    #return the gen taus, the computed jet, the list of PF candidates in each jet, cluster, tracks and genparticles
    return df_tau, jets, pfs_by_jet, df_cl, df_tr, df_gen
#pfs_by_jet


data = ak.from_parquet("../testdata/pythia6_ttbar_0001_pandora.parquet")
