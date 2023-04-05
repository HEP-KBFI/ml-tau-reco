#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out
set -e
set -x

env
df -h

NEV=100
NUM=$1 #random seed
SAMPLE=$2 #main card

#Change these as needed
OUTDIR=/local/joosep/clic_edm4hep/
SIMDIR=/home/joosep/ml-tau-reco/sim
WORKDIR=/scratch/local/$USER/${SAMPLE}_${SLURM_JOB_ID}
FULLOUTDIR=${OUTDIR}/${SAMPLE}

mkdir -p $FULLOUTDIR

mkdir -p $WORKDIR
cd $WORKDIR

cp $SIMDIR/fcc/${SAMPLE}.cmd card.cmd
cp $SIMDIR/fcc/pythia.py ./
cp $SIMDIR/fcc/clic_steer.py ./
cp -R $SIMDIR/fcc/PandoraSettings ./
cp -R $SIMDIR/fcc/clicRec_e4h_input.py ./

echo "Random:seed=${NUM}" >> card.cmd
cat card.cmd

#Use a tagged version of Key4HEP
source /cvmfs/sw.hsf.org/spackages6/key4hep-stack/2023-01-15/x86_64-centos7-gcc11.2.0-opt/csapx/setup.sh

#Run generation
k4run $SIMDIR/fcc/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd

#Run simulation
ddsim --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml \
      --outputFile out_sim_edm4hep.root \
      --steeringFile clic_steer.py \
      --inputFiles out.hepmc \
      --numberOfEvents $NEV \
      --random.seed $NUM

#Run reconstruction
k4run clicRec_e4h_input.py -n $NEV --EventDataSvc.input out_sim_edm4hep.root --PodioOutput.filename out_reco_edm4hep.root

#Copy the outputs
cp out_reco_edm4hep.root reco_${SAMPLE}_${NUM}.root
cp reco_${SAMPLE}_${NUM}.root $FULLOUTDIR/

rm -Rf $WORKDIR
