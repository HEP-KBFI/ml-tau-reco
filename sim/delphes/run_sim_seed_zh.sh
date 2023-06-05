#!/bin/bash
set +e

NUM=$1

XDIR="out/pythia8_zh_htautau"
mkdir -p $XDIR
OUTROOT="tev14_pythia8_zh_htautau_$NUM.root"
OUT="tev14_pythia8_zh_htautau_$NUM.promc"
LOG="logfile_$NUM.txt"

rm -f $XDIR/$OUTROOT $XDIR/$OUT

#source /opt/hepsim.sh
cp tev14_pythia8_zh_htautau.py tev14_pythia8_zh_htautau.py.${NUM}
echo "Random:seed=${NUM}" >> tev14_pythia8_.py.${NUM}
./main.exe tev14_pythia8_zh_htautau.py.${NUM} $XDIR/$OUT > $XDIR/$LOG 2>&1
delphes/DelphesProMC delphes_card_CMS_PileUp.tcl $XDIR/$OUTROOT $XDIR/$OUT >> $XDIR/$LOG 2>&1
