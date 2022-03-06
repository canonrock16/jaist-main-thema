#!/bin/bash
#PBS -q SINGLE
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -j oe
#PBS -N reproduce
#PBS -M s2030415@jaist.ac.jp -m be

module load openmpi/4.1.1/gcc
module load cmake/3.20.2
cd $PBS_O_WORKDIR
# cd $HOME/dev/jaist-main-thema
# python reproduce_gachi.py
python reproduce_gachi_less_synapse.py
# python reproduce_gachi_less_synapse_no_standalone.py

