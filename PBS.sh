#!/bin/bash
#PBS -q SINGLE
#PBS -l select=1:ncpus=128
#PBS -j oe
#PBS -N reproduce
#PBS -M s2030415@jaist.ac.jp -m be

module load openmpi/4.1.1/gcc
module load cmake/3.20.2
cd $PBS_O_WORKDIR

#poetry run python reproduce_gachi_less_synapse_1_0.py
# poetry run python reproduce_gachi_less_synapse_0_5.py
# poetry run python reproduce_gachi_less_synapse_0_0.py
poetry run python reproduce_gachi_less_synapse.py

