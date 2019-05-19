#!/bin/bash
#SBATCH --partition=next
#SBATCH --time=200:00:00
#SBATCH --mem=1G
#SBATCH --job-name="GLUP2O7"
#SBATCH --cpus-per-task=1
# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
# sample job
sh glucose7.sh
echo NPROCS=$NPROCS
echo "Done"

