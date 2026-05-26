#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j

# Initial working directory:
#SBATCH -D ./

# Job name
#SBATCH -J test_gpu

{% extends "slurm.sh" %}

{% block tasks %}
{# Set defaults for resource variables with fallback values #}
{% set nodes = operations | map(attribute='directives.nodes') | default([1], true) | max %}
{% set tasks_per_node = operations | map(attribute='directives.tasks_per_node') | default([1], true) | max %}
{% set gpu = operations | map(attribute='directives.ngpu') | default([0], true) | max %}
{% set cpu_per_task = operations|map(attribute='directives.cpu_per_task')|default(1, true)|max %}

{% if nodes >= 1 %}
#SBATCH --nodes={{ nodes }}
{% endif %}

#SBATCH --ntasks-per-node={{ tasks_per_node }}
#SBATCH --cpus-per-task={{ cpu_per_task }}

{% if cpu_per_task ==24 %}
#SBATCH --ntasks-per-core=1
{% endif %}

{% if gpu > 0 %}
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:{{ gpu }}
{% endif %}
#SBATCH --time=24:00:00
#SBATCH --mem=11000MB

#SBATCH --mail-type=NONE
#SBATCH --mail-user=bhandarit@mpip-mainz.mpg.de
{% endblock tasks %}

{% block project_header %}
cd {{ project.path }}
module purge
module use /u/mgirard/modules/
module load gcc/14 openmpi/5.0
module load tbb/2022.1
module load cereal eigen
module load python-waterboa/2024.06
module load pybind11/2.13.6
module load rocm/6.3 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${WATERBOA_HOME}/lib
export PYTHONPATH=/u/bhandarit/hoomd:$PYTHONPATH
export PATH=/u/bhandarit/hoomd:$PATH
export PYTHONPATH=/u/bhandarit/hoomd:/u/mgirard/pythonPackages/hoobas:$PYTHONPATH
#export PYTHONPATH=/u/bhandarit/.local/lib/python3.12/site-packages:$PYTHONPATH


{% endblock project_header %}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

{% for operation in operations %}
{% endfor %}

