{% extends "base_script.sh" %}
{% block header %}
#!/bin/sh -l
#PBS -j oe
#PBS -l nodes=1:ppn=16
#PBS -l walltime=96:00:00
#PBS -q low
#PBS -V
#PBS -o /raid6/homes/firstcenter/imideil-2/output/

source activate new-signac
module load gromacs/2018.5

{% endblock %}
