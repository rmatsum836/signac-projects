#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging
import argparse

import signac


def main(args):
    project = signac.init_project('imideil')
    statepoints_init = []
    #for seed in range(args.num_replicas):
    n_IL = 200
    #cations = ['li', 'k']
    cations = ['li']
    anions = ['fsi']
    temperatures = [280, 303, 323, 345]
    concentrations = [3]
    forcefields = ['jc']
    solvent = 'ch3cn'
    for cation in cations:
        for anion in anions:
            for temp in temperatures:
                for conc in concentrations:
                    if solvent == 'none':
                        statepoint = dict(
                                    forcefield='jc',
                                    solvent=solvent,
                                    cation=cation,
                                    anion=anion,
                                    T=temp,
                                    concentration=0

                                    )
                        project.open_job(statepoint).init()
                        statepoints_init.append(statepoint)
                    else:
                        statepoint = dict(
                                   forcefield='jc',
                                   solvent=solvent,
                                   cation=cation,
                                   anion=anion,
                                   T=temp,
                                   concentration=conc
                                   )
                        project.open_job(statepoint).init()
                        statepoints_init.append(statepoint)

    # Writing statepoints to hash table as a backup
    project.write_statepoints(statepoints_init)


if __name__ == '__main__':
     parser = argparse.ArgumentParser(
         description="Initialize the data space.")
     parser.add_argument(
         '-n', '--num-replicas',
         type=int,
         default=1,
         help="Initialize multiple replications.")
     args = parser.parse_args()
  
     logging.basicConfig(level=logging.INFO)
     main(args)
