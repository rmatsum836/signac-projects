# li_imide

This was a signac project to simulate LiFSI and LiTFSI ionic liquids solvated in water and acetonitrile.  The statepoints in this project are:
- anion type
- cation type
- solvent type
- forcefield
- temperature
- solvent concentration

## Requirements

- signac (version 1.2.0 or greater)
- signac-flow (version 0.7.1 or greater)

## Usage

To initialize a workflow such as this, run `python src/init.py` in the root directory.  This command will initalize the statepoints and their directories in `workspace`.  All the other commands of this workflow are contained in `src/project.py`.  To initialize and paramterize a system with [mBuild](https://github.com/mosdef-hub/mbuild) and [foyer](https://github.com/mosdef-hub/foyer) for example, the following command can be run with signac-flow: `python src/project.py -o initialize`.  See [signac-flow](https://docs.signac.io/projects/flow/en/latest/) documentation for more commands.
