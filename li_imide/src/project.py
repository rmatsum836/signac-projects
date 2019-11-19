from flow import FlowProject
import signac
import flow
#import pairing
import pairing
import matplotlib.pyplot as plt
import mbuild as mb
import mdtraj as md
from mtools.pairing import chunks
from scipy import stats
import numpy as np
import pickle
from foyer import Forcefield
from scipy.optimize import curve_fit
from get_mol2 import GetSolv, GetIL
from util.decorators import job_chdir
from pkg_resources import resource_filename
from mtools.gromacs.gromacs import make_comtrj
from mtools.post_process import calc_msd
from ramtools.conductivity import calc_conductivity
from mtools.post_process import calc_density
from multiprocessing import Pool
from scipy.special import gamma
import os
import environment
import itertools as it
import gzip
import shutil
from simtk.unit import *
import MDAnalysis as mda
import MDAnalysis.analysis.hbonds
from collections import defaultdict


def _pairing_func(x, a, b):
    """Stretched exponential function for fitting pairing data"""
    y = np.exp(-1 * b * x ** a)
    return y

def workspace_command(cmd):
    """Simple command to always go to the workspace directory"""
    return ' && '.join([
        'cd {job.ws}',
        cmd if not isinstance(cmd, list) else ' && '.join(cmd),
        'cd ..',
    ])


def _run_overall(trj, mol):
     D, MSD, x_fit, y_fit = calc_msd(trj)
     return D, MSD

 
def _save_overall(job, mol, trj, MSD):
        np.savetxt(os.path.join(job.workspace(), 'msd-{}-overall.txt'.format    (mol)),
                        np.transpose(np.vstack([trj.time, MSD])),
                                header='# Time (ps)\tMSD (nm^2)')

        fig, ax = plt.subplots()
        ax.plot(trj.time, MSD)
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('MSD (nm^2)')
        fig.savefig(os.path.join(job.workspace(),
                    'msd-{}-overall.pdf'.format(mol)))


def _run_multiple(trj, mol):
    D_pop = list()
    for start_frame in np.linspace(0, 4999, num=200, dtype=np.int):
        end_frame = start_frame + 400
        if end_frame < 5000:
            chunk = trj[start_frame:end_frame]
            print('\t\t\t...frame {} to {}'.format(start_frame, end_frame))
            try:
                D_pop.append(calc_msd(chunk)[0])
            except TypeError:
                import pdb
                pdb.set_trace()
        else:
            continue
    D_bar = np.mean(D_pop)
    D_std = np.std(D_pop)
    return D_bar, D_std


init_file = 'top_2.mol2'
em_file = 'em.gro'
nvt_file = 'nvt.gro'
npt_file = 'npt.gro'
sample_file = 'sample.gro'
unwrapped_file = 'com.gro'
msd_file = 'msd-solvent-overall.txt'
pair_file = 'direct-matrices-solvent-anion.pkl.gz'
pair_fit_file = 'matrix-pairs-solvent-anion.txt'
tau_file = 'tau.txt'
rdf_file = 'rdf-solvent-solvent.txt'
all_directs_file = 'all-directs-solvent-peak.pkl.gz'
all_indirects_file = 'all-indirects-solvent-anion.pkl'
cn_file = 'cn-cation-anion-2.txt'
hbond_file = 'hbonds_anion.csv'
hbond_json = 'hbonds_bulk.json'

class Project(FlowProject):
    pass

@Project.label
def initialized(job):
    return job.isfile(init_file)

@Project.label
def minimized(job):
    return job.isfile(em_file)

@Project.label
def hbond_done(job):
    return job.isfile(hbond_file)

@Project.label
def nvt_equilibrated(job):
    return job.isfile(nvt_file)

@Project.label
def npt_equilibrated(job):
    return job.isfile(npt_file)

@Project.label
def sampled(job):
    return job.isfile(sample_file)

@Project.label
def prepared(job):
    return job.isfile(unwrapped_file)

@Project.label
def msd_done(job):
    return job.isfile(msd_file)

@Project.label
def pair_done(job):
    return job.isfile(pair_file)

@Project.label
def pair_fit_done(job):
    return job.isfile(pair_fit_file)

@Project.label
def directs_done(job):
    return job.isfile(all_directs_file)

@Project.label
def indirects_done(job):
    return job.isfile(all_indirects_file)

@Project.label
def tau_done(job):
    return job.isfile(tau_file)

@Project.label
def rdf_done(job):
    return job.isfile(rdf_file)

@Project.label
def json_hbond_done(job):
    return job.isfile(hbond_json)

#@Project.label
#def cn_done(job):
#    return job.isfile(cn_file)

@Project.operation
@Project.post.isfile(init_file)
def initialize(job):
    with job:
        print(job)
        print("Setting up packing ...")
        anion = GetIL(job.sp()['anion'])
        print(anion)
        cation = GetIL(job.sp()['cation'])
        print(cation)
        n_IL = 200
        if job.sp()['solvent'] == 'none':
            packing_box = mb.Box([3.3, 3.3, 3.3])
            system_box = mb.Box([3.5,3.5, 3.5])
            system = mb.fill_box(compound=[cation, anion],
                    n_compounds=[n_IL, n_IL],
                    box=packing_box)
        else:
            solvent = GetSolv(job.sp()['solvent'])
            concentration = job.sp()['concentration']
            if concentration in [1, 1.5]:
                if job.sp()['solvent'] == 'ch3cn':
                    packing_box = mb.Box([5, 5, 5])
                    system_box = mb.Box([4,4,4])
                else:
                    packing_box = mb.Box([3.8, 3.8, 3.8])
                    system_box = mb.Box([4,4,4])
            if concentration in [2,3]:
                if job.sp()['solvent'] == 'ch3cn':
                    packing_box = mb.Box([6, 6, 6])
                    system_box = mb.Box([6.5,6.5,6.5])
                else:
                    packing_box = mb.Box([4, 4, 4])
                    system_box = mb.Box([4.5,4.5,4.5])
            print(packing_box)
            n_solvent = int(round(n_IL * concentration))
            print(n_solvent)
            system = mb.fill_box(compound=[cation, anion, solvent],
                    n_compounds=[n_IL, n_IL, n_solvent],
                    box=packing_box)

        cation = mb.Compound()
        anion = mb.Compound()
        solv = mb.Compound()
        for child in system.children:
            if child.name == job.sp()['cation']:
                cation.add(mb.clone(child))
            elif child.name == job.sp()['anion']:
                anion.add(mb.clone(child))
            else:
                solv.add(mb.clone(child))

        Lopes = Forcefield(forcefield_files = os.path.join(
            resource_filename('ilforcefields', 'lopes'), 'lopes.xml'))
        jc =Forcefield(forcefield_files=os.path.join(
            job._project.root_directory(),
            'src/util/lib/jc_spce.xml'))
        dang =Forcefield(forcefield_files=os.path.join(
            job._project.root_directory(),
            'src/util/lib/dang.xml'))
        tip3p =Forcefield(forcefield_files=os.path.join(
            job._project.root_directory(),
            'src/util/lib/spce.xml'))
        oplsaa = Forcefield(name='oplsaa')
        print("Atomtyping ...")
        if job.sp()['solvent'] == 'spce':
            solventPM = tip3p.apply(solv, residues=[job.sp()['solvent']])
        elif job.sp()['solvent'] in ['pc', 'ch3cn']:
            solventPM = oplsaa.apply(solv, assert_dihedral_params=False,
                    assert_angle_params=False,residues=[job.sp()['solvent']])
        if job.sp()['forcefield'] == 'jc':
            cationPM = jc.apply(cation, residues=[job.sp()['cation']])
        elif job.sp()['forcefield'] == 'deng':
            cationPM = dang.apply(cation, residues=[job.sp()['cation']])
        anionPM = Lopes.apply(anion, residues=[job.sp()['anion']])

        scale = 0.71
        if scale != 1.0:
             print("Scaling charges ... ")
             for atom in cationPM.atoms:
                atom.charge *= scale
             for atom in anionPM.atoms:
                atom.charge *= scale

        if job.sp()['solvent'] == 'none':
            structure = cationPM + anionPM
        else:
            structure = cationPM + anionPM + solventPM

        print("Saving .gro, .pdb and .top ... ")
        #structure.save('init.gro', overwrite=True)
        #structure.save('top.pdb', overwrite=True)
        structure.save('top_2.mol2', overwrite=True)
        #structure.save('init.top', combine='all', overwrite=True)


@Project.operation
@Project.pre.isfile(init_file)
@Project.post.isfile(em_file)
@flow.cmd
def em(job):
    return _gromacs_str('em', 'init', 'init', job)


@Project.operation
@Project.pre.isfile(em_file)
@Project.post.isfile(nvt_file)
@flow.cmd
def nvt(job):
    return _gromacs_str('nvt', 'em', 'init', job)


@Project.operation
@Project.pre.isfile(nvt_file)
@Project.post.isfile(npt_file)
@flow.cmd
def npt(job):
    return _gromacs_str('npt', 'nvt', 'init', job)


@Project.operation
@Project.pre.isfile(npt_file)
@Project.post.isfile(sample_file)
@flow.cmd
def sample(job):
    return _gromacs_str('sample', 'npt', 'init', job)

@Project.operation
@Project.pre.isfile(sample_file)
@Project.post.isfile(unwrapped_file)
def prepare(job):
    #if job.get_id() == '41fd6198b7f5675f9ecd034ce7c5af73':
    #    pass
    #else:
    trr_file = os.path.join(job.workspace(), 'sample.trr')
    xtc_file = os.path.join(job.workspace(), 'sample.xtc')
    gro_file = os.path.join(job.workspace(), 'init.gro')
    tpr_file = os.path.join(job.workspace(), 'sample.tpr')
    if os.path.isfile(xtc_file) and os.path.isfile(gro_file):
        unwrapped_trj = os.path.join(job.workspace(),
        'sample_unwrapped.xtc')
        if not os.path.isfile(unwrapped_trj):
            os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -pbc nojump'.format(xtc_file, unwrapped_trj, tpr_file))
        res_trj = os.path.join(job.ws, 'sample_res.xtc')
        com_trj = os.path.join(job.ws, 'sample_com.xtc')
        unwrapped_com_trj = os.path.join(job.ws,'sample_com_unwrapped.xtc')
        if not os.path.isfile(res_trj):
            os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -pbc res'.format(
                xtc_file, res_trj, tpr_file))
        if os.path.isfile(res_trj) and not os.path.isfile(com_trj):
            trj = md.load(res_trj, top=gro_file)
            comtrj = make_comtrj(trj)
            comtrj.save_xtc(com_trj)
            comtrj[-1].save_gro(os.path.join(job.workspace(),
                'com.gro'))
            print('made comtrj ...')
        if os.path.isfile(com_trj) and not os.path.isfile(unwrapped_com_trj)    :
            os.system('gmx trjconv -f {0} -o {1} -pbc nojump'.format(
                com_trj, unwrapped_com_trj))


@Project.operation
@Project.pre.isfile(unwrapped_file)
@Project.post.isfile(msd_file)
def run_msd(job):
    #if job.get_id() in ['bfb1f9909fd72fa621037bcf9f397fad', '0b8b93eab7be9b1239449f4100aeda33']:
    #    pass
    #else:
    #if job.sp.cation == 'none':
    #    if all([val in job.document.keys() for val in [
    #        'D_all_overall',
    #        'D_all_bar',
    #        'D_all_std',
    #        'D_cation_overall',
    #        'D_cation_bar',
    #        'D_cation_std',
    #        'D_ion_overall',
    #        'D_ion_bar',
    #        'D_ion_std',
    #        'D_anion_overall',
    #        'D_anion_bar',
    #        'D_anion_std']]):
    #        print('In job{}, found everything!'.format(job))
    #        return
    #else:
    #    if all([val in job.document.keys() for val in [
    #        'D_all_overall',
    #        'D_all_bar',
    #         'D_all_std',
    #         'D_solvent_overall',
    #         'D_solvent_overall-2',
    #         'D_solvent_bar',
    #         'D_solvent_std',
    #         'D_cation_overall',
    #         'D_cation_bar',
    #         'D_cation_std',
    #         'D_ion_overall',
    #         'D_ion_bar',
    #         'D_ion_std',
    #         'D_anion_overall',
    #         'D_anion_bar',
    #         'D_anion_std']]):
    #         print('In job{}, found everything!'.format(job))
    #         return
 
    print('Loading trj {}'.format(job))
    top_file = os.path.join(job.workspace(), 'sample.gro')
    trj_file = os.path.join(job.workspace(),
            'sample_unwrapped.xtc')
    trj = md.load(trj_file, top=top_file)
    selections = {'all' : trj.top.select('all'),
                  'ion' : trj.top.select('resname name {0} {1}'.format(job.sp.cation    ,
                      job.sp.anion)),
                  'cation': trj.top.select("resname name '{}'".format(job.sp.cation)    ),
                  'anion': trj.top.select("resname '{}'".format(job.sp.anion)),
                  'solvent' : trj.top.select("not resname '{}' and not resname '{}'".format(job.sp.cation,
                      job.sp.anion))
                  }
 
    for mol, indices in selections.items():
        print('\tConsidering {}'.format(mol))
        if mol == 'solvent':
            if len(indices) == 0:
                continue
        #@if 'D_' + mol + '_overall-2' not in job.document:
        print(mol)
        sliced = trj.atom_slice(indices)
        D, MSD = _run_overall(sliced, mol)
        job.document['D_' + mol + '_overall-2'] = D
        _save_overall(job, mol, sliced, MSD)
        #else:
        #    print('Found D_' + mol + '_overall')
 
        #if ('D_' + mol + '_bar-2' not in job.document or
        #    'D_' + mol + '_bar-2' not in job.document):
        sliced = trj.atom_slice(indices)
        D_bar, D_std = _run_multiple(sliced, mol)
        job.document['D_' + mol + '_bar-2'] = D_bar
        job.document['D_' + mol + '_std-2'] = D_std
        #else:
        #    print('Found D_' + mol + '_bar \t and D_' + mol + '_std')

#@Project.operation
#@flow.cmd
#def remove_gro(job):
#    if job.get_id() in ['41fd6198b7f5675f9ecd034ce7c5af73',
#                      '924b520c82144d6c3093b2ef58291c1e',
#                      '5804cfcd8e301bbda4b3f2d148a8c744',
#                      'f18871d29f85dfbd1ed195f9c735afd0',
#                      '2022aeda4c212e96857358c4b09bae5a',
#                      '85a82ea6c6c74bb01ac9dc8632d28875']:
#       print("Not removing")
#    else:
#       print("Removing")
#       cmd = 'rm nvt.gro && rm nvt.cpt && rm npt.gro && rm npt.cpt && rm sample.gro && rm sample.cpt'
#
#       return workspace_command(cmd)

@Project.operation
@Project.pre.isfile(msd_file)
@Project.post.isfile(pair_file)
def run_pair(job):
    print('hey')
    combinations = [['solvent', 'cation'],
                    ['solvent', 'anion']]
    #combinations = [['solvent', 'cation'],
    #                ['solvent', 'anion'],
    #                ['cation', 'cation'],
    #                ['cation', 'anion'],
    #                ['anion', 'anion']] # ['ion','ion']]
    for combo in combinations:
        if os.path.exists(os.path.join(job.workspace(),'direct-matrices-{}-{}.pkl.gz'.format(combo[0],combo[1]))):
            continue
        else:
            print('Loading trj {}'.format(job))
            if job.get_id() == '1ad289cbe7a639f71461aa6038f16f94':
                trj_file = os.path.join(job.workspace(), 'sample.xtc')
            else:
                trj_file = os.path.join(job.workspace(), 'sample.xtc')
            top_file = os.path.join(job.workspace(), 'init.gro')
            trj = md.load(trj_file, top=top_file)
            #trj = trj[30000:]
            anion = job.statepoint()['anion']
            cation = job.statepoint()['cation']
            solvent = job.statepoint()['solvent']
            if combo == ['solvent', 'solvent']:
                sliced = trj.topology.select('not resname {} {}'.format(cation,anion))
                if job.sp['solvent'] == 'ch3cn':
                    distance = 0.68
                else:
                    distance = 0.48
            elif combo == ['cation', 'cation']:
                sliced = trj.topology.select('resname {} {}'.format(cation,cation))
                distance = 0.43
            elif combo == ['anion', 'anion']:
                sliced = trj.topology.select('resname {} {}'.format(anion, anion))
                if job.sp['anion'] == 'tf2n':
                    distance = 1.25
                else:
                    distance = 0.8
            elif combo == ['cation', 'anion']:
                sliced = trj.topology.select('resname {} {}'.format(cation, anion))
                if job.sp['anion'] == 'tf2n':
                    #distance = 0.55
                    distance = {'li-li': 0.48, 'tf2n-tf2n': 1.25, 'li-tf2n': 0.55, 'tf2n-li':0.55}
                else:
                    #distance = 0.5
                    distance = {'li-li': 0.48, 'fsi-fsi': 0.8, 'li-fsi': 0.5, 'fsi-li':0.5}
            elif combo == ['solvent', 'anion']:
                sliced = trj.topology.select('not resname {}'.format(cation))
                if job.sp['anion'] == 'tf2n':
                    if job.sp['solvent'] == 'ch3cn':
                        distance = {'tf2n-tf2n': 1.25, 'tf2n-ch3cn':0.76,
                               'ch3cn-tf2n':0.76, 'ch3cn-ch3cn':0.68}
                    else:
                        distance = {'tf2n-tf2n': 1.25, 'tf2n-RES':0.76,
                               'RES-tf2n':0.76, 'RES-RES':0.45}
                else:
                    if job.sp['solvent'] == 'ch3cn':
                        distance = {'fsi-fsi': 0.8, 'fsi-ch3cn':0.63,
                               'ch3cn-fsi':0.63, 'ch3cn-ch3cn':0.68}
                    else:
                        distance = {'fsi-fsi': 0.8, 'fsi-RES':0.63,
                               'RES-fsi':0.63, 'RES-RES':0.45}
            elif combo == ['solvent', 'cation']:
                sliced = trj.topology.select('not resname {}'.format(anion))
                if job.sp['solvent'] == 'ch3cn':
                    distance = {'li-li': 0.48, 'li-ch3cn':0.3,
                               'ch3cn-li':0.3, 'ch3cn-ch3cn':0.68}
                else:
                    distance = {'li-li': 0.48, 'li-RES':0.28,
                               'RES-li':0.28, 'RES-RES':0.45}
                
            trj_slice = trj.atom_slice(sliced)
            trj_slice = trj_slice[:-1]
            direct_results = []
            print('Analyzing trj {}'.format(job))
            pair = 1001
            if pair > 1000:
                chunk_size = 100
                for chunk in chunks(range(trj_slice.n_frames),chunk_size): #500
                    trj_chunk = trj_slice[chunk]
                    first = make_comtrj(trj_chunk[0])
                    first_direct = pairing.generate_direct_correlation(
                                    first, cutoff=distance)

                    # Math to figure out frame assignments for processors
                    proc_frames = (len(chunk)-1) / 16
                    remain = (trj_chunk.n_frames-1) % 16
                    index = (trj_chunk.n_frames-1) // 16
                    starts = np.empty(16)
                    ends = np.empty(16)
                    i = 1
                    j = index+1
                    for x in range(16):
                        starts[x] = i
                        if x < remain:
                            j += 1
                            i += 1
                        ends[x] = j
                        i += index
                        j += index
                    starts = [int(start) for start in starts]
                    ends = [int(end) for end in ends]
                    params = [trj_chunk[i:j] for i,j in zip(starts,ends)]

                    print('Checking direct')
                    with Pool() as pool:
                        directs = pool.starmap(pairing.check_pairs, zip(params,
                                it.repeat(distance), it.repeat(first_direct)))
                    directs[0].insert(0, first_direct)
                    directs = np.asarray(directs)
                    direct_results.append(directs)

                print("saving now")

                with open(os.path.join(job.workspace(),'direct-matrices-{}-{}.pkl'.format(
                  combo[0],combo[1])), 'wb') as f:
                  pickle.dump(direct_results, f)

                with open(os.path.join(job.workspace(), 'direct-matrices-{}-{}.pkl'.format(
                  combo[0],combo[1])), 'rb') as f_in, gzip.open(os.path.join(job.workspace(),
                  'direct-matrices-{}-{}.pkl.gz'.format(combo[0],combo[1])), 'wb') as f_out:
                  shutil.copyfileobj(f_in, f_out)
                print("saved")

@Project.operation
@Project.pre.isfile(pair_file)
@Project.post.isfile(pair_fit_file)
def run_pairing_fit_matrix(job):
    print(job.get_id())
    combinations = [['solvent', 'anion'],
                    ['solvent', 'cation']]
    #combinations = [['solvent', 'cation'],
    #                ['solvent', 'anion'],
    #                ['cation', 'cation'],
    #                ['cation', 'anion']] # ['ion','ion']]
    for combo in combinations:
        print(combo)
        direct_results = []
        if os.path.exists(os.path.join(job.workspace(),'direct-matrices-{}-{}.pkl.gz'.format(combo[0],combo[1]))):
            with gzip.open(os.path.join(job.workspace(),'direct-matrices-{}-{}.pkl.gz'.format(combo[0],combo[1])), 'rb') as f:
                    direct_results = pickle.load(f)
            frames = 4999 
            #if job.document()['tau_pair_matrix'] > 1000:
            chunk_size = 100
            overall_pairs = []
            for chunk in direct_results:
                for proc in chunk:
                    for matrix in proc:
                        pairs = []
                        for row in matrix:
                            N = len(row)
                            count = len(np.where(row == 1)[0])
                            pairs.append(count)
                        pairs = np.sum(pairs)
                        pairs = (pairs - N) / 2
                        overall_pairs.append(pairs)
            ratio_list = []
            for i, pair in enumerate(overall_pairs):
                if i % chunk_size == 0:
                    divisor = pair
                    ratio_list.append(1)
                else:
                    if pair == 0:
                        ratio_list.append(0)
                    else:
                        pair_ratio = pair/ divisor
                        ratio_list.append(pair_ratio)
            new_ratio = []
            i = 0
            for j in range(chunk_size, frames, chunk_size):
                x = ratio_list[i:j]
                new_ratio.append(x)
                i = j
            mean = np.mean(new_ratio, axis=0)
            time_interval = [(frame * 1) for frame in range(chunk_size)]
            time_interval = np.asarray(time_interval)
            popt, pcov = curve_fit(_pairing_func,time_interval, mean)
            fit = _pairing_func(time_interval,*popt)

            np.savetxt(os.path.join(job.workspace(),
                 'matrix-pairs-{}-{}.txt'.format(combo[0],combo[1])),
                 np.column_stack((mean, time_interval, fit)),
                 header = 'y = np.exp(-1 * b * x ** a) \n' +
                     str(popt[0]) + ' ' + str(popt[1]))

            job.document['pairing_fit_a_matrix_{}_{}'.format(combo[0],combo[1])] = popt[0]
            job.document['pairing_fit_b_matrix_{}_{}'.format(combo[0],combo[1])] = popt[1]

@Project.operation
#@Project.pre.isfile(pair_fit_file)
def run_tau(job):
    combinations = [['solvent', 'cation'],
                    ['solvent', 'anion'],
                    ['cation', 'cation'],
                    ['cation', 'anion']] # ['ion','ion']]
    for combo in combinations:
        if 'pairing_fit_a_matrix_{}_{}'.format(combo[0],combo[1]) in job.document:
            a = job.document['pairing_fit_a_matrix_{}_{}'.format(combo[0],combo[1])]
            b = job.document['pairing_fit_b_matrix_{}_{}'.format(combo[0],combo[1])]
            tau_pair = gamma(1 / a) * np.power(b,(-1 / a)) / a

            with open(os.path.join(job.workspace(), 'tau_{}_{}.txt'.format(combo[0],combo[1])), 'w') as f:
                f.write(str(tau_pair))
            print('saving')

            job.document['tau_pair_matrix_{}_{}'.format(combo[0],combo[1])] = tau_pair


@Project.operation
@Project.pre.isfile(msd_file)
def run_rdf(job):
    print('Loading trj {}'.format(job))
    if os.path.exists(os.path.join(job.workspace(), 'com.gro')):
        top_file = os.path.join(job.workspace(), 'com.gro')
        trj_file = os.path.join(job.workspace(), 'sample_com.xtc')
        trj = md.load(trj_file, top=top_file, stride=10)

        selections = dict()
        selections['cation'] = trj.topology.select('name {}'.format(job.statepoint()['cation']))
        selections['anion'] = trj.topology.select('name {}'.format(job.statepoint()['anion']))
        selections['ion'] = trj.topology.select('name {0} {1}'.format(job.statepoint()['cation'],
                                                                                job.statepoint()['anion']))
        selections['solvent'] = trj.topology.select('not name {0} {1}'.format(job.statepoint()['cation'],
                                                                              job.statepoint()['anion']))
        selections['all'] = trj.topology.select('all')

        combos = [('ion', 'ion'),
                  ('cation', 'anion'),
                  ('cation','cation'),
                  ('anion','anion'),
                  ('solvent','anion'),
                  ('solvent','cation'),
                  ('solvent', 'solvent')]
        for combo in combos:

            print('running rdf between {0} ({1}) and\t{2} ({3})\t...'.format(combo[0],
                                                                             len(selections[combo[0]]),
                                                                             combo[1],
                                                                             len(selections[combo[1]])))
            r, g_r = md.compute_rdf(trj, pairs=trj.topology.select_pairs(selections[combo[0]], selections[combo[1]]), r_range=((0.0, 2.0)))

            data = np.vstack([r, g_r])
            np.savetxt(os.path.join(job.workspace(),
                    'rdf-{}-{}.txt'.format(combo[0], combo[1])),
                np.transpose(np.vstack([r, g_r])),
                header='# r (nm)\tg(r)')
            print(' ... done\n')


@Project.operation
@Project.pre.isfile(msd_file)
def run_cond(job):
    if 'D_cation_bar-2' in job.document().keys():
        top_file = os.path.join(job.workspace(), 'sample.gro')
        trj_file = os.path.join(job.workspace(),
                'sample_unwrapped.xtc')
        trj = md.load(trj_file, top=top_file)
        cation = trj.topology.select('resname {}'.format(
                 job.statepoint()['cation']))
        cation_msd = job.document()['D_cation_bar-2']
        anion_msd = job.document()['D_anion_bar-2']
        volume = float(np.mean(trj.unitcell_volumes))
        N = len(cation)
        T = job.sp['T']

        conductivity = calc_conductivity(N, volume, cation_msd, anion_msd, T=T)
        print(conductivity)
        job.document['ne-conductivity_2'] = conductivity
        print('Conductivity calculated')

@Project.operation
@Project.pre.isfile(msd_file)
def run_eh_cond(job):
    print(job.get_id())
    top_file = os.path.join(job.workspace(), 'com.gro')
    trj_file = os.path.join(job.workspace(), 'sample_com_unwrapped.xtc')
    trj_frame = md.load_frame(trj_file, top=top_file, index=0)

    trj_ion = trj_frame.atom_slice(trj_frame.top.select('resname li {}'.format(
        job.statepoint()['anion'])))
    charges = get_charges(trj_ion, job.statepoint()['anion'])
    new_charges = list()
    for charge in charges:
        if charge != 1:
            if charge > 0:
                charge = 1
            else:
                charge = -1
            new_charges.append(charge)
 
    chunks = np.arange(200,250,1)
    #trj = md.load(trj_file, top=top_file)
    #trj = trj.atom_slice(trj.top.select('resname li {}'.format(
    #      job.statepoint()['anion'])))
    #slope_list = list()
    #for i,start_frame in enumerate(np.linspace(0, 4999, num=200, dtype=np.int)):
    #    end_frame = start_frame + 200
    #    if end_frame < 5000:
    #        chunk = trj[start_frame:end_frame]
    #        trj_time = chunk.time
    #        M = dipole_moments_md(chunk, new_charges)
    #        running_avg += [np.linalg.norm((M[i] - M[0]))**2 for i in range(len(M))]
    #    else:
    #        continue

    for chunk in chunks:
        running_avg = np.zeros(chunk)
        for i,trj in enumerate(md.iterload(trj_file, top=top_file, chunk=chunk, skip=100)):
            if i == 0:
                trj_time = trj.time
            if trj.n_frames != chunk:
                continue
            trj = trj.atom_slice(trj.top.select('resname li {}'.format(
                  job.statepoint()['anion'])))
            M = dipole_moments_md(trj, new_charges)
            running_avg += [np.linalg.norm((M[i] - M[0]))**2 for i in range(len(M))]
    
        x = (trj_time - trj_time[0]).reshape(-1)
        y = running_avg / i

    slope, intercept, r_value, p_value, std_error = stats.linregress(
            x, y)

    kB = 1.38e-23 * joule / kelvin
    V = np.mean(trj_frame.unitcell_volumes, axis=0) * nanometer ** 3
    T = job.statepoint()['T'] * kelvin
    
    sigma = slope * (elementary_charge * nanometer) ** 2 / picosecond / (6 * V * kB * T)
    seimens = seconds ** 3 * ampere ** 2 / (kilogram * meter ** 2)
    sigma = sigma.in_units_of(seimens / meter)
    print(sigma)
    print(job.document()['ne-conductivity_2'])

    job.document['eh-conductivity-2'] = sigma / sigma.unit

@Project.operation
@Project.pre.isfile(msd_file)
@Project.post.isfile(all_directs_file)
def run_directs(job):
    if job.get_id() in ['1ad289cbe7a639f71461aa6038f16f94','509a76782f2eda70bfe5c3619485b689']:
        trj_file = os.path.join(job.workspace(), 'sample.xtc')
    else:
            trj_file = os.path.join(job.workspace(), 'sample.xtc')
    top_file = os.path.join(job.workspace(), 'init.gro')
    trj = md.load(trj_file, top=top_file)
    combinations = [['solvent','solvent']]
    #               ['cation','anion']]
    #                ['anion', 'anion'],
    #                ['cation', 'cation'],
    #                ['solvent', 'solvent']] # ['ion','ion']]
    for combo in combinations:
        print('Loading trj {}'.format(job))
        anion = job.statepoint()['anion']
        cation = job.statepoint()['cation']
        if combo == ['solvent', 'solvent']:
            sliced = trj.topology.select('not resname {} {}'.format(cation,anion))
            if job.sp['solvent'] == 'ch3cn':
                #distance = 0.68
                distance = 0.49
            else:
                #distance = 0.48 THIS IS FOR FIRST WELL
                distance = 0.28
        elif combo == ['cation', 'cation']:
            sliced = trj.topology.select('resname {} {}'.format(cation,cation))
            distance = 0.43
        elif combo == ['anion', 'anion']:
            sliced = trj.topology.select('resname {} {}'.format(anion, anion))
            if job.sp['anion'] == 'tf2n':
                distance = 1.25
            else:
                distance = 0.8
        elif combo == ['cation', 'anion']:
            sliced = trj.topology.select('resname {} {}'.format(cation, anion))
            if job.sp['anion'] == 'tf2n':
                #distance = 0.55
                distance = {'li-li': 0.48, 'tf2n-tf2n': 1.25, 'li-tf2n': 0.55, 'tf2n-li':0.55}
            else:
                #distance = 0.5
                distance = {'li-li': 0.48, 'fsi-fsi': 0.8, 'li-fsi': 0.5, 'fsi-li':0.5}
        elif combo == ['solvent', 'cation']:
            sliced = trj.topology.select('not resname {}'.format(anion))
            if job.sp['solvent'] == 'ch3cn':
                distance = {'li-li': 0.48, 'li-ch3cn':0.3,
                           'ch3cn-li':0.3, 'ch3cn-ch3cn':0.68}
            else:
                distance = {'li-li': 0.48, 'li-RES':0.28,
                           'RES-li':0.28, 'RES-RES':0.45}
        elif combo == ['solvent', 'anion']:
            sliced = trj.topology.select('not resname {}'.format(cation))
            if job.sp['anion'] == 'tf2n':
                if job.sp['solvent'] == 'ch3cn':
                    distance = {'tf2n-tf2n': 1.25, 'tf2n-ch3cn':0.76,
                           'ch3cn-tf2n':0.76, 'ch3cn-ch3cn':0.68}
                else:
                    distance = {'tf2n-tf2n': 1.25, 'tf2n-RES':0.76,
                           'RES-tf2n':0.76, 'RES-RES':0.45}
            else:
                if job.sp['solvent'] == 'ch3cn':
                    distance = {'fsi-fsi': 0.8, 'fsi-ch3cn':0.63,
                           'ch3cn-fsi':0.63, 'ch3cn-ch3cn':0.68}
                else:
                    distance = {'fsi-fsi': 0.8, 'fsi-RES':0.63,
                           'RES-fsi':0.63, 'RES-RES':0.45}
        #sliced = trj.topology.select('resname ch3cn')
        trj_slice = trj.atom_slice(sliced)
        trj_slice = trj_slice[:-1]
        index = trj_slice.n_frames / 16
        starts = np.empty(16)
        ends = np.empty(16)
        i = 0
        j = index
        for x in range(16):
            starts[x] = i
            ends[x] = j
            i += index
            j += index
        starts = [int(start) for start in starts]
        ends = [int(end) for end in ends]
        params = [trj_slice[i:j:10] for i,j in zip(starts,ends)]
        results = [] 

        with Pool() as pool:
            directs = pool.starmap(pairing.mult_frames_direct, zip(params, it.repeat(distance)))
        directs = np.asarray(directs)
 
        #with open(os.path.join(job.workspace(),'all-directs-{}-{}.pkl'.format(
        #     combo[0],combo[1])), 'wb') as f:
        #    pickle.dump(directs, f)
        with open(os.path.join(job.workspace(),'all-directs-solvent-peak.pkl'), 'wb') as f:
            pickle.dump(directs, f)
 
        #with open(os.path.join(job.workspace(), 'all-directs-{}-{}.pkl'.format(
        #     combo[0],combo[1])), 'rb') as f_in, gzip.open(os.path.join(job.workspace(),
        #     'all-directs-{}-{}.pkl.gz'.format(combo[0],combo[1])), 'wb') as f_out:
        #    shutil.copyfileobj(f_in, f_out)
        with open(os.path.join(job.workspace(), 'all-directs-solvent-peak.pkl')) as f_in, gzip.open(os.path.join(job.workspace(), 'all-directs-solvent-peak.pkl.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


@Project.operation
@Project.pre.isfile(all_directs_file)
#@Project.post.isfile(all_indirects_file)
def run_indirects(job):
    combinations = [['solvent','solvent']]
    #combinations = [['cation','cation'],
    #                ['anion', 'anion'],
    #                ['cation', 'anion'],
    #                ['solvent', 'solvent']] # ['ion','ion']]
    print(job.get_id())
    for combo in combinations:
        #with gzip.open(os.path.join(job.workspace(), 'all-directs-{}-{}.pkl.gz'.format(combo[0],combo[1])), 'rb') as f:
        #with gzip.open(os.path.join(job.workspace(), 'all-directs-solvent-peak.pkl.gz'), 'rb') as f:
        with open(os.path.join(job.workspace(), 'all-directs-solvent-peak.pkl'), 'rb') as f:
            direct = pickle.load(f)

        with Pool() as pool:
            indirects = pool.map(pairing.calc_indirect, direct)
            reducs = pool.map(pairing.calc_reduc, indirects)

        #with open(os.path.join(job.workspace(), 'all-indirects-{}-{}.pkl'.format(combo[0],combo[1])), 'wb') as f:
        with open(os.path.join(job.workspace(), 'all-indirects-solvent-peak.pkl'), 'wb') as f:
            pickle.dump(indirects, f)

        with open(os.path.join(job.workspace(), 'all-indirects-solvent-peak.pkl'), 'rb') as f_in, gzip.open(os.path.join(job.workspace(),'all-indirects-solvent-peak.pkl.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        #with open(os.path.join(job.workspace(), 'all-reducs-{}-{}.pkl'.format(combo[0],combo[1])), 'wb') as f:
        with open(os.path.join(job.workspace(), 'all-reducs-solvent-peak.pkl'), 'wb') as f:
            pickle.dump(reducs, f)


@Project.operation
@Project.pre.isfile(rdf_file)
def run_rho(job):
    print('Loading trj {}'.format(job))
    top_file = os.path.join(job.workspace(), 'sample.gro')
    trj_file = os.path.join(job.workspace(), 'sample.xtc')
    trj = md.load(trj_file, top=top_file)

    # Compute density in kg * m ^ -3
    rho = calc_density(trj)

    uob.document['rho'] = float(np.mean(rho))

    # Compute and store volume in nm ^ -3
    job.document['volume'] = float(np.mean(trj.unitcell_volumes))


@Project.operation
@Project.pre.isfile(rdf_file)
@Project.post.isfile(cn_file)
def run_cn(job):
    combinations = [['solvent','solvent'],
                    ['cation','cation'],
                    ['anion','anion'],
                    ['solvent','cation'],
                    ['cation', 'anion'],
                    ['solvent','anion']]
    for combo in combinations:
        r, g_r = np.loadtxt(os.path.join(job.workspace(),
              'rdf-{}-{}.txt'.format(
               combo[0],combo[1]))).T
        #if combo == ['anion', 'anion']:
        if 'anion' in combo:
            if 'cation' in combo:
                if job.sp['anion'] == 'fsi':
                    chunk = np.where((r>0.3) & (r<0.8))
                else:
                    chunk = np.where((r>0.45) & (r<0.8))
            else:
                if job.sp['anion'] == 'fsi':
                    chunk = np.where((r>0.5) & (r<0.85))
                else:
                    chunk = np.where((r>0.75) & (r<1.3))
        elif combo == ['solvent', 'solvent']:
            if job.sp['solvent'] == 'spce':
                chunk = np.where((r>0.3) & (r<0.55))
            else:
                chunk = np.where((r>0.4) & (r<0.8))
        else:
            chunk = np.where((r>0.3) & (r<0.8))
        g_r_chunk = g_r[chunk]
        r_chunk = r[chunk]
        if combo == ['solvent', 'solvent']:
             rho = (200*job.sp['concentration']) / job.document['volume']
        elif combo == ['cation', 'cation'] or combo == ['anion', 'anion']:
             rho = (200) / job.document['volume']
        elif combo == ['solvent', 'cation'] or combo == ['solvent', 'anion']:
             rho = ((200*job.sp['concentration'])+200) / job.document['volume']
        elif combo == ['cation', 'anion']:
             rho = (400 / job.document['volume'])

        N = [np.trapz(4 * rho * np.pi * g_r[:i] * r[:i] **2, r[:i], r) for i in range(len(r))]

        # Store CN near r = 0.8
        index = np.where(g_r == np.amin(g_r_chunk))
        print('combo is {}'.format(combo))
        print('g_r is {}'.format(g_r[index]))
        print('r is {}'.format(r[index]))
        #index = r[maxr]
        #if combo == 'solvent-solvent':
        #    if job.sp['solvent'] == 'spce':
        #        index = np.argwhere(r > 0.31)[0]
        #    else:
        #        index = np.argwhere(r > 0.5)[0]
        #else:
        #    if job.sp['anion'] == 'fsi':
        #        index = np.argwhere(r > 0.4)[0]
        #    else:
        #        index = np.argwhere(r > 0.5)[0]

        #if combo == 'solvent-solvent':
        #    job.document['cn_solvent_solvent'] = N[int(index)]
        #elif combo == 'cation-anion':
        #    job.document['cn_cation_anion'] = N[int(index)]
        job.document['cn_{}_{}'.format(combo[0], combo[1])] = N[int(index[0])]
        
        # Save entire CN
        np.savetxt(os.path.join(job.workspace(),
                  'cn-{}-{}.txt'.format(combo[0],combo[1])),
                  np.transpose(np.vstack([r, N])),
                  header='# r (nm)\tCN(r)')


@Project.operation
def calc_caging(job):
    print(job.get_id())
    #types = 'cation-anion'
    #types = 'solvent-cation'
    types = 'solvent-solvent'
    #with gzip.open(os.path.join(job.workspace(), 'all-directs-{}.pkl.gz'.format(types)), 'rb') as f:
    with open(os.path.join(job.workspace(), 'all-directs-solvent-peak.pkl'), 'rb') as f:
        directs = pickle.load(f)

    # Get Matrices into a better array
    #if job.get_id() not in ['absc']:#['6b35a854a0fff3e9b990786a46262d65', '3cc98f4d7ffc7b021d85f74e96009381']:
    if job.get_id() in ['343ce2738d272b1bbb1536ea4f77c0ed', '876059e82269b28113ef9dc3babff2e7']:
        matrix_list = list()
        chunk_size = 500
        for chunk in directs:
        #    for proc in chunk:
        #        for matrix in proc:
        #            matrix_list.append(matrix)
            for matrix in chunk:
                matrix_list.append(matrix)
        matrix_list = np.asarray(matrix_list)
        
        overall_cages = dict()
        set_dict = dict()
        for chunk in chunks(range(len(matrix_list)),chunk_size):
            matrix_chunk = matrix_list[chunk]
            for i in range(len(chunk)):
                for index, row in enumerate(matrix_chunk[i]):
                    count = len(np.where(row==1)[0])
                    count -= 1 # Subtract itself to get neighbors
                    count_set = tuple(set(list(np.where(row==1)[0]))) # Check the pairs, we'll check later if this pair has already been counted for
                    if len(count_set) != 0: # Artificially setting this to 0 so that we don't flter out any of the counts
                        if count in overall_cages.keys():
                            overall_cages[count] += 1
                        else:
                            overall_cages[count] = 1
                    else:
                        if count_set in set_dict:
                            pass
                        else:
                            set_dict[count_set] = 0
                            if count in overall_cages.keys():
                                overall_cages[count] += 1
                            else:
                                overall_cages[count] = 1

        
        #sums = 0 # Sanity check
        #sums_original = 0
        #for k, v in overall_cages.items():
        #    overall_cages[k] = v / (k+1)
        #    sums += (v/(k+1))
        #    sums_original += v
        neighbors = list()
        count = list()
        for k,v in overall_cages.items():
            neighbors.append(k)
            count.append(v)
        import pdb; pdb.set_trace()
        #with open(os.path.join(job.workspace(), 'neigh_solv_solv.txt'), 'w') as f:
        #    #f.write(str(overall_cages))
        #    f.write([neighbors, count])
        #np.savetxt(os.path.join(job.workspace(), 'neigh-{}.txt'.format(types)),
        np.savetxt(os.path.join(job.workspace(), 'neigh-peak.txt'),
                   np.c_[neighbors, count])

class HydrogenBondAnalysis_spce(MDAnalysis.analysis.hbonds.HydrogenBondAnalysis):
    DEFAULT_DONORS = {"spce": tuple(set(['O', 'spce', 'fsi', 'tf2n']))}
    DEFAULT_ACCEPTORS = {"spce": tuple(set(['O','spce', 'fsi', 'tf2n']))}

class HydrogenBondAnalysis_ch3cn(MDAnalysis.analysis.hbonds.HydrogenBondAnalysis):
    DEFAULT_DONORS = {"ch3cn": tuple(set(['ch3cn', 'tf2n', 'fsi', 'O', 'F', 'N']))}
    #                  "fsi": tuple(set(['O', 'F', 'N'])),
    #                  "tf2n": tuple(set(['O', 'F', 'N']))},
    DEFAULT_ACCEPTORS = {"ch3cn": tuple(set(['ch3cn', 'tf2n', 'fsi', 'O', 'F', 'N']))}
    #                  "fsi": tuple(set(['O', 'F', 'N'])),
    #                  "tf2n": tuple(set(['O', 'F', 'N']))}

@Project.operation
@Project.post.isfile(hbond_file)
def calc_hbonds(job):
    import pandas as pd
    print('Loading trj {}'.format(job))
    print(job.get_id())
    anion = job.statepoint()['anion']
    cation = job.statepoint()['cation']
    top_file = os.path.join(job.workspace(), 'top_2.mol2')
    trj_file = os.path.join(job.workspace(),
            'sample_unwrapped.xtc')
    #trj = md.load(trj_file, top=top_file)
    universe = mda.Universe(top_file, trj_file)
    solvent = universe.select_atoms('resname {}'.format(job.sp()['solvent']))

    #hbond  = MDAnalysis.analysis.hbonds.HydrogenBondAnalysis(universe, selection, selection)
    if job.sp()['solvent'] == 'spce':
        hbond = HydrogenBondAnalysis_spce(universe, selection1='name O and resname spce', 
                  selection2='resname {}'.format(anion), selection1_type='both',
                  selection2_type='both', forcefield='spce')
    else:
        hbond = HydrogenBondAnalysis_ch3cn(universe, selection1='name N or name F or name O', 
                  selection2='name O or name F or name N'.format(anion), selection1_type='both',
                  selection2_type='both', forcefield='ch3cn', distance=.6)
        #hbond = HydrogenBondAnalysis_acn(universe, selection1='name O or name F or name N', 
        #          selection2='name O or name F or name N', selection1_type='both',
        #          selection2_type='both', forcefield='acn', distance=4.7)
    #          selection2='name O and resname spce', selection1_type='both',
    #         'name H and resname spce', forcefield='spce')
    hbond.run(start=0, stop=100)
    hbond.generate_table()
    df = pd.DataFrame.from_records(hbond.table)
    import pdb; pdb.set_trace()

    df.to_csv(os.path.join(job.workspace(), 'hbonds_anion.csv'))
    #hbonds = md.wernet_nilsson(sliced_trj, exclude_water=False)
    #hbonds = md.baker_hubbard(sliced_trj, exclude_water=False)

@Project.operation
@Project.pre.isfile(hbond_file)
@Project.post.isfile(hbond_json)
def get_hbonds(job):
    import pandas as pd
    import json
    from collections import defaultdict

    def _calc_hbonds(csv_1, csv_2):
        """
        Read in hbonds from csv and calculate
        """
        df_1 = pd.read_csv(csv_1)
        df_2 = pd.read_csv(csv_2)
        df = pd.concat([df_1, df_2], keys='time') # Join two dataframes together and sort by time
    
        duplicates = df[(df['acceptor_atom'] == 'H') & (df['donor_atom'] == 'H')]

        hbond_dict = _calc_bulk(df)

        return hbond_dict

    
    print(job.get_id())
    if os.path.exists(os.path.join(job.workspace(), 'hbonds_anion.csv')):
        cation = job.sp()['cation']
        anion = job.sp()['anion']
        solvent = job.sp()['solvent']
        temp = job.sp()['T']
    
        # Call 'calc_hbonds'
        path_1 = os.path.join(job.workspace(), 'hbonds.csv')
        path_2 = os.path.join(job.workspace(), 'hbonds_anion.csv')
        #hbonds = _calc_hbonds(path_1, path_2)
        hbonds = _calc_hbonds(path_1, path_2)
        print("Writing to json file")

        if 'bulk' in hbonds:
            with open(os.path.join(job.workspace(), 'hbonds_bulk.json'), 'w') as fp:
                json.dump(hbonds, fp)
    
            with open(os.path.join(job.workspace(), 'hbonds_bulk.pkl'), 'wb') as fp:
                pickle.dump(hbonds, fp)
        else:
            with open(os.path.join(job.workspace(), 'hbonds.json'), 'w') as fp:
                json.dump(hbonds, fp)
    
            with open(os.path.join(job.workspace(), 'hbonds.pkl'), 'wb') as fp:
                pickle.dump(hbonds, fp)
                

def _gromacs_str(op_name, gro_name, sys_name, job):
    """Helper function, returns grompp command string for operation """
    if op_name == 'em':
        mdp = signac.get_project().fn('src/util/mdp_files/{}.mdp'.format(op_name))
    else:
        mdp = signac.get_project().fn('src/util/mdp_files/{}-{}.mdp'.format(op_name, job.sp.T))
    #mdp = signac.get_project().fn('src/util/mdp_files/{}.mdp'.format(op_name))
    cmd = ('gmx grompp -f {mdp} -c {gro}.gro -p {sys}.top -o {op}.tpr --maxwarn 1 && gmx mdrun -deffnm {op} -cpi {op}.cpt -ntmpi 1')
    return workspace_command(cmd.format(mdp=mdp,op=op_name, gro=gro_name, sys=sys_name))

def get_charges(trj, anion):
    charges = np.zeros(shape=(trj.n_atoms))

    for i, atom in enumerate(trj.top.atoms):
        if anion == 'fsi':
            if atom.name == 'fsi':
                charges[i] = -0.6
            elif atom.name == 'li':
                charges[i] = 0.6
        else:
            if atom.name == 'tf2n':
                charges[i] = -0.8
            elif atom.name == 'li':
                charges[i] = 0.8
    return charges

def dipole_moments_md(traj, charges):
    local_indices = np.array([(a.index, a.residue.atom(0).index) for a in traj.top.atoms], dtype='int32')
    local_displacements = md.compute_displacements(traj, local_indices, periodic=False)

    molecule_indices = np.array([(a.residue.atom(0).index, 0) for a in traj.top.atoms], dtype='int32')
    molecule_displacements = md.compute_displacements(traj, molecule_indices, periodic=False)

    xyz = local_displacements + molecule_displacements

    moments = xyz.transpose(0, 2, 1).dot(charges)

    return moments

def _calc_num(df):
    """
    Calc number of hbonds
    """
    # Loop through and grab residues that are hbonded
    residues = list(set([i for i in df.acceptor_resid]))
    hbond_dict = dict()
    times = list(set(df.time))
    #for time in times[800:1000]:
    for time in times:
        for residue in residues:
            hbond = df[(df['time'] == time) & (df['acceptor_resid'] == residue)]
            if len(hbond) in hbond_dict.keys():
                hbond_dict[len(hbond)] += 1
            else:
                hbond_dict[len(hbond)] = 1

    return hbond_dict

def _calc_bulk(df):
    """
    Calc if water is bulk or interfacial
    """
    residues = list(set([i for i in df.donor_resid]))
    hbond_dict = defaultdict(int)
    count_dict = defaultdict(int)
    bulk_list = list()
    inter_list = list()
    times = list(set(df.time))
    #for time in times[800:1000]:
    for time in times:
        for residue in residues:
            hbond = df[(df['time'] == time) & (df['donor_resid'] == residue)]
            #if len(hbond) >= 2: # Check if it has 2 or more hbonds
            if len(hbond) > 0:
                if len(list(set(hbond.acceptor_resnm))) < 2 and 'spce' in list(set(hbond.acceptor_resnm)):
                    hbond_dict['bulk'] += 1
                    bulk_list.append(len(hbond)) 
                else:
                    hbond_dict['interfacial'] += 1
                    inter_list.append(len(hbond))
    import pdb; pdb.set_trace()
            #else:
            #    hbond_dict['interfacial'] += 1

    return hbond_dict


if __name__ == '__main__':
    Project().main()
