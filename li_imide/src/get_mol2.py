import inspect
import os
import mbuild as mb
from pkg_resources import resource_filename

def GetSolv(solv_name):
    cache_dir = '/raid6/homes/firstcenter/imideil-2/src/util/lib/mol2'
    filename = '{}.mol2'.format(solv_name)
    if any(file == filename for file in os.listdir(cache_dir)):
        solv = mb.load(os.path.join(cache_dir, filename))
        solv.name = solv_name

    return solv

def GetIL(il_name):
    #cache_dir = resource_filename('/raid6/homes/raymat/science/', 'util/lib    /mol2')
    cache_dir = '/raid6/homes/firstcenter/imideil-2/src/util/lib/mol2'
    filename = '{}.mol2'.format(il_name)
    if any(file == filename for file in os.listdir(cache_dir)):
        il = mb.load(os.path.join(cache_dir, filename))
        il.name = il_name

    return il
