import numpy as np
from ase import Atoms

def read_lammps_trajectory(file_path):
    from lammps import lammps
    lmp = lammps()
    lmp.command(f"read_data {file_path}")
    positions = lmp.extract_lammps_data('positions')
    return np.array(positions)

def convert_to_ase_atoms(positions, atomic_numbers, lattice_vectors):
    return Atoms(positions=positions, numbers=atomic_numbers, cell=lattice_vectors, pbc=True)
