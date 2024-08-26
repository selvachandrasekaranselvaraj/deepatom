from pymatgen.core import Structure
from pymatgen.analysis.descriptors import CoulombMatrix
from pymatgen.analysis.electronic_structure import EwaldSum
from gemd.descriptors import ACSF
from ase.calculators.soap import SOAP
from mbtr import MBTR
from mbtr.descriptors import RadialDescriptor, AngularDescriptor
from mbtr.representations import MBTR as MBTR_rep
import numpy as np

def calculate_coulomb_matrix(structure):
    coulomb_matrix = CoulombMatrix(max_atoms=20)
    return coulomb_matrix.calculate(structure)

def sine_matrix(positions):
    n_atoms = len(positions)
    sine_matrix = np.zeros((n_atoms, n_atoms))
    # Compute sine matrix values here
    return sine_matrix

def calculate_ewald_sum(structure):
    ewald = EwaldSum(structure)
    return ewald.get_ewald_matrix()

def calculate_acsf(atoms):
    acsf = ACSF(cutoff=6.0, n_max=8, l_max=8)
    return acsf.calculate(atoms)

def calculate_soap_descriptors(atoms):
    soap = SOAP(cutoff=5.0, n_max=6, l_max=6, atomic_numbers=[13, 8])
    atoms.set_calculator(soap)
    return atoms.get_calculator().get_descriptor(atoms)

def calculate_mbtr_descriptors(positions, atomic_numbers):
    radial_descriptor = RadialDescriptor(cutoff=5.0, n_max=6, spacing='log')
    angular_descriptor = AngularDescriptor(cutoff=5.0, l_max=6)
    mbtr = MBTR_rep(radial_descriptor=radial_descriptor, angular_descriptor=angular_descriptor, normalization='all')
    return mbtr.transform(positions, atomic_numbers)
