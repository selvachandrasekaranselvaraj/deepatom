"""
This module contains functions for calculating various atomic descriptors used in materials science and machine learning.

The descriptors implemented here include:
- Sine Matrix
- Atom-centered Symmetry Functions (ACSF)
- Smooth Overlap of Atomic Positions (SOAP)

Note: Some functions (Coulomb Matrix, Ewald Sum, and MBTR) are currently commented out and may be implemented in future versions.
"""

from dscribe.descriptors import ACSF, SOAP
import numpy as np

# def calculate_coulomb_matrix(structure):
#     coulomb_matrix = CoulombMatrix(max_atoms=20)
#     return coulomb_matrix.calculate(structure)

def sine_matrix(positions):
    """
    Calculate the sine matrix for a given set of atomic positions.

    The sine matrix is a distance-based descriptor that captures the pairwise distances
    between atoms in a structure using sine functions.

    Parameters:
    positions (numpy.ndarray): An array of shape (n_atoms, 3) containing the 3D coordinates
                               of each atom in the structure.

    Returns:
    numpy.ndarray: A 2D array of shape (n_atoms, n_atoms) representing the sine matrix.

    Note:
    This function currently returns a zero matrix and needs to be implemented.
    """
    n_atoms = len(positions)
    sine_matrix = np.zeros((n_atoms, n_atoms))
    # Compute sine matrix values here
    return sine_matrix

# def calculate_ewald_sum(structure):
#     ewald = EwaldSum(structure)
#     return ewald.get_ewald_matrix()

def calculate_acsf(structure, species):
    """
    Calculate the Atom-centered Symmetry Functions (ACSF) descriptor for a given structure.

    ACSF is a descriptor that captures local atomic environments using a set of symmetry functions.

    Parameters:
    structure (ase.Atoms): An ASE Atoms object representing the atomic structure.
    species (list): A list of atomic species present in the structure.

    Returns:
    numpy.ndarray: An array containing the ACSF descriptor values for the given structure.

    Note:
    The ACSF parameters (r_cut, g2_params, g4_params) are hardcoded and may need to be adjusted
    for different use cases.
    """
    acsf = ACSF(
        species=species,
        r_cut=6.0,
        g2_params=[[1, 1], [1, 2], [1, 3]],
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
        periodic=True,
    )
    return acsf.create(structure)

def calculate_soap(structure, species):
    """
    Calculate the Smooth Overlap of Atomic Positions (SOAP) descriptor for a given structure.

    SOAP is a descriptor that represents local atomic environments using an expansion of
    Gaussian-smeared atomic densities into spherical harmonics.

    Parameters:
    structure (ase.Atoms): An ASE Atoms object representing the atomic structure.
    species (list): A list of atomic species present in the structure.

    Returns:
    numpy.ndarray: An array containing the SOAP descriptor values for the given structure.

    Note:
    The SOAP parameters (r_cut, n_max, l_max, sigma) are hardcoded and may need to be adjusted
    for different use cases.
    """
    soap = SOAP(
        species=species,
        periodic=True,
        r_cut=5,
        n_max=8,
        l_max=8,
        sigma=0.2,
        sparse=False,
    )
    return soap.create(structure)