from dscribe.descriptors import ACSF, SOAP
import numpy as np

#def calculate_coulomb_matrix(structure):
#    coulomb_matrix = CoulombMatrix(max_atoms=20)
#    return coulomb_matrix.calculate(structure)

def sine_matrix(positions):
    n_atoms = len(positions)
    sine_matrix = np.zeros((n_atoms, n_atoms))
    # Compute sine matrix values here
    return sine_matrix

#def calculate_ewald_sum(structure):
#    ewald = EwaldSum(structure)
#    return ewald.get_ewald_matrix()

def calculate_acsf(structure, species):
    acsf = ACSF(
        species=species,
        r_cut=6.0,
        g2_params=[[1, 1], [1, 2], [1, 3]],
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
        periodic=True,
    )
    return acsf.create(structure)

def calculate_soap(structure, species):
    soap = SOAP(
        species=species,
        periodic=True,
        r_cut=5,
        n_max=8,
        l_max=8,
        sigma=0.2,
        #average="off",
        sparse=False,
        #ccompression={"mode":"mu1nu1"},
        )
    return soap.create(structure)

