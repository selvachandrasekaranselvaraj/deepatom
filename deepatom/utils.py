import numpy as np
from ase import Atoms
from ase.io import read
    
def read_lammps_trajectory(dump_file):
    '''
    Reads a LAMMPS trajectory file, skipping an initial number of frames based on the total number of frames.

    :return: list, trajectory data
    '''
    #For testing, you can provide the input_file as a command line argument
    try:
        input_file = dump_file #sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"autopsy lammps_traj_file  \"")
        print()
        exit()
    # For demonstration purposes, using fixed input_file
    #input_file = 'd.lmp'

    print(f'Reading {input_file}...')

    # Read the LAMMPS trajectory file
    data = read(input_file, format="lammps-dump-text", index=":")

    n_frames = len(data)

    # Define the number of initial frames to skip based on the total number of frames
    if n_frames > 10000:
        initial_frames_skipped = 200
    elif n_frames >= 1000 and n_frames < 10000:
        initial_frames_skipped = 100
    elif n_frames > 5 and n_frames < 1000:
        initial_frames_skipped = 1
    else:
        initial_frames_skipped = 0

    initial_frames_skipped = 0
    # Skip the initial frames
    data = data[initial_frames_skipped:n_frames-1]


    print(f"Number of initial frames skipped are {initial_frames_skipped}")
    print(f"Number of frames in the trajectory are {len(data)}")
    print()

    return data

#def convert_to_ase_atoms(positions, atomic_numbers, lattice_vectors):
#    return Atoms(positions=positions, numbers=atomic_numbers, cell=lattice_vectors, pbc=True)
