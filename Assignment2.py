import numpy as np
import math
import os

atom_charges = {
    'C': 0.6163,
    'CA': 0.5973,
    'CB': -0.0152,
    'CG': -0.1415,
    'CG1': -0.2061,
    'CG2': -0.1913,
    'CD': 0.8185,
    'CD1': 0.7341,
    'CD2': 0.7455,
    'CE': 0.9916,
    'CE1': 0.9841,
    'CE2': 0.9565,
    'CE3': 0.9876,
    'CZ': 0.9616,
    'CZ2': 0.9756,
    'CZ3': 0.9769,
    'CH2': 0.9949,
    'N': -0.4787,
    'NA': -0.3660,
    'NB': -0.5932,
    'NC': -0.8360,
    'ND1': -0.5659,
    'ND2': -0.5816,
    'NE': -0.6690,
    'NE1': -0.7582,
    'NE2': -0.7902,
    'NZ': -0.9209,
    'O': -0.5196,
    'OH': -0.5761,
    'OXT': -0.5103,
    'S': 1.0560,
    'SD': 0.7843,
    'SG': 0.7657,
}


def read_pdbm(file_path, atom_charges):
    atoms = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_type = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                charge = atom_charges.get(atom_type, 0)  # Use 0 as a default charge if the atom type is not in the dictionary

                atom = {
                    'type': atom_type,
                    'x': x,
                    'y': y,
                    'z': z,
                    'charge': charge
                }
                atoms.append(atom)
    return atoms



def load_lj_params(filename):
    c6_matrix = {}
    c12_matrix = {}

    with open(filename, "r") as file:
        for line in file:
            if line.strip():
                data = line.split()
                atom_type1, atom_type2, c6, c12 = data
                if atom_type1 not in c6_matrix:
                    c6_matrix[atom_type1] = {}
                if atom_type1 not in c12_matrix:
                    c12_matrix[atom_type1] = {}
                c6_matrix[atom_type1][atom_type2] = float(c6)
                c12_matrix[atom_type1][atom_type2] = float(c12)

                # Add the reverse pairs for easier access later
                if atom_type2 not in c6_matrix:
                    c6_matrix[atom_type2] = {}
                if atom_type2 not in c12_matrix:
                    c12_matrix[atom_type2] = {}
                c6_matrix[atom_type2][atom_type1] = float(c6)
                c12_matrix[atom_type2][atom_type1] = float(c12)

    return c6_matrix, c12_matrix


def distance(atom1, atom2):
    dx = atom1['x'] - atom2['x']
    dy = atom1['y'] - atom2['y']
    dz = atom1['z'] - atom2['z']
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def electrostatic(atom1, atom2, epsilon, distance):
    q1 = atom1['charge']
    q2 = atom2['charge']
    conversion_factor = 332.06371
    energy = (conversion_factor * q1 * q2) / (epsilon * distance)
    return energy


def lennard_jones(atom1, atom2, c6_matrix, c12_matrix, distance):
    c6 = c6_matrix[atom1['type']][atom2['type']]
    c12 = c12_matrix[atom1['type']][atom2['type']]
    energy = c12 / (distance ** 12) - c6 / (distance ** 6)
    return energy



def rmsd(coords1, coords2):
    # Filter out hydrogen atoms
    coords1_filtered = [coord for coord in coords1 if coord['type'][0] != 'H']
    coords2_filtered = [coord for coord in coords2 if coord['type'][0] != 'H']

    assert len(coords1_filtered) == len(coords2_filtered), "Coordinate arrays must be the same length"
    n_atoms = len(coords1_filtered)
    diff_squared_sum = 0
    for i in range(n_atoms):
        diff_squared_sum += np.sum((np.array([coords1_filtered[i]['x'], coords1_filtered[i]['y'], coords1_filtered[i]['z']]) - np.array([coords2_filtered[i]['x'], coords2_filtered[i]['y'], coords2_filtered[i]['z']])) ** 2)
    return np.sqrt(diff_squared_sum / n_atoms)


def rotate_ligand(ligand, beta):
    rotation_matrix = np.array([
        [math.cos(beta), 0, math.sin(beta)],
        [0, 1, 0],
        [-math.sin(beta), 0, math.cos(beta)],
    ])

    rotated_ligand = []
    for atom in ligand:
        original_coords = np.array([atom['x'], atom['y'], atom['z']])
        rotated_coords = np.dot(rotation_matrix, original_coords)
        rotated_atom = atom.copy()
        rotated_atom.update({'x': rotated_coords[0], 'y': rotated_coords[1], 'z': rotated_coords[2]})
        rotated_ligand.append(rotated_atom)
    return rotated_ligand

def docking_energy(protein, ligand, c6_matrix, c12_matrix, epsilon):
    total_energy = 0
    atom_pair_energies = []

    for protein_atom in protein:
        for ligand_atom in ligand:
            dist = distance(protein_atom, ligand_atom)

            electrostatic_energy = electrostatic(protein_atom, ligand_atom, epsilon, dist)
            lj_energy = lennard_jones(protein_atom, ligand_atom, c6_matrix, c12_matrix, dist)

            atom_pair_energies.append({
                'protein_atom': protein_atom,
                'ligand_atom': ligand_atom,
                'electrostatic_energy': electrostatic_energy,
                'lennard_jones_energy': lj_energy
            })

            total_energy += electrostatic_energy + lj_energy

    return total_energy, atom_pair_energies



def main():
    # Read protein, starting ligand, and actual ligand from PDB files
    protein = read_pdbm("protein.pdbm", atom_charges)
    first_atom = protein[0]
    print("First atom coordinates: {:.3f}  {:.3f}  {:.3f}".format(first_atom['x'], first_atom['y'], first_atom['z']))
    ligand_starting = read_pdbm("ligand_starting.pdbm", atom_charges)
    ligand_actual = read_pdbm("ligand_actual.pdb", atom_charges)

    # Load Lennard-Jones parameters and set epsilon value
    c6_matrix, c12_matrix = load_lj_params("ffG43b1nb.params")
    epsilon = 138.935485

    # Calculate RMSD between starting and actual ligands
    rmsd_start_actual = rmsd(ligand_starting, ligand_actual)
    print("RMSD between ligand starting and actual:", rmsd_start_actual)

    # Set angle step size and initialize variables for best pose
    angle_step = math.radians(3.6)
    best_energy = float('inf')
    best_beta = 0
    best_rotated_ligand = None

    # Loop over all possible beta angles in 3.6 degree increments
    for i in range(int(360 / 3.6)):
        # Calculate beta angle and rotate ligand
        beta = i * angle_step
        rotated_ligand = rotate_ligand(ligand_starting, beta)

        # Calculate docking energy for rotated ligand
        energy, atom_pair_energies = docking_energy(protein, rotated_ligand, c6_matrix, c12_matrix, epsilon)
        print(f"Energy for β = {beta}: {energy}")

        if i == 0:  # Print the first protein-ligand atom pair energy values
            first_pair = atom_pair_energies[0]
            print("First protein-ligand atom pair energy values:")
            print(f"Electrostatic energy: {first_pair['electrostatic_energy']:.9f}")
            print(f"Lennard-Jones energy: {first_pair['lennard_jones_energy']:.9f}")

        # Update best pose if current energy is lower than previous best
        if energy < best_energy:
            best_energy = energy
            best_beta = beta
            best_rotated_ligand = rotated_ligand

    # Print out best pose and corresponding energy
    print("Best β angle:", best_beta)
    print("Best energy:", best_energy)

    # Calculate RMSD between best pose and actual ligand
    rmsd_best_actual = rmsd(best_rotated_ligand, ligand_actual)
    print("RMSD between best pose and actual:", rmsd_best_actual)

    # Return the best rotation matrix
    best_rotation_matrix = np.array([
        [math.cos(best_beta), 0, math.sin(best_beta)],
        [0, 1, 0],
        [-math.sin(best_beta), 0, math.cos(best_beta)],
    ])
    return best_rotation_matrix



if __name__ == "__main__":
    best_rotation_matrix = main()
    print("Best rotation matrix:\n", best_rotation_matrix)
