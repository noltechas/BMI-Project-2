import numpy as np
import math
import os

def read_pdbm(filename):
    atoms = []
    with open(filename, "r") as file:
        for line in file:
            if line.startswith("ATOM"):
                atom_data = line.split()
                atom = {
                    'type': atom_data[2],
                    'x': float(atom_data[5]),
                    'y': float(atom_data[6]),
                    'z': float(atom_data[7]),
                    'charge': float(atom_data[-2]),  # Read charge from the second last column
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

def lennard_jones(c6, c12, r):
    r_inv = 1 / r
    r6 = r_inv ** 6
    r12 = r6 * r6
    return c12 * r12 - c6 * r6

def electrostatic(qi, qj, r, epsilon):
    return qi * qj / (r * epsilon)

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
    for protein_atom in protein:
        for ligand_atom in ligand:
            r = distance(protein_atom, ligand_atom) / 10  # Convert angstroms to nanometers
            c6 = c6_matrix[protein_atom['type']][ligand_atom['type']]
            c12 = c12_matrix[protein_atom['type']][ligand_atom['type']]
            lj_energy = lennard_jones(c6, c12, r)
            electrostatic_energy = electrostatic(protein_atom['charge'], ligand_atom['charge'], r, epsilon)
            total_energy += lj_energy + electrostatic_energy
    return total_energy


def main():
    protein = read_pdbm("protein.pdbm")
    ligand_starting = read_pdbm("ligand_starting.pdbm")
    ligand_actual = read_pdbm("ligand_actual.pdb")
    c6_matrix, c12_matrix = load_lj_params("ffG43b1nb.params")
    epsilon = 138.935485

    rmsd_start_actual = rmsd(ligand_starting, ligand_actual)
    print(f"RMSD between ligand_starting.pdbm and ligand_actual.pdb: {rmsd_start_actual}")

    n_samples = 100
    beta_step = 2 * math.pi / n_samples
    best_energy = float('inf')
    best_beta = None
    best_rotated_ligand = None

    for i in range(n_samples):
        beta = i * beta_step
        rotated_ligand = rotate_ligand(ligand_starting, beta)
        energy = docking_energy(protein, rotated_ligand, c6_matrix, c12_matrix, epsilon)
        print(f"Energy for β = {beta}: {energy}")  # Print energy values for each β angle

        if energy < best_energy:
            best_energy = energy
            best_beta = beta
            best_rotated_ligand = rotated_ligand

    print(f"\nMost favorable energy: {best_energy}")
    print(f"Most favorable β angle: {best_beta}")

    rmsd_best_actual = rmsd(best_rotated_ligand, ligand_actual)
    print(f"RMSD between most favorable pose and ligand_actual.pdb: {rmsd_best_actual}")

if __name__ == "__main__":
    main()

