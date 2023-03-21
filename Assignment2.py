import numpy as np

# Step 1: Read in protein.pdbm and ligand starting.pdbm files
with open('protein.pdbm') as f:
    protein_lines = f.readlines()
with open('ligand_starting.pdbm') as f:
    ligand_lines = f.readlines()

# Step 2: Parse Lennard-Jones parameters
lj_params = {}
with open('ffG43b1nb.params') as f:
    for line in f:
        atomtype1, atomtype2, c6, c12 = line.split()
        lj_params[(atomtype1, atomtype2)] = (float(c6), float(c12))

# Step 3: Parse partial charges
charges = {}
with open('protein.pdbm') as f:
    for line in f:
        if line.startswith('ATOM'):
            atom_id = line[6:11].strip()
            charge_str = line[79:81].strip()
            charge = float(charge_str) if charge_str else 0.0
            charges[atom_id] = charge



# Step 4: Extract atom coordinates and types
protein_coords = []
ligand_coords = []
for line in protein_lines:
    if line.startswith('ATOM'):
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        atom_type = line[76:78].strip()
        if atom_type in ['C', 'N', 'O', 'S']:
            protein_coords.append([x, y, z])

for line in ligand_lines:
    if line.startswith('ATOM'):
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        ligand_coords.append([x, y, z])

# Step 5: Define Lennard-Jones potential function
def lj_potential(r, c6, c12):
    return c12 / (r ** 12) - c6 / (r ** 6)

# Step 6: Calculate electrostatic potential function
def electrostatic_potential(r, charge1, charge2):
    eps = 138.935485
    if r < 1.32:
        eps_r = 8
    else:
        A = 6.02944
        B = 72.37056
        lamda = 0.018733345
        k = 213.5782
        eps_r = A + (B / (1 + k * np.exp(-lamda * B * r)))
    return (charge1 * charge2) / (4 * np.pi * eps * eps_r * r)

# Step 7: Calculate total potential energy
def total_energy(protein_coords, ligand_coords):
    energy = 0
    for i in range(len(protein_coords)):
        for j in range(len(ligand_coords)):
            distance = np.linalg.norm(protein_coords[i] - ligand_coords[j])
            atom_type1 = ' '.join(protein_lines[i].split()[2:4])
            atom_type2 = ' '.join(ligand_lines[j].split()[2:4])
            if (atom_type1, atom_type2) in lj_params:
                c6, c12 = lj_params[(atom_type1, atom_type2)]
                lj_energy = lj_potential(distance / 10, c6, c12)
                charge1 = charges.get(protein_lines[i][6:11].strip(), 0)
                charge2 = charges.get(ligand_lines[j][6:11].strip(), 0)
                elec_energy = electrostatic_potential(distance, charge1, charge2)
                energy += lj_energy + elec_energy
    return energy

# Step 8: Sample rotational space
rot_angles = range(-180, 181, 5) # Sample every 5 degrees
energies = []
for angle in rot_angles:
    rot_matrix = np.array([[np.cos(np.radians(angle)), 0, np.sin(np.radians(angle))],
                           [0, 1, 0],
                           [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]])
    ligand_coords_rotated = np.dot(ligand_coords, rot_matrix)
    energy = total_energy(protein_coords, ligand_coords_rotated)
    energies.append(energy)

# Step 9: Find optimal docking pose
min_energy_idx = np.argmin(energies)
optimal_angle = rot_angles[min_energy_idx]
optimal_energy = energies[min_energy_idx]
optimal_matrix = np.array([[np.cos(np.radians(optimal_angle)), 0, np.sin(np.radians(optimal_angle))],
                           [0, 1, 0],
                           [-np.sin(np.radians(optimal_angle)), 0, np.cos(np.radians(optimal_angle))]])
ligand_coords_optimal = np.dot(ligand_coords, optimal_matrix)

# Step 10: Output results
print(f"Optimal rotation angle: {optimal_angle}")
print(f"Optimal energy: {optimal_energy}")
print(f"Optimal ligand coordinates:\n{ligand_coords_optimal}")
