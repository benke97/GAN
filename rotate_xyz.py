import pandas as pd
import numpy as np

def read_xyz(file_name):
    with open(file_name, 'r') as file:
        atoms = file.readlines()
        atom_count = int(atoms[0].strip())
        atom_data = [line.split() for line in atoms[2:2 + atom_count]]  # Skip header and read atom lines
    return pd.DataFrame(atom_data, columns=['label', 'x', 'y', 'z']).astype({'x': 'float', 'y': 'float', 'z': 'float'})

def write_xyz(df, file_name):
    with open(file_name, 'w') as file:
        file.write(f'{len(df)}\n')
        file.write('XYZ file after rotation\n')
        for index, row in df.iterrows():
            file.write(f'{row["label"]} {row["x"]} {row["y"]} {row["z"]}\n')

def rotate_points(df, angle_deg):
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                [np.sin(angle_rad), np.cos(angle_rad), 0],
                                [0, 0, 1]])
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].dot(rotation_matrix)
    return df

# Example usage
input_file = 'cropped.xyz'  # Replace with your input file path
output_file = 'cropped_out.xyz'  # Replace with your output file path

# Read, rotate, and save the structure
df = read_xyz(input_file)
center_of_mass = df[['x', 'y', 'z']].mean()
df_centered = df[['x', 'y', 'z']] - center_of_mass  # Center the points
rotated_df = df.copy()
rotated_df[['x', 'y', 'z']] = rotate_points(df_centered, 40) + center_of_mass  # Rotate and shift back
write_xyz(rotated_df, output_file)