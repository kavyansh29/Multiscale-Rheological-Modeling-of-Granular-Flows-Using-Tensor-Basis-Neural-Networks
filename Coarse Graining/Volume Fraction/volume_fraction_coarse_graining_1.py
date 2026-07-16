import glob
import torch
import numpy as np
import os
from tqdm import tqdm

def get_avg_coordNum(file_path):
    """Get average coordination number from the output file."""
    last_value = None
    
    with open(file_path, 'r') as f:
        for line in f:
            # Check if line starts with a number (timestep line)
            if line.strip() and line.split()[0].isdigit():
                # Get the last column value
                columns = line.split()
                last_value = columns[-1]
    coordination_number = float(last_value)
    return coordination_number

def read_radius_from_config(config_file):
    """
    Read LAMMPS config file and extract radius from diameter column
    Returns array of radius values ordered by atom ID
    """
    with open(config_file, 'r') as f:
        lines = f.readlines()
    
    # Find Atoms section
    particles_start = None
    for i, line in enumerate(lines):
        if 'Atoms' in line:
            particles_start = i + 2  # Skip "Atoms" line and blank line
            break
    
    if particles_start is None:
        raise ValueError("Atoms section not found in config file")
    
    # Extract ID and diameter
    particle_data = []
    for line in lines[particles_start:]:
        if not line.strip():  # Stop at blank line
            break
        parts = line.split()
        if len(parts) >= 3:
            particle_id = int(parts[0])
            diameter = float(parts[2])  # dia is 3rd column
            particle_data.append((particle_id, diameter))
    
    # Sort by particle ID to ensure correct order
    particle_data.sort(key=lambda x: x[0])
    
    # Extract diameters and convert to radius
    diameters = np.array([d for _, d in particle_data])
    radius = diameters / 2.0
    
    return radius

def read_dump_file(file_path):
    """
    Read a LAMMPS dump file and extract particle data
    Returns a dictionary with keys: 'N', 'dim', 'x', 'v', 'type', etc.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    frame_data = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ITEM: TIMESTEP"):
            i += 1
            frame_data['timestep'] = int(lines[i].strip())
        elif line.startswith("ITEM: NUMBER OF ATOMS"):
            i += 1
            frame_data['N'] = int(lines[i].strip())
        elif line.startswith("ITEM: BOX BOUNDS"):
            i += 1
            bounds = []
            for _ in range(3):
                bounds.append(list(map(float, lines[i].strip().split())))
                i += 1
            frame_data['box_bounds'] = np.array(bounds)
            frame_data['dim'] = 3  # Assuming 3D
            continue  # Skip incrementing i here
        elif line.startswith("ITEM: ATOMS"):
            headers = line.split()[2:]  # Get column headers
            data = []
            for j in range(frame_data['N']):
                i += 1
                parts = lines[i].strip().split()
                data.append([float(part) for part in parts])
            data_array = np.array(data)
            
            # Map headers to data columns
            for idx, header in enumerate(headers):
                frame_data[header] = data_array[:, idx]
        i += 1
    
    return frame_data

def read_all_frames(folder_path, pattern="Dump_Shear.*", max_frames=1000000000):
    """
    Read all dump files in the specified folder matching the pattern
    Returns a list of frames (dictionaries)
    """
    file_paths = sorted(glob.glob(os.path.join(folder_path, pattern)))
    frames = []
    for file_path in tqdm(
        file_paths[:max_frames],
        desc="Reading LAMMPS dump files",
        unit="file"):

        frame = read_dump_file(file_path)
        frames.append(frame)
    return frames
# ====================================================================================================================
# ====================================================================================================================
frames = read_all_frames(
    "/Users/kavyanshrajsingh/Desktop/Data/Dump",
    pattern="Dump_Shear.*"
)
radius = read_radius_from_config(config_file = r"/Users/kavyanshrajsingh/Desktop/Data/Dump/Shear_Boundary_50x40.txt")
for frame in frames:
    frame["radius"] = radius
#====================================================================================================================
frames_sorted = sorted(frames, key=lambda x: int(x['timestep']))