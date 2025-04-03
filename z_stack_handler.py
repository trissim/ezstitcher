import os
import re
import shutil
from pathlib import Path

def organize_zstack_folders(plate_folder):
    """
    Check if TimePoint_1 contains ZStep_* folders, and if so:
    1. Move all files from each ZStep folder to TimePoint_1 
    2. Rename files to include _z{***} in the filename
    
    Args:
        plate_folder: Base folder for the plate
        
    Returns:
        bool: True if Z-stack was detected and organized, False otherwise
    """
    # Construct path to TimePoint_1
    timepoint_path = Path(plate_folder) / "TimePoint_1"
    
    if not timepoint_path.exists():
        print(f"TimePoint_1 folder does not exist in {plate_folder}")
        return False
    
    # Check for ZStep_* folders
    zstep_pattern = re.compile(r'^ZStep_(\d+)$')
    zstep_folders = []
    
    for item in timepoint_path.iterdir():
        if item.is_dir():
            match = zstep_pattern.match(item.name)
            if match:
                # Store tuple of (folder_path, z_index)
                zstep_folders.append((item, int(match.group(1))))
    
    if not zstep_folders:
        print(f"No ZStep folders found in {timepoint_path}")
        return False
    
    # Sort by Z-index
    zstep_folders.sort(key=lambda x: x[1])
    print(f"Found {len(zstep_folders)} Z-step folders: {[f[0].name for f in zstep_folders]}")
    
    # Process each Z-step folder
    for zstep_folder, z_index in zstep_folders:
        # Zero-pad z_index to 3 digits
        z_suffix = f"_z{z_index:03d}"
        
        print(f"Processing {zstep_folder.name} (z-index: {z_index})")
        
        # Get all image files in the folder
        image_files = []
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            image_files.extend(list(zstep_folder.glob(f"*{ext}")))
        
        # Move and rename each file
        for img_file in image_files:
            # Insert z_suffix before the file extension
            base, ext = os.path.splitext(img_file.name)
            new_filename = f"{base}{z_suffix}{ext}"
            destination = timepoint_path / new_filename
            
            print(f"  Moving {img_file.name} to {new_filename}")
            
            # Move the file
            shutil.move(str(img_file), str(destination))
    
#    # Optionally, remove empty Z-step folders after processing
#    for zstep_folder, _ in zstep_folders:
#        print(f"Removing empty folder {zstep_folder.name}")
#        zstep_folder.rmdir()
    for zstep_folder, _ in zstep_folders:
        # Delete all files within the folder first
        for file in zstep_folder.iterdir():
            if file.is_file():
                file.unlink()  # Delete the file
        
        print(f"Removing empty folder {zstep_folder.name}")
        zstep_folder.rmdir()
    
    print(f"Z-stack organization complete. All files moved to {timepoint_path} with z-index in filenames.")
    return True

# Function to handle both z-stacks and regular folders
def preprocess_plate_folder(plate_folder):
    """
    Preprocesses a plate folder before stitching:
    1. Checks if it contains a Z-stack and organizes it if needed
    2. Performs any other necessary preprocessing steps
    
    Args:
        plate_folder: Base folder for the plate
        
    Returns:
        bool: True if preprocessing was successful
    """
    print(f"Preprocessing plate folder: {plate_folder}")
    
    # First, check and organize Z-stacks if present
    has_zstack = organize_zstack_folders(plate_folder)
    
    if has_zstack:
        print(f"Z-stack detected and organized in {plate_folder}")
    else:
        print(f"No Z-stack detected in {plate_folder}")
    
    # Other preprocessing steps could be added here
    
    return True

## Integration with main code
#def modified_process_plate_folder(plate_folder, **kwargs):
#    """
#    Modified version of process_plate_folder that handles Z-stacks
#    
#    Args:
#        plate_folder: Base folder for the plate
#        **kwargs: All the original parameters for process_plate_folder
#    """
#    # Preprocess to handle Z-stacks
#    preprocess_plate_folder(plate_folder)
#    
#    # Now call the original process_plate_folder function
#    process_plate_folder(plate_folder, **kwargs)

## Example usage
#if __name__ == "__main__":
#    # Example plate folder to process
#    plate_folder = "/path/to/your/plate/folder"
#    
#    # Preprocess and then process the plate folder
#    modified_process_plate_folder(
#        plate_folder,
#        reference_channels=["1","2"],
#        preprocessing_funcs={"1": process_bf},
#        tile_overlap=10
#    )
