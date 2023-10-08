"""
Configuration settings for various project paths and files.

This configuration module defines the directory paths, file locations, and parameters
used throughout the project. It ensures consistency and centralization of settings
across different scripts and modules.

Attributes:
- PATH_DATA (str): Base directory where various data related to the project are stored.
- PATH_DATASET (str): Directory where the primary dataset for the project resides.
- PATH_IMAGES_GENERATED (str): Location where generated images from the model are saved.
- PATH_VIDEOS_GENERATED (str): Location where generated videos from the model are saved.
- PATH_PARAMS (str): Path to the JSON file containing various model and training parameters.
- COPY_SOURCE_FOLDER (str): Path to the source folder from which files are read.
- COPY_DESTINATION_FOLDER (str): Path to the destination folder where files are copied to.
- COPY_PERCENTAGE_TO_COPY (int): Percentage of files to copy from the source folder.
- COPY_RANDOM_MODE (bool): Determines if files should be copied randomly or sequentially.
- COPY_FIXED_NUMBER_TO_COPY (int or None): Fixed number of files to copy. If None, the 
percentage-based method is used.

Notes:
- Ensure that directories exist, or the necessary error handling is in place in scripts using 
these paths.
- Adjust these paths and settings according to your project's directory structure and requirements.
"""

# STRUCTURE PROJECT SETTINGS
PATH_DATA = "data"
PATH_DATASET = "dataset"
PATH_IMAGES_GENERATED = 'generated/images'
PATH_VIDEOS_GENERATED = 'generated/videos'

# JSON FILES SETTINGS
PATH_PARAMS = 'src/json/params.json'

# COPY RANDOMIC FILES SETTINGS
COPY_SOURCE_FOLDER = '__your_source_folder_to_read_files__'
COPY_DESTINATION_FOLDER = '__your_destination_folder_to_copy_files__'
COPY_PERCENTAGE_TO_COPY = int(50)
COPY_RANDOM_MODE = True
COPY_FIXED_NUMBER_TO_COPY = None
