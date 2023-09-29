"""
Configuration settings for various project paths and files.

This configuration module defines the directory paths and file locations used 
throughout the project. It ensures consistency and centralization of path settings 
across different scripts and modules.

Attributes:
- PATH_DATA (str): Base directory where various data related to the project are stored.
- PATH_DATASET (str): Directory where the primary dataset for the project resides.
- PATH_IMAGES_GENERATED (str): Location where generated images from the model are saved.
- PATH_VIDEOS_GENERATED (str): Location where generated videos from the model are saved.
- PATH_PARAMS (str): Path to the JSON file containing various model and training parameters.

Notes:
- Ensure that directories exist, or the necessary error handling is in place in scripts using 
these paths.
- Adjust these paths according to your project's directory structure.
"""
# STRUCTURE PROJECT SETTINGS
PATH_DATA = "data"
PATH_DATASET = "dataset"
PATH_IMAGES_GENERATED = 'generated/images'
PATH_VIDEOS_GENERATED = 'generated/videos'

# JSON FILES SETTINGS
PATH_PARAMS = 'src/json/params.json'
