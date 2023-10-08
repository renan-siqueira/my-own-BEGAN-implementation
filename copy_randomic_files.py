"""Script to copy files based on provided configurations."""

import os
import shutil
import random

from src.config.settings import (
    COPY_DESTINATION_FOLDER,
    COPY_SOURCE_FOLDER,
    COPY_PERCENTAGE_TO_COPY,
    COPY_RANDOM_MODE,
    COPY_FIXED_NUMBER_TO_COPY
)


def copy_files(
        source_folder,
        destination_folder,
        percentage=None,
        fixed_number=None,
        random_mode=True
    ):
    """
    Copy files from source to destination folder based on given percentage or fixed number.

    Parameters:
    - source_folder (str): Path to the source folder.
    - destination_folder (str): Path to the destination folder.
    - percentage (int): Percentage of files to copy. Defaults to None.
    - fixed_number (int): Fixed number of files to copy. Defaults to None.
    - random_mode (bool): Whether to copy files randomly or sequentially.

    Returns:
    - None
    """
    # Listing all files from the source folder
    files = [file_name for file_name in os.listdir(source_folder)
             if os.path.isfile(os.path.join(source_folder, file_name))]

    # Calculating the number of files to be copied
    if percentage is not None:
        total_to_copy = int(len(files) * percentage / 100)
    elif fixed_number is not None:
        total_to_copy = min(fixed_number, len(files))
    else:
        raise ValueError("Either percentage or fixed_number must be provided!")

    # Selecting files based on the mode (random or sequential)
    if random_mode:
        chosen_files = random.sample(files, total_to_copy)
    else:
        chosen_files = files[:total_to_copy]

    # Copying the selected files to the destination folder
    for file_name in chosen_files:
        shutil.copy2(os.path.join(source_folder, file_name), destination_folder)

    print(f"{total_to_copy} files have been copied from {source_folder} to {destination_folder}.")


if __name__ == '__main__':
    copy_files(
        COPY_SOURCE_FOLDER,
        COPY_DESTINATION_FOLDER,
        percentage=COPY_PERCENTAGE_TO_COPY,
        fixed_number=COPY_FIXED_NUMBER_TO_COPY,
        random_mode=COPY_RANDOM_MODE
    )
