"""
This script provides utility functions for processing and resizing images.

Functions:
- `gamma_correction`: Adjusts the gamma levels of an image, making it appear lighter 
or darker.
- `process_and_resize_image`: Processes an image by resizing it to a specified width, 
sharpening it, enhancing its contrast, and applying gamma correction.

Dependencies:
- PIL (Image, ImageFilter, ImageEnhance): For various image processing operations.
- numpy: For mathematical operations and matrix manipulation.

Notes:
- The `gamma_correction` function accepts an image and a gamma value. A gamma value 
greater than 1 makes the image darker, while a value less than 1 makes it lighter.
- The `process_and_resize_image` function resizes the image proportionally based on the 
specified width, sharpens the image, enhances its contrast by 25%, and optionally adjusts 
its gamma. The function returns the processed image in numpy array format.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def gamma_correction(image, gamma):
    """
    Adjusts the gamma level of an image.
    
    Parameters:
    - image (PIL.Image.Image or numpy.array): The input image.
    - gamma (float): The gamma correction value. Values greater than 1 will 
                     make the image darker, and values less than 1 will make 
                     it lighter.

    Returns:
    - PIL.Image.Image: Gamma-corrected image.
    """
    image = np.array(image)
    image = 255 * (image / 255) ** (1 / gamma)
    return Image.fromarray(np.uint8(image))


def process_and_resize_image(image_np, new_width=256, gamma=1.1):
    """
    Processes an image by resizing, sharpening, enhancing contrast, and 
    applying gamma correction.
    
    Parameters:
    - image_np (numpy.array): The input image in numpy array format.
    - new_width (int, optional): Desired width for the resized image. 
                                 Default is 256 pixels.
    - gamma (float, optional): Gamma correction value for final processing step. 
                               Default is 1.1.
    
    Returns:
    - numpy.array: Processed image in numpy array format.
    """
    original_image = Image.fromarray(image_np)
    width, height = original_image.size

    new_height = int(new_width * height / width)

    resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    resized_image = resized_image.filter(ImageFilter.SHARPEN)

    enhancer = ImageEnhance.Contrast(resized_image)
    enhanced_image = enhancer.enhance(1.25)

    gamma_corrected_image = gamma_correction(enhanced_image, gamma)
    final_image_np = np.array(gamma_corrected_image)

    return final_image_np
