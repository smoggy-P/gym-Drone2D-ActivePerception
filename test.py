import numpy as np
from PIL import Image
from scipy import stats

def remove_background(image):
    """Convert the background of the image to transparency."""
    # Convert image to numpy array
    data = np.array(image)
    
    # Find the most frequent color in the image, which is assumed to be the background
    bg_color = stats.mode(data.reshape(-1, 4), axis=0)[0][0]

    # Create a mask where the background color is present
    mask = np.all(data == bg_color, axis=-1)

    # Change all pixels in the mask to transparent
    data[mask] = [0, 0, 0, 0]

    return Image.fromarray(data)

def stack_images(image_files, output_file):
    # Open images and convert them to RGBA (to support transparency)
    images = [Image.open(img).convert("RGBA") for img in image_files]

    # Remove the background from all but the first image
    images = [images[0]] + [remove_background(img) for img in images[1:]]

    # Result image
    result = images[0]

    # Number of images
    n_images = len(images)

    # Process each image
    for i in range(1, n_images):
        # Calculate alpha based on order
        # Alpha decreases as the image order increases
        alpha = 1 - (i / (n_images - 1))

        # Adjust the alpha channel of the image
        data = np.array(images[i])
        data[:,:,3] = (data[:,:,3] * alpha).astype(np.uint8)

        # Layer the adjusted image onto the result
        result = Image.alpha_composite(result, Image.fromarray(data))

    # Mirror the image (flip horizontally)
    img_mirror = result.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotate the image
    # The argument to rotate is the rotation angle in degrees
    img_rotated = img_mirror.rotate(90)  # replace 90 with your desired angle

    # Save the new image
    img_rotated.save('new_image.png')  # replace 'new_image.png' with your desired output file
    # # Save the result
    # result.save(output_file)

# List of image files
image_files = [str(i)+'.png' for i in range(1, 5)]

# Output file
output_file = 'stacked_shape.png'

# Stack images
stack_images(image_files[::-1], output_file)
