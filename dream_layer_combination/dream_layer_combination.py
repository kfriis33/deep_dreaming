import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math

# Image manipulation.
import PIL.Image
from scipy.ndimage.filters import gaussian_filter
import inception5h
import cv2

inception5h.maybe_download()


model = inception5h.Inception5h()


print("done")

gifimgs=[]

def load_image(filename):
    image = PIL.Image.open(filename)

    return np.float32(image)


def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')


def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.

    if True:
        # Convert the pixel-values to the range between 0.0 and 1.0
        image = np.clip(image/255.0, 0.0, 1.0)

        image = cv2.resize(image, (width, height))
        plt.imshow(image, interpolation='lanczos')
        plt.show()
        gifimgs.append(image)
    else:
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        # Convert pixels to bytes.
        image = image.astype(np.uint8)

        # Convert to a PIL-image and display it.
        display(PIL.Image.fromarray(image))


def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm



def plot_gradient(gradient):
    # Normalize the gradient so it is between 0.0 and 1.0
    gradient_normalized = normalize_image(gradient)

    # Plot the normalized gradient.
    plt.imshow(gradient_normalized, interpolation='bilinear')
    plt.show()


def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor

        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2.
        size = size[0:2]

    # The height and width is reversed in numpy vs. PIL.
    size = tuple(reversed(size))

    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)

    # Convert the pixels to 8-bit bytes.
    img = img.astype(np.uint8)

    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)

    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)

    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)

    return img_resized

def get_tile_size(num_pixels, tile_size=400):
    """
    num_pixels is the number of pixels in a dimension of the image.
    tile_size is the desired tile-size.
    """

    # How many times can we repeat a tile of the desired size.
    num_tiles = int(round(num_pixels / tile_size))

    # Ensure that there is at least 1 tile.
    num_tiles = max(1, num_tiles)

    # The actual tile-size.
    actual_tile_size = math.ceil(num_pixels / num_tiles)

    return actual_tile_size

def tiled_gradient(gradient, image, tile_size=400):
    # Allocate an array for the gradient of the entire image.
    grad = np.zeros_like(image)

    # Number of pixels for the x- and y-axes.
    x_max, y_max, _ = image.shape

    # Tile-size for the x-axis.
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    # 1/4 of the tile-size.
    x_tile_size4 = x_tile_size // 4

    # Tile-size for the y-axis.
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    # 1/4 of the tile-size
    y_tile_size4 = y_tile_size // 4

    # Random start-position for the tiles on the x-axis.
    # The random value is between -3/4 and -1/4 of the tile-size.
    # This is so the border-tiles are at least 1/4 of the tile-size,
    # otherwise the tiles may be too small which creates noisy gradients.
    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)
    while x_start < x_max:
        # End-position for the current tile.
        x_end = x_start + x_tile_size

        # Ensure the tile's start- and end-positions are valid.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        # Random start-position for the tiles on the y-axis.
        # The random value is between -3/4 and -1/4 of the tile-size.
        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            # End-position for the current tile.
            y_end = y_start + y_tile_size

            # Ensure the tile's start- and end-positions are valid.
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            # Get the image-tile.
            img_tile = image[x_start_lim:x_end_lim,
                             y_start_lim:y_end_lim, :]

            # Create a feed-dict with the image-tile.
            feed_dict = model.create_feed_dict(image=img_tile)

            # Use TensorFlow to calculate the gradient-value.
            g = session.run(gradient, feed_dict=feed_dict)

            # Normalize the gradient for the tile. This is
            # necessary because the tiles may have very different
            # values. Normalizing gives a more coherent gradient.
            g /= (np.std(g) + 1e-8)

            # Store the tile's gradient at the appropriate location.
            grad[x_start_lim:x_end_lim,
                 y_start_lim:y_end_lim, :] = g

            # Advance the start-position for the y-axis.
            y_start = y_end

        # Advance the start-position for the x-axis.
        x_start = x_end

    return grad


def optimize_image(layer_tensors, layer_weights, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=False):
  
    img = image.copy()
    original_img = img.copy()


 
    gradients = []
    for layer_tensor in layer_tensors:
        gradients.append(model.get_gradient(layer_tensor))

    for i in range(num_iterations):
       
        grads = []
        for index in range(len(gradients)):
            gradient = gradients[index]
            grad = tiled_gradient(gradient=gradient, image=original_img, tile_size=tile_size)
            sigma = (i * 4.0) / num_iterations + 0.5
            grad_smooth1 = gaussian_filter(grad, sigma=sigma)
            grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
            grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
            grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)
            step_size_scaled = step_size / (np.std(grad) + 1e-8)
            weighted_scaled_step_size = step_size_scaled*layer_weights[index]
            img += grad * weighted_scaled_step_size

    return img

def recursive_optimize(layer_tensors, layer_weights, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=10, step_size=3.0,
                       tile_size=400):

    if num_repeats>0:
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

        # Downscale the image.
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)
    
        img_result = recursive_optimize(layer_tensors=layer_tensors, layer_weights=layer_weights,
                                        image=img_downscaled,
                                        num_repeats=num_repeats-1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)

        # Upscale the resulting image back to its original size.
        img_upscaled = resize_image(image=img_result, size=image.shape)

        # Blend the original and processed images.
        image = blend * image + (1.0 - blend) * img_upscaled

    print("Recursive level:", num_repeats)

    # Process the image using the DeepDream algorithm.
    img_result = optimize_image(layer_tensors=layer_tensors, layer_weights=layer_weights,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size)

    return img_result



session = tf.InteractiveSession(graph=model.graph)
image = load_image(filename='images/bowen.jpg')

height = image.shape[0]
width= image.shape[1]
image = image[:,:,:3]

print("plotting 1")
plot_image(image)

layer_tensors = [model.layer_tensors[2], model.layer_tensors[10]]
img_result = recursive_optimize(layer_tensors=layer_tensors, layer_weights=[0.7,0.3], image=image,
             num_iterations=10, step_size=3.0, rescale_factor=0.7,
             num_repeats=6, blend=0.2)
print("final")
plot_image(img_result)

