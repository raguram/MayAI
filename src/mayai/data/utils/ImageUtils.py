from matplotlib import pyplot as plt
import numpy as np
from statistics import mean


def show_images(images, titles=None, cols=10, figSize=(15, 15)):
    """
    Shows images with its labels. Expected PIL Image.
    """
    figure = plt.figure(figsize=figSize)
    num_of_images = len(images)
    rows = np.ceil(num_of_images / float(cols))
    for index in range(0, num_of_images):
        plt.subplot(rows, cols, index + 1)
        plt.axis('off')
        if titles is not None:
            plt.title(titles[index])
        plt.imshow(np.asarray(images[index]), cmap="gray")


def get_stats(images, name):
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]

    max_width = max(widths)
    max_height = max(heights)

    min_width = min(widths)
    min_height = min(heights)

    avg_width = mean(widths)
    avg_height = mean(heights)

    print(f"Stats of {name}\n")
    print(f"Number of images: {len(images)}")
    print(f"Min width, Max width, Min Height, Max height: {min_width}, {max_width}, {min_height}, {max_height}")
    print(f"Average width and height {avg_width}, {avg_height}")
