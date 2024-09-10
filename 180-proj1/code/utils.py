import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import resize
import cv2
from skimage.metrics import structural_similarity as ssim
import math
import os

def align(image, target, window_size=5, initial_displacement=(0, 0)):
    """
    Find the best displacement vector within the given window size that aligns the image with the target.
    
    Input:
    - image: The input image (to be shifted).
    - target: The reference image (the image to align to).
    - window_size: The size of the search window for the displacement.
    - initial_displacement: The center of search window (useful in pyramid).

    Output:
    - aligned_image: The image after alignment using the best displacement.
    - best_displacement: The displacement vector that has the highest metric score.
    """
    best_score = -float('inf')
    best_displacement = initial_displacement
    for dx in range(initial_displacement[0] - window_size, initial_displacement[0] + window_size + 1):
        for dy in range(initial_displacement[1] - window_size, initial_displacement[1] + window_size + 1):
            shifted_image = np.roll(image, (dy, dx), axis=(0, 1))
            score = ssim(shifted_image, target, data_range=1.0)
            if score > best_score:
                best_score = score
                best_displacement = (dx, dy)
    aligned_image = np.roll(image, best_displacement, axis=(1, 0))
    return aligned_image, best_displacement


def align_pyramid(image, target, window_size=15, levels=None, displacement=(0, 0)):
    """
    Recursively align the image with the targe starting from the coarsest level.
    
    Input:
    - image: The input image (to be shifted).
    - target: The reference image (the image to align to).
    - window_size: The size of the search window for the displacement.
    - levels: The number of levels in the pyramid based on image size.
    - displacement: The estimate of the displacement vector at current level.

    Output:
    - aligned_image: The image after alignment using the best displacement.
    - best_displacement: The displacement vector that has the highest metric score.
    """
    if levels is None:
        min_dim = min(image.shape[0], image.shape[1])
        levels = int(np.log2(min_dim // 200))

    if levels == 0:
        return align(image, target, window_size, displacement)

    downscaled_image = resize(image, (image.shape[0] // 2, image.shape[1] // 2))
    downscaled_target = resize(target, (target.shape[0] // 2, target.shape[1] // 2))
    _, refined_displacement = align_pyramid(downscaled_image, downscaled_target, window_size, levels - 1, displacement)
    refined_displacement = (refined_displacement[0] * 2, refined_displacement[1] * 2)
    reduced_window_size = math.ceil(window_size / (2 ** (levels)))
    
    return align(image, target, reduced_window_size, refined_displacement)

def auto_crop(image, pixel_threshold=25, count_threshold=0.75):
    """
    See details in webpage.
    """
    def find_edges(img):
        h, w, _ = img.shape
        def channel_edges(channel):
            row_diff = np.abs(channel[1:, :] - channel[0, :]) > pixel_threshold
            col_diff = np.abs(channel[:, 1:] - channel[:, 0].reshape(-1, 1)) > pixel_threshold
            top = np.argmax(np.sum(row_diff, axis=1) > count_threshold * w)
            left = np.argmax(np.sum(col_diff, axis=0) > count_threshold * h)
            bottom = h - np.argmax(np.sum(row_diff[::-1], axis=1) > count_threshold * w)
            right = w - np.argmax(np.sum(col_diff[:, ::-1], axis=0) > count_threshold * h)
            return top, bottom, left, right
        return [channel_edges(img[:,:,i]) for i in range(3)]

    edges = find_edges(image)
    top = max(e[0] for e in edges)
    bottom = min(e[1] for e in edges)
    left = max(e[2] for e in edges)
    right = min(e[3] for e in edges)

    return image[top:bottom, left:right]

def auto_contrast(image):
    """
    Automatically adjust the contrast using histogram equalization.
    """
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    
    return equalized_image

def auto_white_balance(image):
    """
    Apply automatic white balance to the image using the Gray World assumption.
    """
    result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    result[:, :, 1] = result[:, :, 1] - ((np.average(result[:, :, 1])- 128) * (result[:, :, 0] / 255))
    result[:, :, 2] = result[:, :, 2] - ((np.average(result[:, :, 2]) - 128) * (result[:, :, 0] / 255))
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result

def process_image(imname, type, crop = False, contrast = False, white_balance = False):
    """
    Reads an image file containing three color channels stacked vertically,
    aligns the green and red channels to the blue channel, and outputs an aligned RGB image.

    Input:
    - imname: The path of the input image file.
    - type: The type of the image ('jpg' or 'tif').

    Output:
    - Saves the aligned RGB image.
    """
    
    im = os.path.join(imname)
    im = skio.imread(im)
    im = sk.img_as_float(im)

    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]      
    g = im[height: 2*height]  
    r = im[2*height: 3*height]  

    if type == 'tif':
        ag, g_displacement = align_pyramid(g, b)  
        ar, r_displacement = align_pyramid(r, b)  
    elif type == 'jpg':
        ag, g_displacement = align(g, b, window_size=15)
        ar, r_displacement = align(r, b, window_size=15)

    print(f"Best displacement vectors: Green: {g_displacement}, Red: {r_displacement}")
    im_out = np.dstack([ar, ag, b])
    im_out = (im_out * 255).astype(np.uint8)

    output_dir = './output'
    if crop:
        im_out = auto_crop(im_out)
        output_dir += "_cropped"
    if contrast:
        im_out = auto_contrast(im_out)
        output_dir += "_contrasted"
    if white_balance:
        im_out = auto_white_balance(im_out)
        output_dir += "_white_balance"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = 'aligned_' + os.path.splitext(os.path.basename(imname))[0] + '.jpg'
    output_path = os.path.join(output_dir, output_filename)
    skio.imsave(output_path, im_out)

    return output_path