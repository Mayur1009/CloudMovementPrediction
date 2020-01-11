# Detect and select clouds, and draw bounding box around them

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from skimage.filters import threshold_multiotsu    # scikit-image >= 0.16.1
from skimage.morphology import closing, square

rcParams['image.cmap'] = 'gray'
rcParams['figure.max_open_warning'] = 50


# Read Images
def read_images(path='F:\\PycharmProjects\\CloudMovement'):
    import os
    imgs = []
    for root, dirs, files in os.walk(path):
        for names in files:
            if names.endswith('.tif'):
                imgs.append(io.imread(str(root + '\\' + names), as_gray=True))
    del os, root, dirs, files
    return imgs


# Create mask to show only clouds, using kmeans
def cloud_mask_kmeans(image, k=3):
    image = np.array(image)
    km = KMeans(k).fit(image.reshape(-1, 1))
    labels, centers = km.labels_, km.cluster_centers_
    cloud_center = np.argmax(centers)
    mask = ((labels == cloud_center) & 1).reshape(image.shape)
    mask = closing(mask, square(5))
    del km, labels, centers, cloud_center
    return np.array(mask)


# Create mask to show only clouds, using multiotsu thresholding
def cloud_mask_otsu(image):
    image = np.array(image)
    thresh = threshold_multiotsu(image, 4)
    max_thresh = max(thresh)
    mask = closing(image > max_thresh, square(10))
    del max_thresh, thresh
    return np.array(mask)


# Label masked clouds
def label_mask(mask):
    label_img, n_comp = label(mask, return_num=True)
    return label_img, n_comp


# Select Clouds with area greater than threshold
def select_clouds(label_img, threshold=5000):
    big_clouds = np.empty(label_img.shape)
    for region in regionprops(label_img):
        if region.area < threshold:
            for x, y in region.coords:
                big_clouds[x, y] = 0
        else:
            for x, y in region.coords:
                big_clouds[x, y] = 1
    del region
    return big_clouds


# Draw bounding box around selected clouds
def make_bbox(lab_img, img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    for region in regionprops(lab_img):
        minr, minc, maxr, maxc = region.bbox
        rect = Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)


# Test the functions
def test(img, thres):
    mask = cloud_mask_otsu(img)
    lab_img, n_comp = label_mask(mask)
    oc = select_clouds(lab_img, thres)
    lab_oc, n_selected = label_mask(oc)
    make_bbox(lab_oc, img)
    dic = {
        'mask': mask,
        'lab_img': lab_img,
        'n_comp': n_comp,
        'oc': oc,
        'lab_oc': lab_oc,
        'n_selected': n_selected
    }
    return dic


# Find Center of Mass for cloud cluster
def find_COM(x, y, original_img):
    xc, yc, sm = 0, 0, 0
    for i in range(len(x)):
        sm += original_img[x[i], y[i]]
        xc += (original_img[x[i], y[i]] * x[i])
        yc += (original_img[x[i], y[i]] * y[i])

    return xc / sm, yc / sm


# Returns Boundary Pixels of selected cloud cluster
def boundary(lab_oc, pixels_coords):
    boundary_pixels = []
    for x, y in pixels_coords:
        if x-1 > 0 and y-1 > 0 and x+1 < lab_oc.shape[0] and y+1 < lab_oc.shape[1]:
            nei = [[x - 1, y], [x + 1, y], [x, y + 1], [x, y - 1]]
            for nx, ny in nei:
                if lab_oc[nx, ny] == 0:
                    boundary_pixels.append((ny, nx))
    del x, y, nx, ny
    boundary_pixels = np.array(boundary_pixels)
    return boundary_pixels


# Same as boudary(), but slower, needs improvement
def boundary2(lab_oc, pixels_coords):
    boundary_pixels = []
    processed = np.empty_like(lab_oc)
    for x, y in pixels_coords:
        if processed[x, y] == 0 and (x-1 > 0 and y-1 > 0 and x+1 < lab_oc.shape[0] and y+1 < lab_oc.shape[1]):
            processed[x, y] = 1
            if lab_oc[x-2, y] == 1:
                processed[x-1, y] = 1
            if lab_oc[x+2, y] == 1:
                processed[x+1, y] = 1
            if lab_oc[x, y-2] == 1:
                processed[x, y-1] = 1
            if lab_oc[x, y+2] == 1:
                processed[x, y+1] = 1

            nei = [[x - 1, y], [x + 1, y], [x, y + 1], [x, y - 1]]
            for nx, ny in nei:
                if lab_oc[nx, ny] == 0:
                    boundary_pixels.append((ny, nx))
    del x, y, nx, ny, processed
    boundary_pixels = np.array(boundary_pixels)
    return boundary_pixels
