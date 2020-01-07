import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import label
import numpy as np
import pandas as pd
import os
from fbprophet import Prophet


def prophet_predict(ts, nahead, freq):
    date_seq = pd.date_range('2010-01-01 00:00:00', periods=len(ts), freq=freq)
    ts_df = pd.DataFrame([list(date_seq), list(ts)], index=['ds', 'y']).T
    m = Prophet().fit(ts_df)
    mkdf = m.make_future_dataframe(nahead)
    pred_df = m.predict(mkdf)['yhat']
    preds = pred_df[-nahead:]
    return m, np.array(preds)


def make_ts(cloudcoms):
    xts, yts = [], []
    for i in cloudcoms:
        for j in i:
            xts.append(j[0, 0])
            yts.append(j[0, 1])
    return xts


def cluster_image(img, k=4, showimage=False):
    km = KMeans(k).fit(img.reshape(-1, 1))
    lab = km.labels_.reshape(img.shape)
    centroids = km.cluster_centers_
    new = np.multiply((lab == np.argmax(centroids)), 1)
    if showimage is True:
        plt.figure()
        plt.title("Kmeans(k = " + str(k) + ")")
        io.imshow(new, cmap="gray")

    return new


def find_COM(x, y, original_img):
    xc, yc, sm = 0, 0, 0
    for i in range(len(x)):
        sm += original_img[x[i], y[i]]
        xc += (original_img[x[i], y[i]] * x[i])
        yc += (original_img[x[i], y[i]] * y[i])

    return xc / sm, yc / sm


def cloud_COM(original_img, labelled_img, n_cloud, n_comp):
    cloud_sizes = []
    for i in range(n_comp):
        cloud_sizes.append(np.count_nonzero(labelled_img == i))
    sorted_ind = np.argsort(cloud_sizes)
    sorted_ind = sorted_ind[::-1]

    ans = []
    for i in range(1, n_cloud + 1):
        x, y = np.nonzero(labelled_img == sorted_ind[i])
        xc, yc = find_COM(x, y, original_img)
        ans.append((yc, xc))
    ans = np.array(ans)
    ans.sort(0)
    return ans


# Get the path of images
images_path = []
for root, dirs, files in os.walk("F:\\PycharmProjects\\CloudMovement"):
    for name in files:
        if name.endswith(".tif"):
            images_path.append(str(root + '\\' + name))

# Read Images
imgs = io.imread_collection(images_path)

cloudcoms = []
for i in range(45):
    a = np.array(imgs[i])

    # Remove non clouds by clustering
    cloud_only = cluster_image(a, showimage=False)

    # Identify different clouds using connected components
    labelled_img, n_comp = label(cloud_only, structure=np.ones((3, 3)))

    # Mention no. of clouds to select and find there COM
    cord = cloud_COM(a, labelled_img, 1, n_comp)
    cloudcoms.append(cord)
    print(cord)

    # plt.figure()
    # plt.imshow(a, 'gray')
    # plt.scatter(cord[:, 0], cord[:, 1])
# plt.show()
