import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import numpy as np
from skimage.io import imread
from skimage.transform import rescale
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics.pairwise import euclidean_distances

from PIL import Image

name = input("Hvad er navnet på din fil? ")
def jpg_to_png (name: str):
    image_raw = imread(name)
    image_width = 300
    image = rescale(image_raw, image_width/image_raw.shape[0], mode='reflect', channel_axis=2, anti_aliasing=True)
    shape = image.shape
    image_pixels_lab = rgb2lab(image).reshape(-1,3)

    def cluster_assignments(X, Y):
        return np.argmin(euclidean_distances(X,Y), axis=1)

    K = 2
    centers =np.array([image_pixels_lab.mean(0) + (i/10) for i in range(K)])
    y_kmeans = cluster_assignments(image_pixels_lab, centers)
    last_centers = []
    antal = 0
    # repeat estimation a number of times (could do something smarter, like comparing if clusters change)
    while not np.array_equal(centers,last_centers):
        last_centers = centers.copy()
        # assign each point to the closest center
        y_kmeans = cluster_assignments(image_pixels_lab, centers)
        # move the centers to the mean of their assigned points (if any)
        for i, c in enumerate(centers):
            points = image_pixels_lab[y_kmeans == i]
            if len(points):
                centers[i] = points.mean(0)
        antal +=1

    k = np.argmax(np.bincount(y_kmeans))  

    rgb = lab2rgb(image_pixels_lab.reshape(shape))         
    pil_img = Image.fromarray((rgb * 255).round().astype(np.uint8)).convert('RGBA')

    alpha = np.where(y_kmeans.reshape(shape[:2]) == k, 0, 255).astype(np.uint8)
    pil_img.putalpha(Image.fromarray(alpha))
    for i in range(len(name)):
        if name[i] ==".":
            new_name=name[0:i:1]
    pil_img.save(new_name+'.png')                   

    print(f"Din fil er nu gemt som {new_name}.png")

jpg_to_png(name)