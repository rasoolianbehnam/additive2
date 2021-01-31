# pylint: disable=missing-docstring, invalid-name, c-extension-no-member
# pylint: disable=trailing-whitespace, unnecessary-lambda
# %%
import os
import cv2  # type: ignore
import igraph as gp  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from itertools import combinations
from typing import List, Tuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import convolve2d  # type: ignore
from scipy.spatial.distance import cdist, pdist  # type: ignore
from scipy.spatial.qhull import ConvexHull  # type: ignore
from scipy.stats import ttest_1samp  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from additive2.plotEllipsoid import Ellipsoid, EllipsoidTool

Point3d = Tuple[int, int, int]
CubeCorners = Tuple[Point3d, Point3d, Point3d, Point3d]


def dfe(file):
    d, f = os.path.split(file)
    f, e = os.path.splitext(f)

    return d, f, e


def get_largest_contour(mask):
    contours, = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)

    return max(contours, key=lambda x: len(x))


def draw_largest_contour(mask, fill=-1):
    max_contour = get_largest_contour(mask)
    imt = np.zeros_like(mask)
    cv2.drawContours(imt, [max_contour], -1, 1, fill)

    return imt.astype('uint8')


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    m = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, m, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


def get_rotated_rect(img, is_image=True):
    if is_image:
        if isinstance(img, str):
            img = cv2.imread(img, 0)
        contours, = cv2.findContours(img.astype('uint8'), cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=lambda x: len(x))

        return cv2.minAreaRect(max_contour)

    return cv2.minAreaRect(img)


def remove_outliers(x, n_classes=2):
    # x = np.array(x).reshape(-1, 1)
    kmeans = KMeans(n_classes)
    classes = kmeans.fit_predict(x)
    c = max(list(range(n_classes)), key=lambda val: np.sum(classes == val))

    return x[classes == c], classes, c


def min_max_scale(x,
                  new_min=0,
                  new_max=255,
                  dtype='float32',
                  return_stats=False):
    mn, mx = np.min(x), np.max(x)
    res = (x - mn) / (mx - mn + 1e-4)
    res = (res * (new_max - new_min) + new_min).astype(dtype)

    if return_stats:
        return res, mn, mx

    return res


def get_circle_mask(img: np.ndarray, erode_kernel: np.ndarray):
    """to get a mask for the middle
    circle in super-high-res images"""
    w, h, = img.shape
    img_floodfill = np.zeros((w + 2, h + 2), dtype='uint8')
    scaled_img = min_max_scale(img, dtype='uint8')
    cv2.floodFill(scaled_img, img_floodfill, (0, 0), 1)

    if erode_kernel is not None:
        out = cv2.dilate(img_floodfill, erode_kernel)
    else:
        out = img_floodfill

    return 1 - out[1:-1, 1:-1]


def make_mean_kernel(n):
    out = np.ones((n, n)) / (n * n)

    return out


def find_pores_2(img, mask, kernel_size, pore_area_ratio):
    if kernel_size % 2 != 1:
        raise ValueError("kernel size should be odd.")
    w, h, = img.shape
    img_area = w * h
    scaled_img = (min_max_scale(img, dtype='uint8'))
    smoothed_img = cv2.GaussianBlur(scaled_img, (3, 3), 1).astype('uint32')

    pore_img_ = get_pore_image(kernel_size, smoothed_img, mask)
    dilated_pore_img = cv2.dilate(pore_img_, np.ones((1, 1)))

    pore_img = get_good_contours(dilated_pore_img, img_area, pore_area_ratio,
                                 pore_img_)

    return pore_img


def get_pore_image(kernel_size, smoothed_img, mask):
    kernel = make_mean_kernel(kernel_size)
    means = convolve2d(smoothed_img, kernel, mode='valid')
    means_sq = convolve2d(smoothed_img ** 2, kernel, mode='valid')
    std = np.sqrt(means_sq - means ** 2)
    k2 = kernel_size // 2
    assert smoothed_img[k2:-k2, k2:-k2].shape == means.shape
    pore_img_ = ((smoothed_img[k2:-k2, k2:-k2] <
                  (means - 2.1 * std)) * mask[k2:-k2, k2:-k2]).astype('uint8')

    return pore_img_


def get_good_contours(dilated_pore_img, img_area, pore_area_ratio, pore_img_):
    _, contours, = cv2.findContours(dilated_pore_img, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    good_contours = [
        cnt for cnt in contours if len(cnt) / img_area > pore_area_ratio
    ]
    pore_img = np.zeros_like(pore_img_)
    pore_img = cv2.drawContours(pore_img, good_contours, -1, 255, -1)

    return pore_img


def draw_pores(pores_img, mask=None, ax=None):
    if ax is None:
        plt.figure(figsize=(15, 15))
        ax = plt.axes(projection='3d')
    assert len(pores_img.shape) == 3

    if mask is not None:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
        assert len(contours) == 1
        # let's reshape it to 2d
        contour = contours[0].reshape(-1, 2)

        for zs in range(len(pores_img)):
            ax.scatter3D(contour[:, 0].reshape(-1),
                         contour[:, 1].reshape(-1),
                         zs=zs,
                         s=1,
                         alpha=.01,
                         color='cyan')

    x, y, z = np.where(pores_img)
    ax.scatter(z, y, x, s=1)


def get_clusters(pore_img: np.ndarray):
    eligible_dists, (ix, iy), xyz = get_eligible_dists(pore_img, are_pts=False)
    # the way I get pores is that I get the derivative of eligible points,
    # if two pixels are adjacent, only the first one will be
    # set to 1 and the rest will be zero
    # this makes use of the fact that np.where
    # lists pixels that are adjacent to each other.
    eligible_dists_appended = np.hstack([[0], eligible_dists])  #
    clusters_ = np.cumsum((eligible_dists - eligible_dists_appended[:-1]) == 1)
    clusters = clusters_ * eligible_dists
    clusters2d = np.zeros((len(xyz),) * 2, dtype='int')
    clusters2d[(ix, iy)] = clusters

    return clusters2d, xyz


def get_eligible_dists(pore_img: np.ndarray, are_pts: bool):
    if are_pts:
        xyz = pore_img
    else:
        xyz = np.array(np.where(pore_img)).T
    n = len(xyz)
    distances_ = pdist(xyz)
    # concatenating n zeros at the beginning to account for n points themselves
    distances = np.hstack([np.zeros(n), distances_])
    # only pick pixel that are close that $\sqrt{2}$
    eligible_dists = (distances < np.sqrt(3 + 1e-8)) * 1
    # getting distnaces x and y indices
    ix_, iy_ = np.triu_indices(n, k=1)
    # adding indiceses of n points themselves to the beginning of ix and iy
    ix = np.concatenate([np.arange(n), ix_])
    iy = np.concatenate([np.arange(n), iy_])

    return eligible_dists, (ix, iy), xyz


def get_pores(pores_img: np.ndarray, are_pts):
    """
    :param pores_img: a 2d binary image representing pores or the
        coordinates corresponding to pores
    :param are_pts: A flag indicating whether
        pores_img is an image or coordinates
    :return: a tuple (x, y) where y is the index of
        non-zero pixels in pores_img and x is
        the cluster of the pore to which the pore belongs
    """
    eligible_dists, (ix, iy), xyz = get_eligible_dists(pores_img, are_pts)
    eligible_dists_i = np.where(eligible_dists)
    n_vertices = max(np.max(ix), np.max(iy)) + 1
    g = gp.Graph()
    g.add_vertices(n_vertices)
    edges = zip(ix[eligible_dists_i], iy[eligible_dists_i])
    g.add_edges(edges)
    clusters_ = g.clusters()
    clusters = [i for i, _1 in enumerate(clusters_) for _2 in _1]

    return clusters, xyz, clusters_


def find_matching_pores(p1: np.ndarray, p2: np.ndarray):
    """
    :param p1: a 1d array representing the coordinates corresponding to
        pores in image 1
    :param p2: a 1d array representing the coordinates corresponding to
    pores in image 2
    :return: two 1d arrays: p1_p2_map and p2_p1_map. p1_p2_map[i] refers
        to a coordinate index in image 2 where
        coordinate with index i refers to. The opposite is true for p2_p1_map.
    """
    dist = cdist(p1, p2)
    p1_matches, p2_matches = np.where(dist == 0)
    p1_p2_map = np.zeros(len(p1), dtype='int') - 1
    p2_p1_map = np.zeros(len(p2), dtype='int') - 1
    p1_p2_map[p1_matches] = p2_matches
    p2_p1_map[p2_matches] = p1_matches

    return p1_p2_map, p2_p1_map


def find_matching_clusters(p1_p2_map, cl1, cl2):
    cl12 = [(cl1[i], cl2[j]) for i, j in enumerate(p1_p2_map) if j > -1]

    return pd.Series(cl12).drop_duplicates().values


def get_pore_size_diff(cl_map, cl1, cl2, name1='a', name2='b'):
    res_ = []

    for i, j in cl_map:
        x, y = len(cl1[i]), len(cl2[j])
        res_.append([x, y, np.round(abs(x - y) / min(x, y), 2)])
    res = pd.DataFrame(res_, columns=[name1, name2, 'rel. error'])

    return res


pore_featuers = {}


def feature(name: str = None):
    def feature_(fun):
        pore_featuers[name or fun.__name__] = fun

        return fun

    return feature_


class PoreFeatures:
    et = EllipsoidTool()

    def __init__(self, pts_in: np.ndarray, expand: bool = True):
        """
        Parameters
        ----------
        pts_in : a 2d array of points/coordinates for the pores
        expand : Whether to add 1 to each combination of dimensions
            to make pore 3d
        """
        assert len(pts_in.shape) == 2, "pts must be 2d"
        self.pts_in = pts_in

        if expand:
            n_dims = pts_in.shape[1]
            pts = [pts_in]

            for i in range(1, n_dims + 1):
                combs = combinations(range(n_dims), i)

                for comb in combs:
                    x = pts_in.copy()
                    x[:, comb] += 1
                    pts.append(x)
            self.pts = pd.DataFrame(np.concatenate(
                pts, axis=0)).drop_duplicates().values
        else:
            self.pts = pts_in
        self.convex_hull: ConvexHull = ConvexHull(self.pts)
        self.ellipsoid: Ellipsoid = self.et.getMinVolEllipse(self.pts)

    def __repr__(self):
        return f"PoreFeatures({repr(self.pts_in)})"

    @property  # type: ignore
    @feature("Convex Hull Area")
    def a_h(self):
        return self.convex_hull.area

    @property  # type: ignore
    @feature("Pore Volume")
    def v_p(self):
        return self.pts.shape[0]

    @property  # type: ignore
    @feature("Convex Hull Volume")
    def v_h(self):
        return self.convex_hull.volume

    @property  # type: ignore
    @feature("Ellipsoid Volume")
    def v_e(self):
        return self.et.getEllipsoidVolume(self.ellipsoid)

    @property  # type: ignore
    @feature("Ellipsoid Area")
    def a_e(self):
        a, b, c = self.ellipsoid.radii

        return 4 * np.pi * (((a * b) ** 1.6 + (a * c) ** 1.6 +
                             (b * c) ** 1.6) / 3) ** (1 / 1.6)

    @property  # type: ignore
    @feature("Ellipsoid Elongation")
    def elongation_e(self):
        return self.a_e ** 3 / self.v_e ** 2 / (36 * np.pi)

    @property  # type: ignore
    @feature
    def convexity(self):
        return self.v_p / self.v_e

    @property  # type: ignore
    @feature("Pore Sphericity")
    def sphericity(self):
        top = (6 * self.v_h) ** (2 / 3)
        bottom = self.a_h

        return (np.pi ** (1 / 3)) * top / bottom

    @property  # type: ignore
    @feature("Pore Elongation")
    def elongation(self):
        return self.a_h ** 3 / self.v_h ** 2 / (36 * np.pi)

    @property
    def all_features(self):
        out = {}

        for name, fun in pore_featuers.items():
            out[name] = fun(self)

        return out


def get_feature_pairs(cluster_map: List[Tuple[int, int]], pts1: np.ndarray,
                      cls1: np.ndarray, pts2: np.ndarray,
                      cls2: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    --------------
    cluster_map
    pts1 : each row must be a (2d or 3d) point
    cls1:  1d array the same # of rows as pts1 which assings a cluster 
        to each point
    pts2 : each row must be a (2d or 3d) point
    cls2:  1d array the same # of rows as 
        pts1 which assings a cluster to each point

    Returns
    ------------
    Each dataframe the same size as input points and clusters where 
    each row lists the pore features
    """
    mid_features_ = []
    high_features_ = []

    for cl1, cl2 in cluster_map:
        pf1 = PoreFeatures(pts1[np.array(cls1) == cl1])
        mid_features_.append(pf1.all_features)
        pf2 = PoreFeatures(pts2[np.array(cls2) == cl2])
        high_features_.append(pf2.all_features)
    features1 = pd.DataFrame(mid_features_).reset_index().rename(
        columns={'index': 'cluster'})
    features2 = pd.DataFrame(high_features_).reset_index().rename(
        columns={'index': 'cluster'})

    return features1, features2


# %%


def ttest_pvalue(x: pd.Series) -> float:
    return ttest_1samp(x, 0).pvalue


def draw_voxels(pts: np.ndarray, ax, alpha: float = .1):
    for pt in pts:
        x_, y_, z_ = pt
        cube_definition = ((x_, y_, z_), (x_ + 1, y_, z_), (x_, y_ + 1, z_), (x_, y_, z_ + 1))
        plot_cube(cube_definition, ax=ax, alpha=alpha)


def plot_cube(cube_definition: CubeCorners, alpha=.1, ax=None):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    faces = Poly3DCollection(edges, linewidths=0, edgecolors='k')
    faces.set_facecolor((1, 0, 1, alpha))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    # ax.scatter(points[:,0], points[:,1], points[:,2], s=0)
