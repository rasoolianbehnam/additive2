import joblib
import numpy as np
import unittest

from additive2.utility import get_pores, find_matching_pores, find_matching_clusters, PoreFeatures, get_feature_pairs


class ClusteringTest(unittest.TestCase):
    pts1 = np.array([
        [0, 0], [0, 1], [2, 3], [3, 4]
    ])

    pts2 = np.array([
        [10, 15], [11, 15], [40, 30], [0, 0]
    ])

    def test_get_pores_01(self):
        cl1, pts1, cl1_ = get_pores(self.pts1, are_pts=True)
        self.assertTrue(np.allclose(cl1, [0, 0, 1, 1]))
        self.assertTrue(
            np.allclose(pts1, [[0, 0], [0, 1], [2, 3], [3, 4]])
        )

    def test_get_pores_02(self):
        cl2, pts2, cl2_ = get_pores(self.pts2, are_pts=True)
        self.assertTrue(np.allclose(cl2, [0, 0, 1, 2]))
        self.assertTrue(np.allclose(pts2, [[10, 15],
                                           [11, 15],
                                           [40, 30],
                                           [0, 0]]))

    def test_find_matching_pores_01(self):
        cl1, pts1, cl1_ = get_pores(self.pts1, are_pts=True)
        cl2, pts2, cl2_ = get_pores(self.pts2, are_pts=True)
        p1_p2_map, p2_p1_map = find_matching_pores(pts1, pts2)
        self.assertListEqual(list(p1_p2_map), [3, -1, -1, -1])

    def test_find_matching_clusters_01(self):
        cl1, pts1, cl1_ = get_pores(self.pts1, are_pts=True)
        cl2, pts2, cl2_ = get_pores(self.pts2, are_pts=True)
        p1_p2_map, p2_p1_map = find_matching_pores(pts1, pts2)
        res = find_matching_clusters(p1_p2_map, cl1, cl2)
        self.assertListEqual(list(res), [(0, 2)])

    def test_find_matching_clusters_02(self):
        cl1, pts1, cl1_ = get_pores(self.pts1, are_pts=True)
        cl2, pts2, cl2_ = get_pores(self.pts2, are_pts=True)
        p1_p2_map, p2_p1_map = find_matching_pores(pts1, pts2)
        tmp_ = find_matching_clusters(p2_p1_map, cl2, cl1)
        self.assertListEqual(list(tmp_), [(2, 0)])

    def test_pore_features_01(self):
        pts = np.array([[20, 156, 361],
                        [20, 157, 358],
                        [20, 157, 359],
                        [20, 157, 360],
                        [20, 157, 361],
                        [20, 157, 362],
                        [20, 158, 358],
                        [20, 158, 359],
                        [20, 158, 362],
                        [20, 158, 363],
                        [20, 159, 358],
                        [20, 159, 363],
                        [20, 160, 358],
                        [20, 161, 357],
                        [20, 161, 358]])
        pf = PoreFeatures(pts)
        self.assertAlmostEqual(pf.a_h, 82.50, 2)
        self.assertAlmostEqual(pf.v_h, 30.50, 2)
        self.assertEqual(pf.v_p, 62)
        self.assertAlmostEqual(pf.convexity, .868, 3)
        self.assertAlmostEqual(pf.sphericity, .57, 2)
        self.assertAlmostEqual(pf.elongation, 5.34, 2)

    def test_get_feature_pairs01(self):
        cl_mid_high_map, pts_mid, clusters_mid, pts_high, clusters_high, mid_features, high_features = \
            joblib.load('../../data/Additive/test/test_get_feature_pairs01.pkl')
        f1, f2 = get_feature_pairs(cl_mid_high_map, pts_mid, clusters_mid, pts_high, clusters_high)
        self.assertTrue(np.allclose(f1.values, mid_features.values.round(3), atol=1e-3))
        self.assertTrue(np.allclose(f2.values, high_features.values.round(3), atol=1e-3))



if __name__ == '__main__':
    unittest.main(exit=True, argv=['first_is_empty'])  # in a script
