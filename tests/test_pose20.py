import unittest
import cv2
import numpy as np
from core.face_analyzer import FaceAnalyzer, HEAD_POSE_IDX


class PoseTest(unittest.TestCase):
    def test_projected_neutral_and_each_axis(self):
        analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
        analyzer.pose_convention = 'camera-v2'
        w, h = 640, 480
        camera = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], np.float64)
        for axis, degrees in ((0, 0), (0, 15), (1, 25), (2, 20)):
            vector = np.zeros(3); vector[axis] = np.radians(degrees)
            rotation = cv2.Rodrigues(vector)[0] @ np.diag([1., -1., -1.])
            rvec = cv2.Rodrigues(rotation)[0]
            points, _ = cv2.projectPoints(analyzer.MODEL_POINTS_3D, rvec,
                                          np.array([0., 0., 1800.]), camera, np.zeros(4))
            pts = np.zeros((478, 3)); pts[HEAD_POSE_IDX, :2] = points.reshape(-1, 2)
            yaw, pitch, roll = analyzer._calc_head_pose(pts, w, h)
            expected = np.zeros(3); expected[axis] = degrees
            np.testing.assert_allclose([pitch, yaw, roll], expected, atol=1e-3)


if __name__ == '__main__': unittest.main()
