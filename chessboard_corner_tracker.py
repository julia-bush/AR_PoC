import numpy as np
import cv2

# finds the pose of a 5x8 chessboard from 3D-2D point correspondences
# and renders an axis or a cube in the same pose

# video output to save
output = cv2.VideoWriter(
    "video/result_corner_tracking.avi", cv2.VideoWriter_fourcc(*"XVID"), 20.0, (640, 480)
)

# video input to use
cap = cv2.VideoCapture("video/chess2.avi")

# Load previously saved camera calibration data
with np.load("Jcam.npz") as X:
    mtx, dist, _, _ = [X[i] for i in ("mtx", "dist", "rvecs", "tvecs")]

# 3D points for drawing the cube
axis = np.float32(
    [
        [0, 0, 0],
        [0, 3, 0],
        [3, 3, 0],
        [3, 0, 0],
        [0, 0, -3],
        [0, 3, -3],
        [3, 3, -3],
        [3, 0, -3],
    ]
)


# draw cube function
def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


# read video and render the cube on top of the chessboard
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((5 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:5, 0:8].T.reshape(-1, 2)

    ret, corners = cv2.findChessboardCorners(gray, (5, 8), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        frame = draw_cube(frame, corners2, imgpts)
        cv2.imshow("frame", frame)
        output.write(frame)

    if cv2.waitKey(40) & 0xFF == ord("q"):
        break

output.release()
cap.release()
cv2.destroyAllWindows()
