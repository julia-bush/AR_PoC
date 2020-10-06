import numpy as np
import cv2
import sys
import ezdxf

# finds the pose of a 5x8 chessboard from 3D-2D point correspondences
# and renders lines from a .dxf file in the same pose

# video output to save
output = cv2.VideoWriter(
    "video/result_corner_tracking_dxf.avi", cv2.VideoWriter_fourcc(*"XVID"), 20.0, (640, 480)
)

# video input to use
cap = cv2.VideoCapture("video/chess2.avi")

# Load previously saved camera calibration data
with np.load("Jcam.npz") as X:
    mtx, dist, _, _ = [X[i] for i in ("mtx", "dist", "rvecs", "tvecs")]

# load the .dxf file
try:
    doc = ezdxf.readfile("CAD/chess_3D.dxf")
except IOError:
    print(f"Not a DXF file or a generic I/O error.")
    sys.exit(1)
except ezdxf.DXFStructureError:
    print(f"Invalid or corrupted DXF file.")
    sys.exit(2)
# .dxf modelspace
msp = doc.modelspace()


def render_line(img, entity, rvecs, tvecs, mtx, dist):
    # get line start and end points
    objpts = np.float32([[entity.dxf.start], [entity.dxf.end]]).reshape(-1, 3)

    # project these to image plane
    imgpts, jac = cv2.projectPoints(objpts, rvecs, tvecs, mtx, dist)
    imgpts = imgpts.reshape(-1, 2)

    # render the line between projected start and end points
    img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 2)

    return img


# read video and render the lines from the .dxf on top of the chessboard
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((5 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:5, 0:8].T.reshape(-1, 2)

    ret, corners = cv2.findChessboardCorners(gray, (5, 8), None)

    if ret == True:
        # refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # entity query for all LINE entities in modelspace
        for e in msp.query("LINE"):
            render_line(frame, e, rvecs, tvecs, mtx, dist)

        cv2.imshow("frame", frame)
        output.write(frame)

    if cv2.waitKey(40) & 0xFF == ord("q"):
        break

output.release()
cap.release()
cv2.destroyAllWindows()
