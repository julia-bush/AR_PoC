from skopt import gp_minimize
from cv2 import cv2
import numpy as np
import cv2
import utils

# Playground script to test the idea of computing some form of
# loss/difference/distance between an edge map (e.g. canny)
# and the projected edges of a 3d model

# The first 10 frames are located using PNP (chessboard corner matching algo)
# to get a high-fidelity initialisation. Thereafter all is done using the edge
# maps, searching for a pose similar to the previous pose, which minimises the
# loss of how well the projection overlaps the canny edges.


def run(video_file, calib_file, dxf_file, output_video="video/result.avi"):
    output = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (640, 480)
    )
    msp = utils.load_dxf_modelspace(dxf_file)
    cap = cv2.VideoCapture(video_file)
    mtx, dist = utils.load_camera_calib(calib_file)
    rot, trans = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 10.0])
    frame_number = 0
    while cap.isOpened():
        frame_number += 1
        ret, frame = cap.read()

        if not ret:
            continue

        if frame_number < 10:
            rot, trans = _exact_pose(_convert_grayscale(frame), mtx, dist)
        else:
            rot, trans = _estimate_next_pose(frame, dist, msp, mtx, rot, trans)

        if cv2.waitKey(40) & 0xFF == ord("q"):
            break

        # Overlay the projected model and plot frame
        utils.plot_dxf_model(msp, frame, dist, mtx, rot, trans, colour=(0, 255, 0))
        cv2.imshow("frame", frame)

        output.write(frame)

    output.release()
    cap.release()


def _estimate_next_pose(frame, dist, msp, mtx, prev_rot, prev_trans):
    """Use the new frame, camera details, and the previous pose, to
    estimate the new pose (the one in the current `frame`)
    """

    def loss(pose):
        """Given a pose, return a positive real indicating how well it matches
        the current frame.

        NB: this function uses closures, it 'packages in' `frame`, `msp`, `mtx` `dist`,
            because the global optimiser we use (gp_minimise) requires a function of a
            single variable (`pose`) to be minimised.
        """
        return _evaluate_pose(
            dist, frame, msp, mtx, np.array(pose[:3]), np.array(pose[3:])
        )[0]

    # Only search in the vicinity of the previous pose
    start_pose = np.vstack([prev_rot, prev_trans]).ravel()
    tolerance = np.array([0.03, 0.03, 0.03, 0.4, 0.4, 0.4])
    pose_min = start_pose - tolerance
    pose_max = start_pose + tolerance

    # Global optimiser, takes about 1min per image in single-cpu
    res = gp_minimize(loss, [(pose_min[i], pose_max[i]) for i in range(6)], n_jobs=-1)

    optimal_pose = np.array(res.x)
    optimal_rotation = optimal_pose[:3].reshape(3, 1)
    optimal_translation = optimal_pose[3:].reshape(3, 1)
    opt_loss = res.fun

    print(f"Optimised pose loss = {opt_loss}")

    return optimal_rotation, optimal_translation


def _evaluate_pose(dist, frame, msp, mtx, rotation_vec, translation_vec):
    image_edges = _extract_edges_image(_convert_grayscale(frame))
    model_plot = np.zeros_like(image_edges)
    utils.plot_dxf_model(
        msp,
        model_plot,
        dist,
        mtx,
        rotation_vec,
        translation_vec,
        colour=(255,),
        thickness=1,
    )
    distance_transform = cv2.distanceTransform(
        src=~image_edges, distanceType=cv2.DIST_L2, maskSize=3
    )
    distance_transform = _normalise_img(distance_transform)
    model_plot = _normalise_img(model_plot)
    # The distance transform says 'how far is this pixel from an edge', and we want small
    # distances for the model plot pixels, so we need (1 - distance_transform)
    # We multiply the matrices element-wise, so we only add the distance_loss for the
    # model edge pixels. And we divide by the number of edge pixels, to get a mean.
    edge_to_plot_distance_loss = (
        model_plot * distance_transform
    ).sum() / model_plot.sum()
    return edge_to_plot_distance_loss, distance_transform, model_plot, image_edges


def _normalise_img(img):
    """Replaces the given image with a new image of the same shape, but linearly
    scaled such that all points are between 0 and 1
    """
    img = cv2.normalize(
        src=img,
        dst=img,
        alpha=0,
        beta=1.0,
        norm_type=cv2.NORM_MINMAX,
    )
    return img


def _convert_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _extract_edges_image(gray):
    """Return a binary cv2 image defining the detected edges.

    Currently using 'canny', might be replaced with something
    more advanced.
    """
    return cv2.Canny(gray, 100, 200)


def _exact_pose(gray, mtx, dist):
    """Return a pair (rotation, translation) vectors describing
    what we think is the pose of the 3d model.

    For the chessboard, this uses corner location and PNP matching
    to get a high-accuracy pose.
    """

    ret, corners = cv2.findChessboardCorners(gray, (5, 8), None)

    if ret is True:
        # refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objp = np.zeros((5 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:5, 0:8].T.reshape(-1, 2)
        # Find the rotation and translation vectors.
        _, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
            objp, corners2, mtx, dist
        )

        return rotation_vec, translation_vec
    else:
        # If no frame is found, guess the pose
        return np.array([0.1, 0.1, 0.1]), np.array([0.0, 0.0, 10.0])


if __name__ == "__main__":
    run(
        video_file="video/chess2.avi",
        calib_file="Jcam.npz",
        dxf_file="CAD/chess_small.dxf",
        output_video="video/result_tracking_by_minimising_model_edge_loss_1_min_per_frame.avi",
    )
