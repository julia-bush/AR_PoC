import numpy as np
import ezdxf
from cv2 import cv2


def load_camera_calib(calib_file):
    with np.load(calib_file) as jcam:
        mtx, dist, _, _ = [jcam[i] for i in ("mtx", "dist", "rvecs", "tvecs")]
    return mtx, dist


def load_dxf_modelspace(file):
    try:
        doc = ezdxf.readfile(file)
    except IOError:
        raise IOError(f"Not a DXF file or a generic I/O error.")
    except ezdxf.DXFStructureError:
        raise ValueError(f"Invalid or corrupted DXF file.")
    return doc.modelspace()


def plot_dxf_model(msp, img, dist, mtx, rvecs, tvecs, colour=(0, 0, 255), thickness=2):
    if img.ndim == 2 and len(colour) > 1:
        raise ValueError(f"Grayscale image. Need a single-value colour, got: {colour}")

    for edge_entity in msp.query("LINE"):
        _render_3d_line(
            img=img,
            start=edge_entity.dxf.start,
            end=edge_entity.dxf.end,
            rvecs=rvecs,
            tvecs=tvecs,
            mtx=mtx,
            dist=dist,
            colour=colour,
            thickness=thickness,
        )


def _render_3d_line(
    img, start, end, rvecs, tvecs, mtx, dist, colour=(0, 0, 255), thickness=2
):
    # get line start and end points
    objpts = np.float32([[start], [end]]).reshape(-1, 3)

    # project these to image plane
    imgpts, jac = cv2.projectPoints(objpts, rvecs, tvecs, mtx, dist)
    imgpts = imgpts.reshape(-1, 2)

    # render the line between projected start and end points
    cv2.line(
        img=img,
        pt1=tuple(imgpts[0]),
        pt2=tuple(imgpts[1]),
        color=colour,
        thickness=thickness,
    )