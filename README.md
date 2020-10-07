# AR_PoC

Work in progress.

This is a proof of concept study to explore the feasibility of development of an Augmented Reality (AR) application with 3D model-based tracking, with a view to eventual application in Structural Health Monitoring.

So far only a toy object has been used for this study, namely a black and white chessboard-like pattern printed on an A4 piece of paper. The high-contrast corners are easily identified in an image and have known 3D real-world coordinates.


## Results

The "Results" folder contains the "augmented reality" videos created from the unedited video file "video/chess2.avi" (which I filmed for this purpose with the camera moving around the "chessboard").


## Code

The .py scripts detailed below are executable. To run them yourself, you will need to copy all the .py files from this repository, Jcam.npz, as well as the "CAD" and "video" folders, to your working directory, and install the packages listed in the requirements.txt. Or you can just look at the results.

Here is a short description of what each script does, in order of increasing complexity:


### chessboard_corner_tracker.py

Results/result_corner_tracking.avi was recorded by running this script.

Identifies 2D (image) coordinates of the chessboard corners and computes correspondences of these to the 3D (real-world) chessboard corners coordinates. Hence estimates camera/object pose and renders a cube in the video stream matching the object pose. This creates the impression that the artificially added cube is placed on the chessboard.


### dxf_chessboard_corner_tracker.py

Results/result_corner_tracking_dxf.avi was recorded by running this script.

CAD/chess_3D.dxf was used to generate the geometry.

As above, except instead of a hard-coded cube, the rendered geometry is imported automatically from a .dxf CAD model file, which can be created and edited in AutoCAD among other 3D modelling software.


### dxf_edge_tracking_chessboard.py

Results/result_tracking_by_minimising_model_edge_loss_1_min_per_frame.avi was recorded by running this script. It took around 1 minute to process each frame (using a single core of 8-core i7-9700KF @ 3.6GHz, 32GB RAM).

CAD/chess_small.dxf was used to generate the geometry.

Here, only the first 10 frames of the video use the corner point correspondences (as the two scripts above) to ensure high-fidelity initialisation. Thereafter, the pose is estimated by minimising the distance between the object edges (as they appear in the 2D video frame image) and the line geometry (as imported from the .dxf 3D model file).

In a little more detail: an edge map is generated for each video frame using Canny edge detection. To determine the next pose, we search for a pose similar to the previous one, minimising the distance between the edges detected in the video frame and the 2D projection of the geometry from the .dxf 3D model file.

