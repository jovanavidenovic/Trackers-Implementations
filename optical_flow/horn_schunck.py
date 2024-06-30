import cv2
import numpy as np
from lucas_kanade import lk_pyramid 

def horn_schunck (im1, im2, n_iters, lmbd, LK_init=False, check_conv=False):
    start_time = cv2.getTickCount()
    if LK_init:
        U, V = lk_pyramid(im1, im2)
    else:
        # initialize U, V to matrices of 0s
        U = np.zeros_like(im1)
        V = np.zeros_like(im1)

    # precompute Ix, Iy, It
    im1_smooth = gausssmooth(im1, sigma=1)
    im2_smooth = gausssmooth(im2, sigma=1)
    Ix_filter = cv2.flip(np.array([[-1, 1], [-1, 1]]) / 2, -1)
    Iy_filter = cv2.flip(np.array([[-1, -1], [1, 1]]) / 2, -1)
    It_filter = np.ones((2, 2)) / 4

    Ix = cv2.filter2D(im1_smooth, ddepth=-1, kernel=Ix_filter)
    Iy = cv2.filter2D(im1_smooth, ddepth=-1, kernel=Iy_filter)
    It = cv2.filter2D(im2_smooth - im1_smooth, ddepth=-1, kernel=It_filter) 

    smoothing_kernel = np.ones((3, 3), np.float32)
    det = cv2.filter2D(Ix**2, -1, smoothing_kernel) + cv2.filter2D(Iy**2, -1, smoothing_kernel) + lmbd

    neighbor_filter = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
    for iter_idx in range(n_iters):
        # second step - average U, V over the neighborhood
        U_avg_neigh = cv2.filter2D(U, ddepth=-1, kernel=neighbor_filter)
        V_avg_neigh = cv2.filter2D(V, ddepth=-1, kernel=neighbor_filter)

        # third step - precompute optical flow
        U_prev = U.copy()
        V_prev = V.copy()

        P = It + Ix * U_avg_neigh + Iy * V_avg_neigh
        U = U_avg_neigh - Ix * P / det
        V = V_avg_neigh - Iy * P / det    

        if check_conv:
            # check for convergence
            flow_mag = np.mean(np.sqrt(U**2 + V**2))
            prev_flow_mag = np.mean(np.sqrt(U_prev**2 + V_prev**2))
            if (abs(flow_mag - prev_flow_mag) / flow_mag) < 1e-3:
                print("Converged at iteration", iter_idx)
                break

    end_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
    print(f'Elapsed time: {end_time:.2f} ms, LK_init {LK_init}')
    return U, V 
