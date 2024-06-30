import cv2
import numpy as np

def lucas_kanade (im1, im2, N, sigma=1, avg_ders=True, create_of_mask=True):
    start_time = cv2.getTickCount()

    if avg_ders:
        Ix_1, Iy_1 = gaussderiv(im1, sigma=sigma)
        Ix_2, Iy_2 = gaussderiv(im2, sigma=sigma)
        Ix = (Ix_1 + Ix_2) / 2
        Iy = (Iy_1 + Iy_2) / 2
    else:
        Ix, Iy = gaussderiv(im1, sigma=sigma)
    It = gausssmooth(im2 - im1, sigma=sigma)

    kernel = np.ones((N, N), np.float32)
    Ix2 = cv2.filter2D(Ix**2, ddepth=-1, kernel=kernel)
    Iy2 = cv2.filter2D(Iy**2, ddepth=-1, kernel=kernel)
    Ixy = cv2.filter2D(Ix * Iy, ddepth=-1, kernel=kernel)
    Ixt = cv2.filter2D(Ix * It, ddepth=-1, kernel=kernel)
    Iyt = cv2.filter2D(Iy * It, ddepth=-1, kernel=kernel)

    cov_mtx_det = Ix2*Iy2 - Ixy**2
    cov_mtx_tr = Ix2 + Iy2
    det_mask = cov_mtx_det > 1e-15
    U = - (Iy2 * Ixt - Ixy * Iyt)
    V = -(Ix2 * Iyt - Ixy * Ixt)

    U = np.where(det_mask, U / cov_mtx_det, 0)
    V = np.where(det_mask, V / cov_mtx_det, 0)
    
    if create_of_mask:
        if avg_ders:
            Ix2_1 = cv2.filter2D(Ix_1**2, ddepth=-1, kernel=kernel)
            Iy2_1 = cv2.filter2D(Iy_1**2, ddepth=-1, kernel=kernel)
            Ixy_1 = cv2.filter2D(Ix_1 * Iy_1, ddepth=-1, kernel=kernel)
            det1 = Ix2_1 * Iy2_1 - Ixy_1**2
            tr1 = Ix2_1  + Iy2_1

            corner_response = det1 - 0.05 * tr1**2
        else:
            corner_response = cov_mtx_det - 0.05 * cov_mtx_tr**2
        of_mask = abs(corner_response) > 1e-6

        U[of_mask == False] = 0
        V[of_mask == False] = 0

    end_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
    print(f'Elapsed time: {end_time:.2f} ms, Masking {create_of_mask}')
    return U, V

def lk_pyramid(im1, im2, num_multi_levels=1):
    start_time = cv2.getTickCount()
    # create Gaussian pyramid
    G1 = [im1]
    G2 = [im2]
    for _ in range(4):
        dst_size = (G1[-1].shape[1] // 2, G1[-1].shape[0] // 2)
        im1_pyr = gausssmooth(G1[-1], sigma=1)
        im1_pyr = cv2.resize(im1_pyr, dst_size, interpolation=cv2.INTER_CUBIC)
        G1.append(im1_pyr)
        im2_pyr = gausssmooth(G2[-1], sigma=1)
        im2_pyr = cv2.resize(im2_pyr, dst_size, interpolation=cv2.INTER_CUBIC)
        G2.append(im2_pyr)
        # print(im1_pyr.shape, im2_pyr.shape)
    
    # reverse order of G1, G2
    G1 = G1[::-1]
    G2 = G2[::-1]

    U, V = lucas_kanade(im1_pyr, im2_pyr, 5)
    for _, (im1_pyr, im2_pyr) in enumerate(zip(G1[1:], G2[1:])):
        U = cv2.resize(U, (im1_pyr.shape[1], im1_pyr.shape[0]), interpolation=cv2.INTER_CUBIC)
        V = cv2.resize(V, (im1_pyr.shape[1], im1_pyr.shape[0]), interpolation=cv2.INTER_CUBIC)

        for _ in range(num_multi_levels):
            flow = np.array([U.copy(), V.copy()]).transpose(1, 2, 0)
            h, w = flow.shape[:2]
            flow = -flow
            flow[:,:,0] += np.arange(w)
            flow[:,:,1] += np.arange(h)[:,np.newaxis]
            im2_pyr_warped = cv2.remap(im2_pyr, flow, None, cv2.INTER_LINEAR)

            # compute residual optical flow
            U_warped, V_warped = lucas_kanade(im1_pyr, im2_pyr_warped, 5)
            U = U + U_warped
            V = V + V_warped
        
    end_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
    print(f'Elapsed time (pyramid): {end_time:.2f} ms')
    return U, V
