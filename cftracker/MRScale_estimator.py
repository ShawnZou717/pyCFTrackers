import numpy as np
import math
import cv2


def fft2(x):
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)

def ifft2(x):
    return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)

def subpixel_peak(p):
    delta=0.5*(p[2]-p[0])/(2*p[1]-p[2]-p[0])
    if not np.isfinite(delta):
        delta=0
    return delta

def cos_window(sz):
    """
    width, height = sz
    j = np.arange(0, width)
    i = np.arange(0, height)
    J, I = np.meshgrid(j, i)
    cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
    """

    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
    return cos_window


def generate_image():
    '''
    This function can generate 2 image of 2 eclipse with descending pixel value, different size and shifted center.
    '''
    w = 10**2
    h = 10**2

    frame = np.zeros((h, w, 3))
    # draw circle
    a0 = 25
    b0 = 15
    center1 = (w//2, h//2)
    for i in range(h):
        for j in range(w):
            if (i-center1[1])**2/b0**2+(j-center1[0])**2/a0**2 <= 1:
                value = (i - center1[1]) ** 2 / b0 ** 2 + (j - center1[0]) ** 2 / a0 ** 2
                frame[i, j, :] = np.floor(255-np.array([255, 255, 255])*value/2)
            else:
                frame[i, j, :] = (0, 0, 0)
    cv2.imwrite(".zzz.jpg", frame)

    later_frame = np.zeros((h, w, 3))
    # draw eclipse
    a = 15
    b = 10
    center2 = (w // 2 - 5, h // 2 + 15)
    alpha = math.pi/180*(15)
    for i in range(h):
        for j in range(w):
            term1 = ((center2[0] - j) * math.cos(alpha) + (i - center2[1]) * math.sin(alpha)) ** 2 / a ** 2
            term2 = ((j - center2[0]) * math.sin(alpha) + (i - center2[1]) * math.cos(alpha)) ** 2 / b ** 2
            if term1 + term2 <= 1:
                value = term1 + term2
                later_frame[i, j, :] = np.floor(255-np.array([255, 255, 255])*value/2)
            else:
                later_frame[i, j, :] = (0, 0, 0)
    cv2.imwrite("zzz1.jpg", later_frame)
    return frame, later_frame


def generate_following_matrix(size, chan_num = 1):
    h, w = size[0:2]
    a0 = w/2
    b0 = h/2
    frame = np.zeros((h+2*int(h/2), w+2*int(w/2), chan_num))
    center1 = (w, h)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if (i - center1[1]) ** 2 / b0 ** 2 + (j - center1[0]) ** 2 / a0 ** 2 <= 1:
                frame[i, j, :] = [255]*chan_num
    return frame


def scale_array(x, sf, FILL_VALUE = None, INTERP = "NEAREST", extend_flag = False):
    res = list()
    if extend_flag:
        index = [i / sf for i in range(int(np.ceil(sf*len(x))))]
    else:
        index = [i / sf for i in range(len(x))]  # y = sf*x, so x = y/sf

    for idx in index:
        idx_lf = np.floor(idx)
        idx_rt = np.ceil(idx)
        if int(idx_lf) >= len(x) or int(idx_rt) >= len(x):
            if FILL_VALUE is None:
                res.append(x[-1])
            elif isinstance(FILL_VALUE, int) or isinstance(FILL_VALUE, float):
                res.append(FILL_VALUE)
            else:
                raise Exception("FULL_VALUE must be int or float.")
        elif idx_lf == idx_rt:
            res.append(x[int(idx_rt)])
        else:
            if INTERP == "LINEAR":
                x0 = idx_lf
                y0 = x[int(idx_lf)]
                x1 = idx_rt
                y1 = x[int(idx_rt)]
                y = y0 + (idx - x0) * ((y1 - y0)/(x1 - x0))
                res.append(y)
            elif INTERP == "NEAREST":
                lf = idx - idx_lf
                rt = idx_rt - idx
                if lf < rt:
                    res.append(x[int(idx_lf)])
                else:
                    res.append(x[int(idx_rt)])
            else:
                raise Exception("INTERP must be \"LINEAR\" or \"NEAREST\".")

    return np.array(res)


def MR_estimate(last_frame, cur_frame, use_channel_weight = False, channel_weight = None,
                channel_weight_linearPolar = None, channel_weight_logPolar = None,log_flag = False, img_flag = False,
                erode_times=1):
    if last_frame.shape[2] == 1 or cur_frame.shape[2] == 1:
        raise Exception("Not allowed deep dimension 1.")
    if not np.all(last_frame.shape == cur_frame.shape) and not last_frame.shape[2] == channel_weight.shape[0]:
        raise Exception("img size should be the same.")
    if use_channel_weight and \
            (channel_weight is None or channel_weight_linearPolar is None or channel_weight_logPolar is None):
        raise Exception("channel_weight or channel_weight_linearPolar or channel_weight_logPolar not given.")
    if use_channel_weight and (not np.all(channel_weight.shape == channel_weight_linearPolar.shape) or \
            not np.all(channel_weight_linearPolar.shape == channel_weight_logPolar.shape)):
        raise Exception("shapes of channel_weight, channel_weight_linearPolar, channel_weight_logPolar doesn't coincide.")

    last_frame = np.double(last_frame)
    cur_frame = np.double(cur_frame)

    last_frame = 255*(last_frame-np.min(last_frame))/(np.max(last_frame)-np.min(last_frame))
    cur_frame = 255*(cur_frame-np.min(cur_frame))/(np.max(cur_frame)-np.min(cur_frame))

    last_frame_gray = np.sum(last_frame, axis=2)/last_frame.shape[2]
    cur_frame_gray = np.sum(cur_frame, axis=2)/cur_frame.shape[2]

    h, w, _ = last_frame.shape
    center1 = (w/2, h/2)

    sf_following = 1

    following_matrix = []
    _phi = [i*(360/h) for i in range(h)]
    for idx, phi_val in enumerate(_phi):
        if phi_val <= 45 and phi_val >= 0:
            _phi[idx] = phi_val + 360

    phi_v1 = math.atan(h / w)*180/math.pi
    phi_v2 = math.atan(-h / w) * 180 / math.pi + 180
    phi_v3 = math.atan(h / w) * 180 / math.pi + 180
    phi_v4 = math.atan(-h / w) * 180 / math.pi + 360
    phi_v5 = phi_v4 + 90

    for phi_val in _phi:
        if (phi_val > phi_v1 and phi_val <= phi_v2) or (phi_val > phi_v3 and phi_val <= phi_v4):
            if phi_val == 90 or phi_val == 270:
                sin_phi = 1
            else:
                sin_phi = math.sin(phi_val*math.pi/180)
            radius = abs((h/2)/sin_phi)
        elif (phi_val > phi_v2 and phi_val <= phi_v3) or (phi_val > phi_v4 and phi_val <= phi_v5):
            if phi_val == 180 or phi_val == 360:
                cos_phi = 1
            else:
                cos_phi = math.cos(phi_val*math.pi/180)
            radius = abs((w/2)/cos_phi)
        following_matrix.append(radius)
        
    following_matrix = np.array(following_matrix)
    sf_following = 1.
    following_matrix = following_matrix*sf_following + h/2*(1. - sf_following)

    window = cos_window((last_frame.shape[1], last_frame.shape[0]))

    exp_round = 1
    dx = float("Inf")
    dy = float("Inf")
    dx_res, dy_res = 0, 0
    dphi = float("Inf")
    scale_deviation = float("Inf")

    max_exp_round = 5
    dx_min = 0.01
    dy_min = 0.01
    dphi_min = (360/cur_frame.shape[0])/10
    scale_deviation_min = 0.01
    stop_iter_condition = exp_round <= max_exp_round and \
                          (dx > dx_min or dy > dy_min or dphi > dphi_min or scale_deviation > scale_deviation_min)

    while stop_iter_condition:
        #### center estimator
        H = fft2(last_frame * window[:,:,None])
        # H = fft2(last_frame)
        if use_channel_weight:
            response = np.real(ifft2(fft2(cur_frame * window[:,:,None]) * np.conj(H)))
            response = np.sum(response * channel_weight[None, None, :], axis=2)
        else:
            response = np.real(ifft2(np.sum(fft2(cur_frame * window[:,:,None]) * np.conj(H), axis=2)))
        s_index = np.unravel_index(np.argmax(response, axis=None), response.shape)
        y_idx, x_idx = s_index[:2]

        curr = (x_idx, y_idx)
        v_neighbors = response[curr[1], [(curr[0] - 1) % response.shape[1], (curr[0]) % response.shape[1],
                                (curr[0] + 1) % response.shape[1]]]
        h_neighbors = response[[(curr[1] - 1) % response.shape[0], (curr[1]) % response.shape[0],
                                (curr[1] + 1) % response.shape[0]], curr[0]]
        x_idx = curr[0] + subpixel_peak(v_neighbors)
        y_idx = curr[1] + subpixel_peak(h_neighbors)

        if x_idx + 1 > response.shape[1] / 2:
            x_idx = x_idx - response.shape[1]

        if y_idx + 1 > response.shape[0] / 2:
            y_idx = y_idx - response.shape[0]

        # if dy > dy_min:
        if True:
            dy = y_idx
            dy_res += dy
        # if dx > dx_min:
        if True:
            dx = x_idx
            dx_res += dx

        ######  rotation estimator
        xc, yc = center1[:2]
        r1 = math.hypot(xc, yc)
        r2 = math.hypot(w - xc - 1, yc)
        r3 = math.hypot(xc, h - yc - 1)
        r4 = math.hypot(w - xc - 1, h - yc - 1)
        maxR1 = max(r1, r2, r3, r4)

        last_frame_polar = cv2.linearPolar(last_frame, center1, maxR1, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        center2 = (center1[0] + int(dx), center1[1] + int(dy))
        xc, yc = center2[:2]
        r1 = math.hypot(xc, yc)
        r2 = math.hypot(w - xc - 1, yc)
        r3 = math.hypot(xc, h - yc - 1)
        r4 = math.hypot(w - xc - 1, h - yc - 1)
        maxR2 = max(r1,r2,r3,r4)

        cur_frame_polar = cv2.linearPolar(cur_frame, center2, maxR2, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        H = fft2(last_frame_polar * window[:,:,None])
        # H = fft2(last_frame_polar)
        if use_channel_weight:
            response = np.real(ifft2(fft2(cur_frame_polar * window[:, :, None]) * np.conj(H)))
            response = np.sum(response * channel_weight_linearPolar[None, None, :], axis=2)
        else:
            response = np.real(ifft2(np.sum(fft2(cur_frame_polar * window[:, :, None]) * np.conj(H), axis=2)))
        s_index = np.unravel_index(np.argmax(response, axis=None), response.shape)
        y_idx, x_idx = s_index[:2]

        curr = (x_idx, y_idx)
        v_neighbors = response[curr[1], [(curr[0] - 1) % response.shape[1], (curr[0]) % response.shape[1],
                                         (curr[0] + 1) % response.shape[1]]]
        h_neighbors = response[[(curr[1] - 1) % response.shape[0], (curr[1]) % response.shape[0],
                                (curr[1] + 1) % response.shape[0]], curr[0]]
        x_idx = curr[0] + subpixel_peak(v_neighbors)
        y_idx = curr[1] + subpixel_peak(h_neighbors)

        if x_idx + 1 > response.shape[1] / 2:
            x_idx = x_idx - response.shape[1]

        if y_idx + 1 > response.shape[0] / 2:
            y_idx = y_idx - response.shape[0]

        dphi = y_idx * (360 / response.shape[0])

        ######  scale estimator
        m1 = max(h, w) / math.log(maxR1)
        m2 = max(h, w) / math.log(maxR2)
        m = max(m1, m2)
        last_frame_logpolar = cv2.logPolar(last_frame, center1, m, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        last_frame_polar = np.roll(last_frame_polar, int(y_idx), axis=0)
        last_frame_logpolar = np.roll(last_frame_logpolar, int(y_idx), axis=0)
        # following_matrix_linearPolar = np.roll(following_matrix_linearPolar,
        #                                        int(dphi/(360/following_matrix_linearPolar.shape[0])), axis=0)
        following_matrix = np.roll(following_matrix,
                                               int(y_idx), axis=0)

        cur_frame_logpolar = cv2.logPolar(cur_frame, center2, m, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)


        H = np.fft.fft(last_frame_logpolar * np.hanning(last_frame_logpolar.shape[1])[None, :, None], axis=1)
        # H = np.fft.fft(last_frame_logpolar, axis=1)
        if use_channel_weight:
            response = np.real(np.fft.ifft(np.fft.fft(
                cur_frame_logpolar * np.hanning(last_frame_logpolar.shape[1])[None, :, None], axis=1)
                                                  * np.conj(H), axis=1))
            response = np.sum(response * channel_weight_logPolar[None, None, :], axis=2)
        else:
            response = np.real(np.fft.ifft(np.sum(np.fft.fft(
                cur_frame_logpolar * np.hanning(last_frame_logpolar.shape[1])[None, :, None], axis=1)
                                                  * np.conj(H), axis=2), axis=1))

        s_index = np.argmax(response, axis=1)

        last_frame_polar_copy = last_frame_polar.copy()
        # if scale_deviation > scale_deviation_min:
        if True:
            scale = []
            for idx, s_ in enumerate(s_index):

                result = response[idx, [(s_ - 1) % response.shape[1], s_ % response.shape[1], (s_ + 1) % response.shape[1]]]
                # result = result[None, :]
                W_p = []
                for s_val in [(s_ - 1) % response.shape[1], s_ % response.shape[1], (s_ + 1) % response.shape[1]]:
                    if s_val + 1 > response.shape[1] / 2:
                        s_val -= response.shape[1]
                    W_p.append([np.exp(s_val/m)**2, np.exp(s_val/m), 1])
                W_p = np.array(W_p)
                W_inv = np.linalg.inv(W_p)
                vvv = list(np.dot(W_inv, result))
                aaa = vvv[0]
                bbb = vvv[1]
                sf = -bbb/2/aaa
                scale.append(sf)
                following_matrix[idx] = following_matrix[idx]*sf
                # image scaling
                for _idx in range(last_frame_polar_copy.shape[2]):
                    last_frame_polar_copy[idx, :, _idx] = scale_array(last_frame_polar_copy[idx, :, _idx], sf, FILL_VALUE=0, INTERP="LINEAR")

        scale = np.array(scale)
        scale_deviation = np.sqrt(np.mean(np.square(scale - 1)))
        if log_flag:
            print("%d round dx, dy is : %f, %f" % (exp_round, dx, dy))
            print("%d round dphi is : %f (du)" % (exp_round, dphi))
            print("%d round scale deviation is : %f" % (exp_round, scale_deviation))
            print()

        last_frame_new = cv2.linearPolar(np.abs(last_frame_polar_copy), center2, maxR2, cv2.WARP_INVERSE_MAP)

        if img_flag:
            last_frame_new = np.round(np.clip(last_frame_new, 0, 255))

        last_frame = last_frame_new
        center1 = center2
        exp_round += 1

        stop_iter_condition = exp_round <= max_exp_round and \
                              (dx > dx_min or dy > dy_min or dphi > dphi_min or scale_deviation > scale_deviation_min)

    phi = np.array([i*360/following_matrix.shape[0] for i in range(following_matrix.shape[0])])
    x_cont = following_matrix * np.cos(phi*np.pi/180)
    y_cont = following_matrix * np.sin(phi*np.pi/180)
    w_res = max(x_cont) - min(x_cont)
    h_res = max(y_cont) - min(y_cont)
    return dx_res, dy_res, w_res, h_res


def MR_estimate_one_time(last_frame, last_frame_linear_polar, last_frame_log_polar,
                         cur_frame, cur_frame_linear_polar, cur_frame_log_polar,
                         use_channel_weight = False,log_flag = False,
                         channel_weight = None, channel_weight_linearPolar = None, channel_weight_logPolar = None):
    if last_frame.shape[2] == 1 or cur_frame.shape[2] == 1:
        raise Exception("Not allowed deep dimension 1.")
    if not np.all(last_frame.shape == cur_frame.shape) and not last_frame.shape[2] == channel_weight.shape[0]:
        raise Exception("img size should be the same.")
    if use_channel_weight and \
            (channel_weight is None or channel_weight_linearPolar is None or channel_weight_logPolar is None):
        raise Exception("channel_weight or channel_weight_linearPolar or channel_weight_logPolar not given.")
    if use_channel_weight and (not np.all(channel_weight.shape == channel_weight_linearPolar.shape) or \
            not np.all(channel_weight_linearPolar.shape == channel_weight_logPolar.shape)):
        raise Exception("shapes of channel_weight, channel_weight_linearPolar, channel_weight_logPolar doesn't coincide.")

    # last_frame = np.double(last_frame)
    # cur_frame = np.double(cur_frame)
    #
    # last_frame = 255*(last_frame-np.min(last_frame))/(np.max(last_frame)-np.min(last_frame))
    # cur_frame = 255*(cur_frame-np.min(cur_frame))/(np.max(cur_frame)-np.min(cur_frame))
    #
    # last_frame_gray = np.sum(last_frame, axis=2)/last_frame.shape[2]
    # cur_frame_gray = np.sum(cur_frame, axis=2)/cur_frame.shape[2]
    # # cv2.imwrite("/home/shawn/scripts_output_tmp/last_frame_gray.jpg", last_frame_gray)
    # # cv2.imwrite("/home/shawn/scripts_output_tmp/cur_frame_gray.jpg", cur_frame_gray)

    h, w, _ = last_frame.shape
    center1 = (w/2, h/2)

    sf_following = 1
    # following_matrix = generate_following_matrix((sf_following*h, sf_following*w))
    # following_matrix = np.zeros(h) + h/2

    following_matrix = []
    _phi = [i*(360/h) for i in range(h)]
    for idx, phi_val in enumerate(_phi):
        if phi_val <= 45 and phi_val >= 0:
            _phi[idx] = phi_val + 360

    phi_v1 = math.atan(h / w)*180/math.pi
    phi_v2 = math.atan(-h / w) * 180 / math.pi + 180
    phi_v3 = math.atan(h / w) * 180 / math.pi + 180
    phi_v4 = math.atan(-h / w) * 180 / math.pi + 360
    phi_v5 = phi_v4 + 90
    for phi_val in _phi:
        if (phi_val > phi_v1 and phi_val <= phi_v2) or (phi_val > phi_v3 and phi_val <= phi_v4):
            if phi_val == 90 or phi_val == 270:
                sin_phi = 1
            else:
                sin_phi = math.sin(phi_val*math.pi/180)
            radius = abs((h/2)/sin_phi)
        elif (phi_val > phi_v2 and phi_val <= phi_v3) or (phi_val > phi_v4 and phi_val <= phi_v5):
            if phi_val == 180 or phi_val == 360:
                cos_phi = 1
            else:
                cos_phi = math.cos(phi_val*math.pi/180)
            radius = abs((w/2)/cos_phi)
        following_matrix.append(radius)
    following_matrix = np.array(following_matrix)
    sf_following = 0.
    following_matrix = following_matrix*sf_following + h/2*(1. - sf_following)
    # following_matrix = following_matrix * 0.9 + h / 2 * 0.1

    # cv2.imwrite("/home/shawn/scripts_output_tmp/following_matrix.jpg", following_matrix)
    # center_following = (following_matrix.shape[1]//2, following_matrix.shape[0]//2)
    # center_following_const = (following_matrix.shape[1]//2, following_matrix.shape[0]//2)

    # window = cos_window((last_frame.shape[1], last_frame.shape[0]))

    exp_round = 1
    dx = float("Inf")
    dy = float("Inf")
    dx_res, dy_res = 0, 0
    dphi = float("Inf")
    scale_deviation = float("Inf")

    max_exp_round = 1
    dx_min = 0.01
    dy_min = 0.01
    dphi_min = (360/cur_frame.shape[0])/10
    scale_deviation_min = 0.01
    stop_iter_condition = exp_round <= max_exp_round

    while stop_iter_condition:
        #### center estimator
        H = last_frame
        # H = fft2(last_frame)
        if use_channel_weight:
            response = np.real(ifft2(fft2(cur_frame) * np.conj(H)))
            response = np.sum(response * channel_weight[None, None, :], axis=2)
        else:
            response = np.real(ifft2(np.sum(fft2(cur_frame) * np.conj(H), axis=2)))
        s_index = np.unravel_index(np.argmax(response, axis=None), response.shape)
        y_idx, x_idx = s_index[:2]

        curr = (x_idx, y_idx)
        v_neighbors = response[curr[1], [(curr[0] - 1) % response.shape[1], (curr[0]) % response.shape[1],
                                (curr[0] + 1) % response.shape[1]]]
        h_neighbors = response[[(curr[1] - 1) % response.shape[0], (curr[1]) % response.shape[0],
                                (curr[1] + 1) % response.shape[0]], curr[0]]
        x_idx = curr[0] + subpixel_peak(v_neighbors)
        y_idx = curr[1] + subpixel_peak(h_neighbors)

        if x_idx + 1 > response.shape[1] / 2:
            x_idx = x_idx - response.shape[1]

        if y_idx + 1 > response.shape[0] / 2:
            y_idx = y_idx - response.shape[0]

        # if dy > dy_min:
        if True:
            dy = y_idx
            dy_res += dy
        # if dx > dx_min:
        if True:
            dx = x_idx
            dx_res += dx


        H = last_frame_linear_polar
        # H = fft2(last_frame_polar)
        if use_channel_weight:
            response = np.real(ifft2(fft2(cur_frame_linear_polar) * np.conj(H)))
            response = np.sum(response * channel_weight_linearPolar[None, None, :], axis=2)
        else:
            response = np.real(ifft2(np.sum(fft2(cur_frame_linear_polar) * np.conj(H), axis=2)))
        s_index = np.unravel_index(np.argmax(response, axis=None), response.shape)
        y_idx, x_idx = s_index[:2]

        curr = (x_idx, y_idx)
        v_neighbors = response[curr[1], [(curr[0] - 1) % response.shape[1], (curr[0]) % response.shape[1],
                                         (curr[0] + 1) % response.shape[1]]]
        h_neighbors = response[[(curr[1] - 1) % response.shape[0], (curr[1]) % response.shape[0],
                                (curr[1] + 1) % response.shape[0]], curr[0]]
        x_idx = curr[0] + subpixel_peak(v_neighbors)
        y_idx = curr[1] + subpixel_peak(h_neighbors)

        if x_idx + 1 > response.shape[1] / 2:
            x_idx = x_idx - response.shape[1]

        if y_idx + 1 > response.shape[0] / 2:
            y_idx = y_idx - response.shape[0]

        # if dphi > dphi_min:
        #     dphi = y_idx*(360/response.shape[0])
        # else:
        #     y_idx = 0
        dphi = y_idx * (360 / response.shape[0])

        # ######  scale estimator
        # m1 = max(h, w) / math.log(maxR1)
        # m2 = max(h, w) / math.log(maxR2)
        # m = max(m1, m2)
        # last_frame_logpolar = cv2.logPolar(last_frame, center1, m, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        last_frame_linear_polar = np.roll(last_frame_linear_polar, int(y_idx), axis=0)
        last_frame_log_polar = np.roll(last_frame_log_polar, int(y_idx), axis=0)
        # following_matrix_linearPolar = np.roll(following_matrix_linearPolar,
        #                                        int(dphi/(360/following_matrix_linearPolar.shape[0])), axis=0)
        following_matrix = np.roll(following_matrix,
                                               int(y_idx), axis=0)

        # cur_frame_logpolar = cv2.logPolar(cur_frame, center2, m, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        # cv2.imwrite("/home/shawn/scripts_output_tmp/last_frame_logpolar_%d.jpg" % exp_round, last_frame_logpolar)
        # cv2.imwrite("/home/shawn/scripts_output_tmp/cur_frame_logpolar_%d.jpg" % exp_round, cur_frame_logpolar)

        # H = last_frame_log_polar
        # # H = np.fft.fft(last_frame_logpolar, axis=1)
        # if use_channel_weight:
        #     response = np.real(np.fft.ifft(np.fft.fft(cur_frame_log_polar, axis=1) * np.conj(H), axis=1))
        #     response = np.sum(response * channel_weight_logPolar[None, None, :], axis=2)
        # else:
        #     response = np.real(np.fft.ifft(np.sum(np.fft.fft(cur_frame_log_polar, axis=1) * np.conj(H), axis=2), axis=1))

        H = last_frame_log_polar
        # H = np.fft.fft(last_frame_logpolar, axis=1)
        if use_channel_weight:
            response = np.real(ifft2(fft2(cur_frame_log_polar) * np.conj(H)))
            response = np.sum(response * channel_weight_logPolar[None, None, :], axis=2)
        else:
            response = np.real(
                ifft2(np.sum(fft2(cur_frame_log_polar) * np.conj(H), axis=2)))

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # plt.figure()
        # sns.heatmap(response)
        # plt.savefig("/home/shawn/scripts_output_tmp/heatmap_corr_%d.jpg" % exp_round)

        s_index = np.argmax(response, axis=1)

        # last_frame_polar_copy = last_frame_polar.copy()
        # if scale_deviation > scale_deviation_min:

        radius_f = np.sqrt((cur_frame_log_polar.shape[0]/2)**2+(cur_frame_log_polar.shape[1]/2)**2)
        m = max(cur_frame_log_polar.shape[:2])/np.log(radius_f)

        if True:
            scale = []
            for idx, s_ in enumerate(s_index):
                result = response[idx, [(s_ - 1) % response.shape[1], s_ % response.shape[1], (s_ + 1) % response.shape[1]]]
                # result = result[None, :]
                # W_p = []
                # for s_val in [(s_ - 1) % response.shape[1], s_ % response.shape[1], (s_ + 1) % response.shape[1]]:
                #     if s_val + 1 > response.shape[1] / 2:
                #         s_val -= response.shape[1]
                #     W_p.append([np.exp(s_val/m)**2, np.exp(s_val/m), 1])
                # W_p = np.array(W_p)
                # W_inv = np.linalg.inv(W_p)
                # vvv = list(np.dot(W_inv, result))
                # aaa = vvv[0]
                # bbb = vvv[1]
                # sf = -bbb/2/aaa
                delta = 0.5 * (result[2] - result[0]) / (2 * result[1] - result[2] - result[0])
                if not np.isfinite(delta):
                    delta = 0
                s_ = s_ + delta
                if s_ + 1 > response.shape[1] / 2:
                    s_ = s_ - response.shape[1]
                sf = np.exp(s_/m)
                scale.append(sf)

            scale = np.array(scale)
            # scale_mean = np.mean(scale)
            # if abs(scale_mean - 1) <= 0.1:
            #     stdev = np.sqrt(np.sum((scale-1)**2/scale.shape[0]))
            #     scale_smooth = scale.copy()
            #
            #     for index, sf in enumerate(scale):
            #         if abs(sf - 1) >= 1*stdev:
            #             scale_smooth[index] = np.nan
            #     reg = np.polyfit(list(range(int(scale_smooth.shape[0]))), scale, 5)
            #     scale_fit = np.polyval(reg, list(range(scale_smooth.shape[0])))
            #     scale_smooth[np.isnan(scale_smooth)] = scale_fit[np.isnan(scale_smooth)]
            # else:
            #     scale_smooth = np.zeros_like(scale)
            #     scale_smooth = scale_smooth + 1

            stdev = np.sqrt(np.sum((scale - 1) ** 2 / scale.shape[0]))
            scale_smooth = scale.copy()

            for index, sf in enumerate(scale):
                if abs(sf - 1) >= 1 * stdev:
                    scale_smooth[index] = np.nan
            reg = np.polyfit(list(range(int(scale_smooth.shape[0]))), scale, 5)
            scale_fit = np.polyval(reg, list(range(scale_smooth.shape[0])))
            scale_smooth[np.isnan(scale_smooth)] = scale_fit[np.isnan(scale_smooth)]

            for index, sf in enumerate(scale_smooth):
                following_matrix[index] = following_matrix[index] * sf

        scale_deviation = np.sqrt(np.mean(np.square(scale_smooth - 1)))
        if log_flag:
            print("%d round dx, dy is : %f, %f" % (exp_round, dx, dy))
            print("%d round dphi is : %f (du)" % (exp_round, dphi))
            print("%d round scale deviation is : %f" % (exp_round, scale_deviation))
            print()

        exp_round += 1
        stop_iter_condition = exp_round <= max_exp_round

    phi = np.array([i*360/following_matrix.shape[0] for i in range(following_matrix.shape[0])])
    x_cont = following_matrix * np.cos(phi*np.pi/180)
    y_cont = following_matrix * np.sin(phi*np.pi/180)
    w_res = max(x_cont) - min(x_cont)
    h_res = max(y_cont) - min(y_cont)

    # if w_res > 51 or h_res > 51:
    #     for sf in scale_smooth:
    #         print(sf)

    return dx_res, dy_res, w_res, h_res


def test_polar_corr():
    frame, later_frame = generate_image()
    # frame = cv2.imread("/home/shawn/scripts_output_tmp/zzz1.png")
    # later_frame = cv2.imread("/home/shawn/scripts_output_tmp/zzz2.png")
    #
    # frame = cv2.resize(frame, (50, 50))
    # later_frame = cv2.resize(later_frame, (50, 50))
    # cv2.imwrite("/home/shawn/scripts_output_tmp/zzz1_resized.png", frame)
    # cv2.imwrite("/home/shawn/scripts_output_tmp/zzz2_resized.png", later_frame)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # later_frame = cv2.cvtColor(later_frame, cv2.COLOR_BGR2GRAY)
    # frame = frame[:,:,None]
    # later_frame = later_frame[:,:,None]

    h,w,_ = frame.shape
    center1 = (w/2, h/2)
    # frame, later_frame = cv2.cvtColor(np.uint8(frame),cv2.COLOR_BGR2GRAY),cv2.cvtColor(np.uint8(later_frame),cv2.COLOR_BGR2GRAY)
    # frame, later_frame = frame[:,:,None], later_frame[:,:,None]

    window = cos_window((frame.shape[1], frame.shape[0]))
    # window = np.zeros_like(frame[:,:,0])+1

    cccc = 1
    while cccc <= 10:
        #### center estimator
        H = fft2(frame * window[:,:,None])
        response = np.real(ifft2(np.sum(fft2(later_frame * window[:,:,None]) * np.conj(H), axis=2)))
        s_index = np.unravel_index(np.argmax(response, axis=None), response.shape)
        y_idx, x_idx = s_index[:2]

        if x_idx + 1 > response.shape[1] / 2:
            x_idx = x_idx - response.shape[1]

        if y_idx + 1 > response.shape[0] / 2:
            y_idx = y_idx - response.shape[0]

        dy = y_idx
        dx = x_idx
        print("%d round dx, dy is : %f, %f"%(cccc, dx, dy))

        ######  rotation estimator
        xc, yc = center1[:2]
        r1 = math.hypot(xc, yc)
        r2 = math.hypot(w - xc - 1, yc)
        r3 = math.hypot(xc, h - yc - 1)
        r4 = math.hypot(w - xc - 1, h - yc - 1)
        maxR1 = max(r1, r2, r3, r4)

        # frame = np.roll(frame, int(dy), axis=0)
        # frame = np.roll(frame, int(dx), axis=1)
        center2 = (center1[0]+dx, center1[1]+dy)
        frame = np.roll(frame, int(dy), axis=0)
        frame = np.roll(frame, int(dx), axis=1)
        frame_polar = cv2.linearPolar(frame, center2, maxR1, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        cv2.imwrite("/home/shawn/scripts_output_tmp/zzz_polar.jpg", frame_polar)

        xc, yc = center2[:2]
        r1 = math.hypot(xc, yc)
        r2 = math.hypot(w - xc - 1, yc)
        r3 = math.hypot(xc, h - yc - 1)
        r4 = math.hypot(w - xc - 1, h - yc - 1)
        maxR2 = max(r1,r2,r3,r4)
        # m = max(h,w)/math.log(maxR)

        later_frame_polar = cv2.linearPolar(later_frame, center2, maxR2, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        cv2.imwrite("/home/shawn/scripts_output_tmp/zzz1_polar.jpg", later_frame_polar)

        H = fft2(frame_polar * window[:,:,None])
        response = np.real(ifft2(np.sum(fft2(later_frame_polar * window[:, :, None]) * np.conj(H), axis=2)))
        s_index = np.unravel_index(np.argmax(response, axis=None), response.shape)

        # H = np.fft.fft(frame_polar * np.hanning(frame_polar.shape[0])[:, None, None], axis=0)
        # response = np.real(np.fft.ifft(np.sum(
        #     np.fft.fft(later_frame_polar * np.hanning(frame_polar.shape[0])[:, None, None], axis=0) * np.conj(H),
        #     axis=2), axis=0))
        #
        # s_index = np.argmax(response,axis=0)
        # dphi_s = s_index*(360/response.shape[0])
        # for i in range(len(dphi_s)):
        #     if s_index[i] + 1 > response.shape[0] / 2:
        #         dphi_s[i] = dphi_s[i] - 180

        # import matplotlib.pyplot as plt
        # plt.plot(dphi_s)
        # plt.show()

        y_idx, x_idx = s_index[:2]

        if x_idx + 1 > response.shape[1] / 2:
            x_idx = x_idx - response.shape[1]

        if y_idx + 1 > response.shape[0] / 2:
            y_idx = y_idx - response.shape[0]

        dphi = y_idx*(360/response.shape[0])
        dr = x_idx
        print("%d round dphi is : %f (du)" % (cccc, dphi))


        ######  scale estimator
        m1 = max(h, w) / math.log(maxR1)
        m2 = max(h, w) / math.log(maxR2)
        m = max(m1, m2)
        frame_logpolar = cv2.logPolar(frame, center2, m, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        frame_logpolar = np.roll(frame_logpolar, int(y_idx), axis=0)
        frame_polar = np.roll(frame_polar, int(y_idx), axis=0)

        later_frame_logpolar = cv2.logPolar(later_frame, center2, m, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        H = np.fft.fft(frame_logpolar * np.hanning(frame_logpolar.shape[1])[None, :, None], axis=1)
        response = np.real(np.fft.ifft(np.sum(np.fft.fft(later_frame_logpolar * np.hanning(frame_logpolar.shape[1])[None, :, None], axis=1) * np.conj(H),
                                              axis=2), axis=1))
        s_index = np.argmax(response, axis=1)

        scale = []
        frame_polar_copy = frame_polar.copy()
        for idx, s_ in enumerate(s_index):
            if s_ + 1 > response.shape[1] / 2:
                s_ = s_ - response.shape[1]
            sf = math.exp(s_/m)
            scale.append(sf)
            frame_polar_copy[idx, :, 0] = scale_array(frame_polar_copy[idx, :, 0], sf, FILL_VALUE=None)
            frame_polar_copy[idx, :, 1] = scale_array(frame_polar_copy[idx, :, 1], sf, FILL_VALUE=None)
            frame_polar_copy[idx, :, 2] = scale_array(frame_polar_copy[idx, :, 2], sf, FILL_VALUE=None)
        scale = np.array(scale)
        scale_deviation = np.sqrt(np.mean(np.square(scale - 1)))
        print("%d round scale deviation is : %f" % (cccc, scale_deviation))
        print()

        frame_new = cv2.linearPolar(frame_polar_copy, center2, maxR2, cv2.WARP_INVERSE_MAP)
        frame_new = np.round(np.clip(frame_new, 0, 255))
        cv2.imwrite("/home/shawn/scripts_output_tmp/zzz_new%d.jpg"%cccc, frame_new)

        frame = frame_new
        center1 = center2
        cccc += 1

        import matplotlib.pyplot as plt
        x_data = np.arange(0, len(scale)) * 360 / len(scale)
        plt.plot(x_data, scale)
    plt.legend(["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"], loc=(360/380, 0.3))
    plt.xlim(0, 380)
    plt.xlabel("theta(du)")
    plt.ylabel("scale")
    plt.title("Scale at different round")
    plt.savefig("/home/shawn/scripts_output_tmp/scale_all.jpg")


    return


if __name__ == "__main__":
    frame = generate_following_matrix((500, 500))
    cv2.imshow("huiguiren", frame)
    cv2.waitKey(0)

    frame, later_frame = generate_image()
    MR_estimate(frame, later_frame, log_flag=True, img_flag=True)


