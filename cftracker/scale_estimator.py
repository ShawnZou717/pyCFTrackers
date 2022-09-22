import numpy as np
import scipy
import cv2
from numpy.fft import fft, ifft
from scipy import signal
from lib.eco.fourier_tools import resize_dft
from .feature import extract_hog_feature
from lib.utils import cos_window
from lib.fft_tools import ifft2,fft2

class DSSTScaleEstimator:
    def __init__(self,target_sz,config):
        init_target_sz = np.array([target_sz[0],target_sz[1]])

        self.config=config
        num_scales = self.config.number_of_scales_filter
        scale_step = self.config.scale_step_filter
        scale_sigma = self.config.number_of_interp_scales * self.config.scale_sigma_factor

        scale_exp = np.arange(-np.floor(num_scales - 1)/2,
                              np.ceil(num_scales-1)/2+1,
                              dtype=np.float32) * self.config.number_of_interp_scales / num_scales
        scale_exp_shift = np.roll(scale_exp, (0, -int(np.floor((num_scales-1)/2))))

        interp_scale_exp = np.arange(-np.floor((self.config.number_of_interp_scales - 1) / 2),
                                     np.ceil((self.config.number_of_interp_scales - 1) / 2) + 1,
                                     dtype=np.float32)
        interp_scale_exp_shift = np.roll(interp_scale_exp, [0, -int(np.floor(self.config.number_of_interp_scales - 1) / 2)])

        self.scale_size_factors = scale_step ** scale_exp
        self.interp_scale_factors = scale_step ** interp_scale_exp_shift

        ys = np.exp(-0.5 * (scale_exp_shift ** 2) / (scale_sigma ** 2))
        self.yf = np.real(fft(ys))
        self.window = np.hanning(ys.shape[0]).T.astype(np.float32)
        # make sure the scale model is not to large, to save computation time


        self.num_scales = num_scales
        self.scale_step = scale_step

        if self.config.scale_model_factor ** 2 * np.prod(init_target_sz) > self.config.scale_model_max_area:
            scale_model_factor = np.sqrt(self.config.scale_model_max_area / np.prod(init_target_sz))
        else:
            scale_model_factor = self.config.scale_model_factor

        # set the scale model size
        self.scale_model_sz = np.maximum(np.floor(init_target_sz * scale_model_factor), np.array([8, 8]))
        self.max_scale_dim = self.config.s_num_compressed_dim == 'MAX'
        if self.max_scale_dim:
            self.s_num_compressed_dim = len(self.scale_size_factors)
        else:
            self.s_num_compressed_dim = self.config.s_num_compressed_dim



    def init(self,im,pos,base_target_sz,current_scale_factor):

        # self.scale_factors = np.array([1])
        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz)
        self.s_num = xs
        # compute projection basis
        if self.max_scale_dim:
            self.basis, _ = scipy.linalg.qr(self.s_num, mode='economic')
            scale_basis_den, _ = scipy.linalg.qr(xs, mode='economic')
        else:
            U, _, _ = np.linalg.svd(self.s_num)
            self.basis = U[:, :self.s_num_compressed_dim]
            V, _, _ = np.linalg.svd(xs)
            scale_basis_den = V[:, :self.s_num_compressed_dim]
        self.basis = self.basis.T
        # compute numerator
        feat_proj = self.basis.dot(self.s_num) * self.window
        sf_proj = np.fft.fft(feat_proj, axis=1)
        self.sf_num = self.yf * np.conj(sf_proj)

        # update denominator
        xs = scale_basis_den.T.dot(xs)*self.window
        xsf = fft(xs, axis=1)
        new_sf_den = np.sum((xsf * np.conj(xsf)), 0)
        self.sf_den = new_sf_den


    def update(self, im, pos, base_target_sz, current_scale_factor):
        base_target_sz=np.array([base_target_sz[0],base_target_sz[1]])
        # get scale filter features
        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz)

        # project
        xs = self.basis.dot(xs) * self.window

        # get scores
        xsf = np.fft.fft(xs, axis=1)
        scale_responsef = np.sum(self.sf_num * xsf, 0) / (self.sf_den + self.config.lamBda)
        interp_scale_response = np.real(ifft(resize_dft(scale_responsef, self.config.number_of_interp_scales)))
        recovered_scale_index = np.argmax(interp_scale_response)

        if self.config.do_poly_interp:
            # fit a quadratic polynomial to get a refined scale estimate
            id1 = (recovered_scale_index - 1) % self.config.number_of_interp_scales
            id2 = (recovered_scale_index + 1) % self.config.number_of_interp_scales
            poly_x = np.array([self.interp_scale_factors[id1], self.interp_scale_factors[recovered_scale_index],
                               self.interp_scale_factors[id2]])
            poly_y = np.array(
                [interp_scale_response[id1], interp_scale_response[recovered_scale_index], interp_scale_response[id2]])
            poly_A = np.array([[poly_x[0] ** 2, poly_x[0], 1],
                               [poly_x[1] ** 2, poly_x[1], 1],
                               [poly_x[2] ** 2, poly_x[2], 1]], dtype=np.float32)
            poly = np.linalg.inv(poly_A).dot(poly_y.T)
            scale_change_factor = - poly[1] / (2 * poly[0])
        else:
            scale_change_factor = self.interp_scale_factors[recovered_scale_index]


        current_scale_factor=current_scale_factor*scale_change_factor

        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz)
        self.s_num = (1 - self.config.scale_learning_rate) * self.s_num + self.config.scale_learning_rate * xs
        # compute projection basis
        if self.max_scale_dim:
            self.basis, _ = scipy.linalg.qr(self.s_num, mode='economic')
            scale_basis_den, _ = scipy.linalg.qr(xs, mode='economic')
        else:
            U, _, _ = np.linalg.svd(self.s_num)
            self.basis = U[:, :self.s_num_compressed_dim]
            V,_,_=np.linalg.svd(xs)
            scale_basis_den=V[:,:self.s_num_compressed_dim]
        self.basis = self.basis.T

        # compute numerator
        feat_proj = self.basis.dot(self.s_num) * self.window
        sf_proj = np.fft.fft(feat_proj, axis=1)
        self.sf_num = self.yf * np.conj(sf_proj)

        # update denominator
        xs = scale_basis_den.T.dot(xs)*self.window
        xsf = np.fft.fft(xs, axis=1)
        new_sf_den = np.sum((xsf * np.conj(xsf)), 0)
        self.sf_den = (1 - self.config.scale_learning_rate) * self.sf_den + self.config.scale_learning_rate * new_sf_den
        return current_scale_factor


    def _extract_scale_sample(self, im, pos, base_target_sz, scale_factors, scale_model_sz):
        scale_sample = []
        base_target_sz=np.array([base_target_sz[0],base_target_sz[1]])
        for idx, scale in enumerate(scale_factors):
            patch_sz = np.floor(base_target_sz * scale)
            im_patch=cv2.getRectSubPix(im,(int(patch_sz[0]),int(patch_sz[1])),pos)
            if scale_model_sz[0] > patch_sz[1]:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA

            im_patch_resized = cv2.resize(im_patch, (int(scale_model_sz[0]),int(scale_model_sz[1])), interpolation=interpolation).astype(np.uint8)
            scale_sample.append(extract_hog_feature(im_patch_resized,cell_size=4).reshape((-1, 1)))
        scale_sample = np.concatenate(scale_sample, axis=1)
        return scale_sample


class LPScaleEstimator:
    def __init__(self,target_sz,config):
        self.learning_rate_scale=config.learning_rate_scale
        self.scale_sz_window = config.scale_sz_window
        self.target_sz=target_sz

    def init(self,im,pos,base_target_sz,current_scale_factor):
        w,h=base_target_sz
        avg_dim = (w + h) / 2.5
        self.scale_sz = ((w + avg_dim) / current_scale_factor,
                         (h + avg_dim) / current_scale_factor)
        self.scale_sz0 = self.scale_sz
        self.cos_window_scale = cos_window((self.scale_sz_window[0], self.scale_sz_window[1]))
        self.mag = self.cos_window_scale.shape[0] / np.log(np.sqrt((self.cos_window_scale.shape[0] ** 2 +
                                                                    self.cos_window_scale.shape[1] ** 2) / 4))

        # scale lp
        patchL = cv2.getRectSubPix(im, (int(np.floor(current_scale_factor * self.scale_sz[0])),
                                                 int(np.floor(current_scale_factor * self.scale_sz[1]))), pos)
        patchL = cv2.resize(patchL, self.scale_sz_window)
        patchLp = cv2.logPolar(patchL.astype(np.float32), ((patchL.shape[1] - 1) / 2, (patchL.shape[0] - 1) / 2),
                               self.mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        self.model_patchLp = extract_hog_feature(patchLp, cell_size=4)

    def update(self,im,pos,base_target_sz,current_scale_factor):
        patchL = cv2.getRectSubPix(im, (int(np.floor(current_scale_factor * self.scale_sz[0])),
                                                   int(np.floor(current_scale_factor* self.scale_sz[1]))),pos)
        patchL = cv2.resize(patchL, self.scale_sz_window)
        # convert into logpolar
        patchLp = cv2.logPolar(patchL.astype(np.float32), ((patchL.shape[1] - 1) / 2, (patchL.shape[0] - 1) / 2),
                               self.mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        patchLp = extract_hog_feature(patchLp, cell_size=4)
        tmp_sc, _, _ = self.estimate_scale(self.model_patchLp, patchLp, self.mag)
        tmp_sc = np.clip(tmp_sc, a_min=0.6, a_max=1.4)
        scale_factor=current_scale_factor*tmp_sc
        self.model_patchLp = (1 - self.learning_rate_scale) * self.model_patchLp + self.learning_rate_scale * patchLp
        return scale_factor

    def estimate_scale(self,model,obser,mag):
        def phase_correlation(src1,src2):
            s1f=fft2(src1)
            s2f=fft2(src2)
            num=s2f*np.conj(s1f)
            d=np.sqrt(num*np.conj(num))+2e-16
            Cf=np.sum(num/d,axis=2)
            C=np.real(ifft2(Cf))
            C=np.fft.fftshift(C,axes=(0,1))

            mscore=np.max(C)
            pty,ptx=np.unravel_index(np.argmax(C, axis=None), C.shape)
            slobe_y=slobe_x=1
            idy=np.arange(pty-slobe_y,pty+slobe_y+1).astype(np.int64)
            idx=np.arange(ptx-slobe_x,ptx+slobe_x+1).astype(np.int64)
            idy=np.clip(idy,a_min=0,a_max=C.shape[0]-1)
            idx=np.clip(idx,a_min=0,a_max=C.shape[1]-1)
            weight_patch=C[idy,:][:,idx]

            s=np.sum(weight_patch)+2e-16
            pty=np.sum(np.sum(weight_patch,axis=1)*idy)/s
            ptx=np.sum(np.sum(weight_patch,axis=0)*idx)/s
            pty=pty-(src1.shape[0])//2
            ptx=ptx-(src1.shape[1])//2
            return ptx,pty,mscore

        ptx,pty,mscore=phase_correlation(model,obser)
        rotate=pty*np.pi/(np.floor(obser.shape[1]/2))
        scale = np.exp(ptx/mag)
        return scale,rotate,mscore

class MRScaleEstimator:
    def __init__(self, radius, phi_num):
        if not isinstance(radius, list):
            raise Exception("must be one dimensional list item, such as [1,2,3,4,5]")
        self.phi_num = phi_num
        self.phi_bin = 360/self.phi_num
        self.scale_num = len(radius)
        self.scale_factor = [1 for _ in range(self.scale_num)]
        self.expand_factor = 2
        self.oriented_angle = 0 # du, not rad
        self.radius = radius
        self.img_polar = None
        self.img_center = None

    def estimate_scale(self, img, obj_center, oriented_angle, last_obj_img):
        # get cutted image according to expand factor
        # last_obj_img = last_img[int(last_obj_center[1]-last_rec_size[1]/2):int(last_obj_center[1]+last_rec_size[1]/2),
        #                int(last_obj_center[0]-last_rec_size[0]/2):int(last_obj_center[0]+last_rec_size[0]/2),:]

        h_o, w_o, _ = last_obj_img.shape
        h, w, _ = img.shape
        h_expanded, w_expanded = 2*h_o, 2*w_o
        x_center, y_center = obj_center
        xmin, ymin, xmax, ymax = int(x_center-w_expanded/2), int(y_center-h_expanded/2), int(x_center+w_expanded/2), int(y_center+h_expanded/2)
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(xmax, w-1), min(ymax, h-1)
        img_cut = img[ymin:ymax, xmin:xmax, :]
        h_cut, w_cut, _ =  img_cut.shape
        if h_cut < h_expanded or w_cut < w_expanded:
            w_boarder_left = int((w_expanded-w_cut)/2)
            w_boarder_right = w_expanded - w_boarder_left - w_cut
            h_boarder_top = int((h_expanded-h_cut)/2)
            h_boarder_bottom = h_expanded - h_boarder_top - h_cut
            img_cut = cv2.copyMakeBorder(img_cut, h_boarder_top, h_boarder_bottom, w_boarder_left, w_boarder_right, cv2.BORDER_CONSTANT,(0,0,0))

        if img_cut.shape[0] != h_expanded or img_cut.shape[1] != w_expanded:
            raise Exception("Wrong dimension of img_cut.")

        # convert img from xy to \tho\phi
        import math
        x_center_new, y_center_new = x_center-xmin+w_boarder_left, y_center-ymin+h_boarder_top
        r1 = math.hypot(x_center_new, y_center_new)
        r2 = math.hypot(w_expanded-x_center_new-1, y_center_new)
        r3 = math.hypot(x_center_new, h_expanded-y_center_new-1)
        r4 = math.hypot(w_expanded-x_center_new-1, h_expanded-y_center_new-1)
        maxRadius = max(r1, r2, r3, r4)
        m = max(h_expanded, w_expanded)/math.log(maxRadius)
        img_cut_polar = cv2.logPolar(img_cut, (x_center_new, y_center_new), m, cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
        self.img_center = (x_center_new, y_center_new)
        self.img_polar = img_cut_polar

        # polar axis value
        phi_num, tho_num, _ = img_cut_polar.shape
        phi_bin = 360/phi_num
        phi = [i * phi_bin for i in range(phi_num)]
        tho = list(range(tho_num))

        # the same size last_obj_image
        w_boarder_left_last = int((w_expanded - w_o) / 2)
        w_boarder_right_last = w_expanded - w_boarder_left_last - w_o
        h_boarder_top_last = int((h_expanded - h_o) / 2)
        h_boarder_bottom_last = h_expanded - h_boarder_top_last - h_o
        last_obj_img = cv2.copyMakeBorder(last_obj_img, h_boarder_top_last, h_boarder_bottom_last, w_boarder_left_last, w_boarder_right_last,
                                     cv2.BORDER_CONSTANT, (0, 0, 0))

        if last_obj_img.shape[0] != h_expanded or last_obj_img.shape[1] != w_expanded:
            raise Exception("Wrong dimension of img_cut.")

        # convert last obj image into semi-polar coordinates
        x_center_last, y_center_last = int(w_o/2+w_boarder_left_last), int(h_o/2+h_boarder_top_last)
        r1 = math.hypot(x_center_last, y_center_last)
        r2 = math.hypot(w_expanded - x_center_last - 1, y_center_last)
        r3 = math.hypot(x_center_last, h_expanded - y_center_last - 1)
        r4 = math.hypot(w_expanded - x_center_new - 1, h_expanded - y_center_last - 1)
        maxRadius_last = max(r1, r2, r3, r4)
        m_last = max(h_expanded, w_expanded) / math.log(maxRadius_last)
        last_obj_img_polar = cv2.logPolar(last_obj_img, (x_center_new, y_center_new), m,
                                     cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        # rotate img in semi-polar coordinates
        _ = input("Pay attention to the oriented_angle!!!!!!!!!!!!\n\
                  was delta_angle in (f with respect to h) or backwards\n\
                  here in code it is h+delta_angle = f")
        bin_delta = 0
        if abs(self.oriented_angle + oriented_angle) > self.phi_bin:
            signs = abs(self.oriented_angle + oriented_angle)/(self.oriented_angle + oriented_angle)
            bin_delta = int(abs(self.oriented_angle + oriented_angle)/phi_bin)*signs
            last_obj_img_polar = np.roll(last_obj_img_polar, bin_delta, axis=0)
        self.oriented_angle = (abs(self.oriented_angle + oriented_angle) % self.phi_bin)*signs

        # rotate radius since it comes from the last frame
        radius = []
        resize_factor = (self.scale_num-1)/(h_expanded-1)
        for idx in range(h_expanded):
            idx_scale = idx*resize_factor
            former_index = max(np.floor(idx_scale), 0)
            later_index = min(np.ceil(idx_scale), self.scale_num-1)
            if former_index == later_index:
                radius.append(self.radius[int(later_index)])
            else:
                p = [radius[int(former_index)], radius[int(later_index)]]
                value = (p[1]-p[0])*(idx_scale-former_index)+p[0]
                radius.append(value)
        if bin_delta != 0:
            radius = np.roll(np.array(radius), bin_delta)

        for idx in range(h_expanded):
            idx_scale = idx * resize_factor
            if int(idx_scale) - idx_scale == 0:
                self.radius[int(idx_scale)] = radius[idx]

        # resize
        img_cut_polar = cv2.resize(img_cut_polar, (w_expanded, self.phi_num))
        last_obj_img_polar = cv2.resize(last_obj_img_polar, (w_expanded, self.phi_num))

        # calculate radius each direction
        for idx in range(self.phi_num):
            f = img_cut_polar[idx, :]
            h = last_obj_img_polar[idx, :]
            fft=np.fft.fft
            ifft=np.fft.ifft
            H = fft(h*np.hanning(len(h)))
            # since we don't care about amplitude or energy, there is no need to use restitution coefficient
            response = np.real(ifft(fft(f*np.hanning(len(f)))*np.conj(H)))
            s_index = np.argmax(response)

            p = response[s_index-1:s_index+2]
            delta_s = 0.5 * (p[2] - p[0]) / (2 * p[1] - p[2] - p[0])
            if not np.isfinite(delta_s):
                delta_s = 0
            s = s_index + delta_s

            if s + 1 > len(response) / 2:
                s = s - response.shape[0]

            self.scale_factor[idx] = s
            self.radius[idx] = s*self.radius[idx]

    def estimate_obj(self, img):
        phi = [i*360/self.scale_num for i in range(self.scale_num)]
        x = [self.img_center[0]+int(self.radius[i]*np.cos(phi[i]/180*np.pi)) for i in range(self.scale_num)]
        xmin = min(x)
        xmax = max(x)
        y = [self.img_center[1]+int(self.radius[i]*np.sin(phi[i]/180*np.pi)) for i in range(self.scale_num)]
        ymin = min(y)
        ymax = max(y)
        x,y,w,h=xmin, ymin, xmax-xmin, ymax-ymin
        points = [[x,y] for i in range(self.scale_num)] # anticlockwise points

        flag = np.zeros_like(img)

        for idx in range(self.scale_num):
            cur_point = points[idx]
            next_point = points[(idx+1)%self.scale_num]
            cv2.line(flag, cur_point, next_point, (255,255,255),1)

        flag = np.uint8(flag[:,:,0])
        num, labels = cv2.connectedComponents(flag, 8)
        item_label = labels[self.img_center[1], self.img_center[0]]
        flag[flag == item_label] = 1
        flag[flag != item_label] = 0

        obj_sub_img = img[ymin:ymax, xmin:xmax, :]
        flag_cut = flag[ymin:ymax, xmin:xmax]
        obj = obj_sub_img.copy()
        for i in range(obj.shape[2]):
            obj[:,:,i] = obj[:,:,i]*flag_cut

        return (x,y,w,h), obj


def test_MRScaleEstimator():
    import math
    w = 10**2
    h = 10**2
    frame = np.zeros((h, w, 3))
    # draw circle
    a0 = 50
    b0 = 30
    center1 = (w//2, h//2)
    for i in range(h):
        for j in range(w):
            if (i-center1[1])**2/a0**2+(j-center1[0])**2/b0**2 <= 1:
                frame[i, j, :] = (255, 255, 255)
    cv2.imwrite("/home/shawn/scripts_output_tmp/zzz.jpg", frame)

    later_frame = np.zeros((h, w, 3))
    # draw eclipse
    a = 30
    b = 10
    center2 = (w // 2 - 15, h // 2 + 25)
    alpha = math.pi/6
    for i in range(h):
        for j in range(w):
            term1 = ((i - center2[1]) * math.cos(alpha) + (j-center2[0]) * math.sin(alpha)) ** 2 / a ** 2
            term2 = ((center2[1] - 1) * math.sin(alpha) + (j - center2[0]) * math.cos(alpha)) ** 2 / b ** 2
            if term1 + term2 <= 1:
                later_frame[i, j, :] = (255, 255, 255)
    cv2.imwrite("/home/shawn/scripts_output_tmp/zzz1.jpg", frame)


if __name__ == "__main__":
    test_MRScaleEstimator()












