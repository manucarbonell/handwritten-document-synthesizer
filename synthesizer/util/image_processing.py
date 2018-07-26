import cv2
import numpy as np

def load_image_float(fname, as_color=True):
    if as_color:
        img=cv2.cvtColor(cv2.imread(fname,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    else:
        img=cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
    return img/255.0

def save_image_float(img, fname, as_color=True, mkdir=True):
    img=(img*255).astype('uint8')
    if as_color:
        if len(img.shape)==2:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        else:
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    else:
        if len(img.shape)==3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(fname,255-img)

def box_filter1d(img, box_sz, horizontal=True, iter=1):
    # TODO (anguelos) add proper border options
    assert box_sz % 2 == 0 and box_sz > 0
    if horizontal:
        tmp_img = np.empty([img.shape[0], img.shape[1] + box_sz])
        tmp_img[:, box_sz / 2:-box_sz / 2] = img
        tmp_img[:, :box_sz / 2] = 0
        tmp_img[:, -box_sz / 2:] = 0

        div_map = np.ones(tmp_img.shape) * box_sz
        div_map[:, :box_sz] = np.ones([img.shape[0], box_sz]).cumsum(axis=1)
        div_map[:, -box_sz:] = np.ones(
            [img.shape[0], box_sz]).cumsum(axis=1)[:, ::-1]

        for _ in range(iter):
            new_img = np.empty(tmp_img.shape)
            new_img[:, :box_sz / 2] = 0
            new_img[:, -box_sz / 2:] = 0
            cs = tmp_img.cumsum(axis=1)
            new_img[:, box_sz / 2:-box_sz / 2] = (cs[:, box_sz:] -
                                                  cs[:, :-box_sz])
            new_img /= div_map
            tmp_img = new_img
        return new_img[:, box_sz / 2:-box_sz / 2]
    else:
        tmp_img = np.empty([img.shape[0] + box_sz, img.shape[1]]);
        tmp_img[box_sz / 2:-box_sz / 2, :] = img;
        tmp_img[:box_sz / 2, :] = 0;
        tmp_img[-box_sz / 2:, :] = 0;

        div_map = np.ones(tmp_img.shape) * box_sz
        div_map[:box_sz, :] = np.ones([box_sz, img.shape[1]]).cumsum(axis=0)
        div_map[-box_sz:, :] = np.ones([box_sz, img.shape[1]]).cumsum(axis=0)[
                               ::-1, :]

        for _ in range(iter):
            new_img = np.empty(tmp_img.shape)
            new_img[:box_sz / 2, :] = 0;
            new_img[-box_sz / 2:, :] = 0;
            cs = tmp_img.cumsum(axis=0)
            new_img[box_sz / 2:-box_sz / 2, :] = (
                        cs[box_sz:, :] - cs[:-box_sz, :])
            new_img /= div_map
            tmp_img = new_img
        return new_img[box_sz / 2:-box_sz / 2, :]


def fake_gaussian(img, vertical_horizontal_sigma, iter=3):
    """Gaussian filter aproximation with integrall images.

    :param img:
    :param vertical_horizontal_sigma:
    :param iter: An integer with the number of consecutive box filters used to approximate the gaussian kernel.
    :return: an image of the same size as img.
    """
    sigma_vertical, sigma_horizontal = vertical_horizontal_sigma
    h_blured = box_filter1d(img, sigma_horizontal, horizontal=True, iter=iter)
    blured = box_filter1d(h_blured, sigma_vertical, horizontal=False, iter=iter)
    return blured


def raw_interp2(X, Y, img):
    """
    Samples from an image each point of the provided geometric space.
    To be solved:
    - Artifact Y instability seen on the spline interpolation.
    - Image getting rotated 180 degrees.
    """
    e = .001
    X = X.copy()
    X[X < 0] = 0
    X[X >= img.shape[1] - 2] = img.shape[1] - 2
    Y = Y.shape[0]-Y
    Y[Y < 0] = 0;
    Y[Y >= img.shape[0] - 2] = img.shape[0] - 2

    left = np.floor(X)  # .astype('int32')
    right_coef = X - left
    left_coef = 1 - right_coef
    right = left + 1  # .astype('int32')

    top = np.floor(Y)  # .astype('int32')
    bottom_coef = Y - top
    top_coef = 1 - bottom_coef
    bottom = top + 1  # .astype('int32')

    left = left.astype("int32")
    top = top.astype("int32")
    right = right.astype("int32")
    bottom = bottom.astype("int32")

    res = np.empty(img.shape)
    if len(img.shape)==2:
        res[:] = img[top, left] * left_coef * top_coef + img[
            top, right] * right_coef * top_coef + \
                img[bottom, left] * left_coef * bottom_coef + img[
                    bottom, right] * right_coef * bottom_coef
    elif len(img.shape)==3:
        #res=res.reshape(-1,3)
        for c in range(3):
            cimg=img[:,:,c]
            res[:,:,c] = cimg[top, left] * left_coef * top_coef + cimg[top, right] * right_coef * top_coef + cimg[bottom, left] * left_coef * bottom_coef + cimg[bottom, right] * right_coef * bottom_coef
        res=res.reshape(img.shape)

    res[res < 0] = 0
    res[res > 1] = 1
    return res
