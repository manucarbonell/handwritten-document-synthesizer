# -*- coding: utf-8 -*-

# python built-in
import codecs
import string
import pickle
import os
import sys
import StringIO
from math import floor, ceil

# image processing
import cv2
import scipy.interpolate as interpolate
import scipy.ndimage as ndi
import skimage
import skimage.filters
from matplotlib import pyplot as plt
from PIL import Image

# other packages
import pango
import numpy as np

# own packages
from util import *

NOT_IMPLEMENTED_ERROR_STR = "This method must be implemented in a subclass."

def open_txt(filename, mode):
    """ Opens a text file. If using something other than utf-8, just change it here.
    """
    return codecs.open(filename, mode=mode, encoding="utf-8")

""" Distortion classes """

class PixelOperation(object):
    """ Base class for pixel operations
    """
    def __init__(self, disabled=False):
        self.disabled = disabled

    def generate_parameters(self):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_STR)

    def deterministic(self, img, parameter_list):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_STR)

    def __call__(self, img):
        return self.deterministic(img, self.generate_parameters())

    def apply_on_image(self, img, ltrb):
        if self.disabled:
            self.generate_parameters()
            return img, ltrb
        else:
            return self(img), ltrb


class DocumentNoise(PixelOperation):
    """ Adds gaussian filtered noise to the image
    """
    def __init__(self, img_shape, noise_sparsity=500, disabled=False):
        self.disabled = disabled
        self.img_shape = img_shape
        self.nb_pixels = img_shape[0] * img_shape[1]
        self.noise_sparsity = noise_sparsity

    def generate_parameters(self):
        low_noise_indexes = np.random.randint(0, self.nb_pixels * 256,
                                              self.nb_pixels / self.noise_sparsity)
        return [low_noise_indexes, 0.7, [1, 1], [1, 1], 0.9]

    def deterministic(self, img, parameter_list=[np.array([0], dtype='int32'), 0.5, [1, 1], [1, 1], 0.6]):
        low_noise_indexes, low_pass_ignore, low_pass_sigma, high_pass_sigma, low_pass_range = parameter_list
        #low_noise = np.zeros(img.shape[:2], dtype='float')
        #low_noise.reshape(-1)[low_noise_indexes // 256] = low_noise_indexes % 256
        #plt.imshow(low_noise)
        #plt.show()

        # low_noise=(np.random.rand(img.shape[0],img.shape[1])>(1.0/noise_sparsity)).astype('float32')
        #low_pass = skimage.filters.gaussian(low_noise[:, :], low_pass_sigma, multichannel=False)

        #low_pass = (low_pass - np.min(low_pass)) / (
        #        np.max(low_pass) - np.min(low_pass))

        #low_pass -= low_pass_ignore
        #low_pass = low_pass * (low_pass > low_pass_ignore)
        #low_pass = np.maximum(low_pass, 0)
        high_pass = ((np.random.rand(img.shape[0], img.shape[1]) * (
                 1 - low_pass_range) + (low_pass_range)))[:,:, None]
        high_pass = img * skimage.filters.gaussian(high_pass[:,:,:], high_pass_sigma, multichannel=True)
        # return 1-high_pass
        #res = (low_pass[:,:,None] + high_pass)  # np.maximum(low_pass,img)
        #print res[:,:,0].min(),res[:,:,0].max()
        #print res[:,:,1].min(),res[:,:,1].max()
        #print res[:,:,2].min(),res[:,:,2].max()
        #res = res - res.min() / (res.max() - res.min())
        return high_pass

class ImageBackground(PixelOperation):
    """
    Blends the foreground image with a background image.

    Parameters
    ----------
    image_fname_list : [str]
        A list of paths to background images from which one will be randomly selected.
    resize_mode : str
        The mode which will be used to resize the background in case its dimensions
        do not correspond with the foreground.
        Possible values:
            'scale': Rescales the background so that it fits the foreground.
            'tile': Tiles the background until it fills the foreground.
                    If the foreground is smaller than the background, the background gets cropped.
    blend_mode : str
        The mode which will be used to blend the foreground with the background.
        Possible values:
            'max': For each pixel, takes the one with the maximum value.
            'min': For each pixel, takes the one with the minimum value.
            'mean': Performs the arithmetic mean between the two images.
            'alpha': Treats the foreground as an semi-transparent layer and
                     blends it multiplicatively with the background.
                     The foreground must be a grayscale image where white indicates
                     the lowest alpha and black the highest.
    alpha : float
        If blend mode 'alpha' is specified, this value is used as the transparency
        if the foreground. 0 is completely transparent and 1 is completely opaque.

    """
    def __init__(self, image_fname_list=None, blend_mode='alpha', resize_mode='scale', alpha=1.0, disabled=False):
        self.disabled = disabled
        self.image_fname_list = image_fname_list
        self.alpha = alpha

        resize_modes = {"tile": self.resize_bg_tile, "scale": self.resize_bg_scale}
        blend_modes = {"max": self.blend_max, "min": self.blend_min, "mean": self.blend_mean, "alpha": self.blend_alpha}

        self.resize_bg = resize_modes[resize_mode]
        self.blend = blend_modes[blend_mode]

    def generate_parameters(self):
        return [np.random.choice(self.image_fname_list)]

    def deterministic(self, img, parameter_list):
        image_fname, = parameter_list
        # moved image loading here, since now __init__ executes for every page
        # still could be improved since the same image is going to get loaded several times
        bg = load_image_float(image_fname, as_color=True)
        bg = self.resize_bg(bg, img.shape)

        if len(bg.shape) != len(img.shape):
            if len(bg.shape) == 2 and len(img.shape) == 3:
                bg = np.array([bg,bg,bg]).swapaxes(2,0)
            elif len(bg.shape) == 3 and len(img.shape) == 2:
                img = np.array([img,img,img]).swapaxes(1,0).swapaxes(1,2)
            else:
                raise ValueError
        res = self.blend(img,bg)
        return res

    def __call__(self,img):
        return self.deterministic(img, self.generate_parameters())

    def resize_bg_tile(self,bg,fg_shape):
        fg_width = fg_shape[1]
        fg_height = fg_shape[0]
        bg_width = bg.shape[1]
        bg_height = bg.shape[0]
        res = np.empty([fg_height, fg_width, 3])
        
        for left in range(0, fg_width, bg_width):
            right = min(left + bg_width, fg_width)
            width = right - left
            for top in range(0, fg_height, bg_height):
                bottom = min(top + bg_height, fg_height)
                height = bottom - top
                res[top:bottom,left:right,:] = bg[:height,:width,:]
        
        return res

    def resize_bg_scale(self,bg,fg_shape):
        return cv2.resize(bg,(fg_shape[1],fg_shape[0])) # TODO(anguelos) make cv2 optional

    def blend_max(self,fg,bg):
        return np.maximum(fg,bg)

    def blend_min(self,fg,bg):
        return np.minimum(fg,bg)

    def blend_mean(self,fg,bg):
        return (fg+bg)/2

    def blend_alpha(self, fg, bg):
        return bg*(np.maximum(fg, 1-self.alpha))

class GeometricOperation(object):
    """ Base class for geometric operations
    """
    def __init__(self, disabled=False):
        self.disabled = disabled

    def generate_parameters(self):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_STR)

    def deterministic(self, X, Y, parameter_list):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_STR)

    def __call__(self, x_coordinate_list, y_coordinate_list):
        res_x_coordinate_list = []
        res_y_coordinate_list = []
        params=self.generate_parameters()
        for k in range(len(x_coordinate_list)):
            X = x_coordinate_list[k].copy()
            Y = y_coordinate_list[k].copy()
            X, Y = self.deterministic(X, Y, params)
            res_x_coordinate_list.append(X)
            res_y_coordinate_list.append(Y)
        return res_x_coordinate_list, res_y_coordinate_list

    def apply_on_image(self, img, ltrb):
        if self.disabled:
            self.generate_parameters()
            return img, ltrb

        l = ltrb[:, 0]
        t = ltrb[:, 1]
        r = ltrb[:, 2]
        b = ltrb[:, 3]
        X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        in_x_coords = [X, l, r]#[X, l, l, r, r]
        in_y_coords = [Y, t, b]#[Y, t, b, t, b]
        out_x_coords, out_y_coords = self(in_x_coords,in_y_coords)
        #[X, x1, x2, x3, x4] = out_x_coords
        [X, x1, x2] = out_x_coords
        #[Y, y1, y2, y3, y4] = out_y_coords
        [Y, y1, y2] = out_y_coords
        res_img = raw_interp2(X, Y, img)
        res_ltrp = np.empty(ltrb.shape)
        res_ltrp[:, 0] = x1#np.min([x1,x2],axis=0)#,x3,x4], axis=0)
        res_ltrp[:, 2] = x2##np.max([x1, x2],axis=0)#, x3, x4], axis=0)
        res_ltrp[:, 1] = y1#np.min([y1, y2],axis=0)#, y3, y4], axis=0)
        res_ltrp[:, 3] = y2#np.max([y1, y2],axis=0)#, y3, y4], axis=0)
        return res_img, res_ltrp


class GeometricSequence(GeometricOperation):
    def generate_parameters(self):
        return None

    def __init__(self, disabled=False, *args):
        self.disabled = disabled
        self.transform_sequences = args

    def deterministic(self, X, Y, parameters):
        del parameters
        for transform in self.transform_sequences:
            X, Y = transform(X, Y)
        return X, Y


class GeometricClipper(GeometricOperation):
    def __init__(self,clip_ltrb, disabled=False):
        self.disabled = disabled
        self.clip_ltrb = clip_ltrb

    def generate_parameters(self):
        return self.clip_ltrb

    def deterministic(self, X, Y, parameter_list):
        x_min, y_min, x_max, y_max = parameter_list
        X[X < x_min] = x_min
        X[X > x_max] = x_max
        Y[Y < y_min] = y_min
        Y[Y > y_max] = y_max
        return X, Y

class GeometricPaperWrapper(GeometricOperation):
    """
    Applies a paper wrapping effect on the image using spline interpolation.

    Parameters
    ----------
    max_y_disp : int
        Controls the maximum vertical displacement of the control points from the
        base line (horizontal through the middle of the image).
        The higher, the more accentuated the curves will be.
    num_points : int
        Number of control points which will be used to perform the spline interpolation.
        The higher, the more curves the wrapping will have. Minimum number of points is 3,
        since the first and last point are always fixed at the base line.
    """
    def __init__(self, page_size, max_y_disp, num_points=5, disabled=False):
        self.disabled = disabled
        self.num_points = num_points
        self.page_size = page_size
        self.max_y_disp = max_y_disp

    def generate_parameters(self):
        xpoints = ((np.random.rand(self.num_points)+0.5) * np.array(
            [0] + [1] * (self.num_points - 1))).cumsum()
        xpoints = xpoints / xpoints.max()
        ypoints = (np.random.standard_normal(self.num_points) ) * np.array([0] + \
                   [self.max_y_disp] * (self.num_points - 2) + [0])
        return xpoints, ypoints

    def deterministic(self, X, Y, parameter_list):
        xpoints, ypoints = parameter_list
        ticks = interpolate.splrep(xpoints, ypoints)
        all_x = np.linspace(0, 1, self.page_size[1])
        all_y = (interpolate.splev(all_x, ticks))
        if len(Y.shape)==2:
            Y = Y + all_y[None,:]
        elif len(Y.shape)==1:
            Y[:]=Y[:] + all_y[X[:]]
        else:
            raise ValueError
        return X, Y

class GeometricRandomTranslator(GeometricOperation):
    def __init__(self,x_sigma,x_mean,y_sigma,y_mean, disabled=False):
        self.disabled = disabled
        self.x_sigma=x_sigma
        self.x_mean=x_mean
        self.y_sigma=y_sigma
        self.y_mean=y_mean

    def generate_parameters(self,point_shape):
        res_x = np.random.standard_normal(point_shape) * self.x_sigma + self.x_mean
        res_y = np.random.standard_normal(point_shape) * self.y_sigma + self.y_mean
        return res_x,res_y

    def deterministic(self, X, Y, parameter_list):
        return X+parameter_list[0],Y+parameter_list[1]

    def __call__(self, x_coordinate_list, y_coordinate_list):
        res_x_coordinate_list = []
        res_y_coordinate_list = []
        for k in range(len(x_coordinate_list)):
            X = x_coordinate_list[k].copy()
            Y = y_coordinate_list[k].copy()
            X, Y = self.deterministic(X, Y, self.generate_parameters(X.shape))
            res_x_coordinate_list.append(X)
            res_y_coordinate_list.append(Y)
        return res_x_coordinate_list, res_y_coordinate_list

    def apply_on_image(self, img, ltrb):
        l = ltrb[:, 0]
        t = ltrb[:, 1]
        r = ltrb[:, 2]
        b = ltrb[:, 3]
        in_x_coords = [l,r]
        in_y_coords = [t, b]
        out_x_coords, out_y_coords = self(in_x_coords,in_y_coords)
        res_ltrb = np.empty(ltrb.shape)
        res_ltrb[:, 0] = out_x_coords[0]
        res_ltrb[:, 2] = out_x_coords[1]
        res_ltrb[:, 1] = out_y_coords[0]
        res_ltrb[:, 3] = out_y_coords[1]
        return img, res_ltrb

# code adapted from Augmentor
# https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
class ElasticDistortion(PixelOperation):
    """
    This class performs randomised, elastic distortions on images.
    """
    def __init__(self, img_shape, grid_width, grid_height, magnitude, disabled=False):
        """
        The granularity of the distortions produced by this class
        can be controlled using the width and height of the overlaying
        distortion grid. The larger the height and width of the grid,
        the smaller the distortions. This means that larger grid sizes
        can result in finer, less severe distortions.
        As well as this, the magnitude of the distortions vectors can
        also be adjusted.
        :param grid_width: The width of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param grid_height: The height of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param magnitude: Controls the degree to which each distortion is
         applied to the overlaying distortion grid.
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        """
        self.disabled = disabled
        self.img_shape = img_shape
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = abs(magnitude)

    def generate_parameters(self):
        w, h = self.img_shape

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        # For loop that generates polygons could be rewritten, but maybe harder to read?
        # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

        # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for a, b, c, d in polygon_indices:
            dx = np.random.randint(-self.magnitude, self.magnitude+1)
            dy = np.random.randint(-self.magnitude, self.magnitude+1)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        return [generated_mesh]


    def deterministic(self, img, parameter_list):
        mesh, = parameter_list
        return img.transform(img.size, Image.MESH, mesh, resample=Image.BICUBIC)

# code adapted from ocrodeg
# https://github.com/NVlabs/ocrodeg/blob/master/ocrodeg/degrade.py
class InkDegradation(PixelOperation):
    """
    This class adds randomised ink erasing and ink stains.
    """
    def __init__(self, img_shape, stain_density=2e-5, stain_size=4, stain_roughness=1,
                       erasing_density=2e-4, erasing_size=4, erasing_roughness=2, disabled=False):
        self.disabled = disabled
        self.img_shape = img_shape
        self.stain_density = stain_density
        self.stain_size = stain_size
        self.stain_roughness = stain_roughness
        self.erasing_density = erasing_density
        self.erasing_size = erasing_size
        self.erasing_roughness = erasing_roughness

    def generate_parameters(self):
        h, w = self.img_shape

        def generate_mask(density):
            num_blobs = int(density * w * h)
            mask = np.zeros((h, w), 'i')
            for i in xrange(num_blobs):
                mask[np.random.randint(0, h), np.random.randint(0, w)] = 1
            return mask

        stain_mask = generate_mask(self.stain_density) 
        erasing_mask = generate_mask(self.erasing_density)

        return [stain_mask, erasing_mask]

    def deterministic(self, img, parameter_list):
        stain_mask, erasing_mask = parameter_list
        h, w = self.img_shape

        def random_blobs(mask, size, roughness):
            dt = ndi.distance_transform_edt(1-mask)
            mask =  np.array(dt < size, 'f')
            mask = ndi.gaussian_filter(mask, size/(2*roughness))
            mask -= np.amin(mask)
            mask /= np.amax(mask)
            noise = np.random.rand(h, w)
            noise = ndi.gaussian_filter(noise, size/(2*roughness))
            noise -= np.amin(noise)
            noise /= np.amax(noise)
            return np.array(mask * noise > 0.5, 'uint8')

        stain_blobs = random_blobs(stain_mask, self.stain_size, self.stain_roughness)
        erasing_blobs = random_blobs(erasing_mask, self.erasing_size, self.erasing_roughness)
        new_img = img.copy()
        new_img[stain_blobs == 1] = np.min(img)
        new_img[erasing_blobs == 1] = np.max(img)

        return new_img


""" Classes for data organization """

class Form(object):
    def __init__(self, _id, text, line_data=None, word_data=None):
        self.id = _id
        self.text = text
        self.word_data = word_data
        self.line_data = line_data

class Author(object):
    def __init__(self, id_):
        self.id = id_
        self.font = None
        self.forms = []
    def add_form(self, form):
        self.forms.append(form)

class Corpora(object):
    def __init__(self, corpora_dir):
        self.corpora_dir = corpora_dir
        self.files = os.listdir(self.corpora_dir)
        self.num_files = len(self.files)
        self.curr_file = 0
        self._load_next_file()

    def get_text(self, nchars):
        text = self.text.read(nchars)
        if not text:
            if self._load_next_file():
                text = self.get_text(nchars)
            else:
                print "out of corpora"
                text = ""
        return text

    def _load_next_file(self):
        if self.curr_file == self.num_files:
            return False
        print self.files[self.curr_file]
        with open_txt(os.path.join(self.corpora_dir, self.files[self.curr_file]), mode="r") as f:
            self.text = StringIO.StringIO(f.read())
            self.curr_file += 1
            return True

""" Classes related to the synthesis process """

class Synthesizer(object):
    """ Base class for the synthesizers
    """
    def __init__(self, letter_height, words, lines, out_path, bg_paths, font_list, constant_width, distort_bboxes_bool):

        # variable init
        self.letter_height = letter_height
        self.words = words
        self.lines = lines
        self.out_path = out_path
        self.bg_paths = bg_paths
        self.font_list = font_list
        self.constant_width = constant_width
        self.distort_bboxes_bool = distort_bboxes_bool

        # predetermined values
        self.crop_edge_ltrb = np.array([letter_height*2]*4)
        self.font_alignment = pango.ALIGN_LEFT
        self.page_count = 0

        # folder and file init
        self.params_path = os.path.join(self.out_path, "parameters")
        mkdir_p(self.params_path)
        self.form_gt_path = os.path.join(self.out_path, "gt_forms")
        mkdir_p(self.form_gt_path)
        self.forms_path = os.path.join(self.out_path, "forms")
        mkdir_p(self.forms_path)
        if self.words:
            self.gt_words_file = open_txt(os.path.join(self.out_path, "gt_words.txt"), "w")
            self.words_path = os.path.join(self.out_path, "words")
            mkdir_p(self.words_path)
        if self.lines:
            self.gt_lines_file = open_txt(os.path.join(self.out_path, "gt_lines.txt"), "w")
            self.lines_path = os.path.join(self.out_path, "lines")
            mkdir_p(self.lines_path)

    def finalize(self):
        """ Clean up after the synth has been used
        """
        if self.words:
            self.gt_words_file.close()
        if self.lines:
            self.gt_lines_file.close()

    def render_page_text(self):
        """ Template method for rendering the page text
        """
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_STR)

    def generate_segments(self, segment_data, suffix=""):
        """
        Generates segment bounding boxes and captions given segment data
        :param segment_data: A list of dictionaries containing the following keys:
            "id":string The id which will be assigned to the segment.
                        If not specified, it's automatically generated.
            "start":int The starting index of the segment in the complete page text
            "end":int The ending index of the segment in the complete page text
        :param suffix: Suffix which gets added to the segments ids when automatically generated.
                       Useful to distinguish between different segment types.
        """
        segment_indices = []
        segment_ids = []
        for i, segment in enumerate(segment_data):
            if "id" in segment:
                segment_ids.append(segment["id"])
            else:
                segment_ids.append("{}-{}{}".format(self.page_count, i, suffix))
            segment_indices.append([segment["start"], segment["end"]])

        ranges = np.array(segment_indices, dtype='int32')
        roi_captions, bboxes = self.stitch_ranges(self.current_page_caption, ranges, self.current_grapheme_ltrb)
        bboxes = self.dilate_bboxes(bboxes)
        if self.distort_bboxes_bool:
            bboxes = self.distort_bboxes(bboxes)
        self.current_roi_captions = roi_captions
        self.current_roi_ltrb = bboxes

        return segment_ids

    def save_segments(self, segment_ids, segment_path, gt_file):
        """ Saves generated segments and their associated groundtruth
        """
        gt_img, gt_captions = self.crop_page_boxes(constant_width=self.constant_width)
        samples = len(gt_img)
        for i in range(samples):
            img_name = "{}.png".format(segment_ids[i])
            gt_line = "{}\t{}\n".format(img_name, gt_captions[i])
            gt_file.write(gt_line)
            save_image_float(gt_img[i], os.path.join(segment_path, img_name))

    def get_paragraph_ltrb(self):
        """ Returns the bounding box of the whole page text
        """
        minx = np.min(self.current_grapheme_ltrb[:, 0])
        miny = np.min(self.current_grapheme_ltrb[:, 1])
        maxx = np.max(self.current_grapheme_ltrb[:, 2])
        maxy = np.max(self.current_grapheme_ltrb[:, 3])
        return [minx, miny, maxx, maxy]

    def generate_page(self, font, text):
        """ Synthesizes a page with the given parameters
        """
        self.current_font_name = font
        self.current_page_caption = text
        self.current_img, self.current_page_caption, self.current_grapheme_ltrb, self.page_height, self.page_width, self.text_layout = self.render_page_text()
        self.current_page_caption = u''.join(self.current_page_caption)

        # distortions need to be defined here because they now depend on page size,
        # and page size can't be known before rendering the text, since it gets adjusted
        self.distort_operations = [
                                    InkDegradation(self.current_img.shape,
                                                   stain_density=2e-5, stain_size=self.letter_height // 13, stain_roughness=1,
                                                   erasing_density=2e-4, erasing_size=self.letter_height // 16, erasing_roughness=2),
                                    ImageBackground(self.bg_paths, resize_mode='scale', blend_mode='alpha', alpha=0.7),
                                    DocumentNoise([self.page_height, self.page_width]),
                                    GeometricPaperWrapper(page_size=[self.page_height, self.page_width],
                                                          max_y_disp=self.letter_height / 8.0, num_points=6),
                                    GeometricClipper((0, 0, self.page_width, self.page_height))
                                  ]

        self.distort_page()


    def distort_words(self, wiggliness=2):
        """ Performs elastic distortion on each word.
        This could probably be done in a better way, such that the geometry
        of the whole image is calculated first and then sampled all at once.
        """
        # gets word bounding boxes
        ranges = np.array([[w["start"], w["end"]] for w in self.generate_word_segment_data()], dtype='int32')
        _, bboxes = self.stitch_ranges(self.current_page_caption, ranges, self.current_grapheme_ltrb)
        # applies the distortion to each word
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            word = self.current_img[y1:y2, x1:x2]
            if word.size > 0:
                pil_im = Image.fromarray(word)
                w, h = pil_im.size
                dist = ElasticDistortion((w, h), max(w // 12, 2), max(h // 14, 2), wiggliness)
                im_dist, _ = dist.apply_on_image(pil_im, None)
                im = np.array(im_dist)
                self.current_img[y1:y2, x1:x2] = im
        # corrects ranges of the image (caused by the elastic distortion)
        self.current_img[self.current_img > 1] = 1
        self.current_img[self.current_img < 0] = 0

    def distort_page(self):
        """Takes a new page and applies all defined distortions
        """

        # save_image_float(1-self.current_img, "raw.png")

        # Elastic distortion is word level, so it's not obvious how it should be integrated
        # with the rest of the distortions. For the time being, it's done separately here.
        self.distort_words()
        
        # applies all the distortions defined in the pipeline
        img, bboxes = self.current_img, self.current_grapheme_ltrb
        for i, operation in enumerate(self.distort_operations):
            img, bboxes = operation.apply_on_image(img, bboxes)
            # save_image_float(1-img, "dist_pipeline_{}.png".format(i))

        self.current_grapheme_ltrb = bboxes.copy()
        self.current_img = img.copy()

    def plot_current_page(self, pause=True):
        """Auxiliary method for plotting the page interactively.

        :param pause:
        :return:
        """
        plt.plot(self.current_roi_ltrb[:, [0, 0, 2, 2, 0]].T,
                 self.current_roi_ltrb[:, [1, 3, 3, 1, 1]].T)
        plt.imshow(self.current_img, cmap='gray', vmin=0.0, vmax=1.0)
        plt.ylim((self.current_img.shape[0], 0));
        plt.xlim((0, self.current_img.shape[1]));
        plt.plot(self.current_roi_ltrb[:, [0, 0, 2, 2, 0]].T,
                 self.current_roi_ltrb[:, [1, 3, 3, 1, 1]].T)
        if pause:
            plt.show()

    def get_visible_idx(self, caption):
        """Given a string, returns a boolean numpy array with True for visible characters and False for whitespace.

        :param caption:
        :return:
        """
        return np.array([c not in string.whitespace for c in caption], dtype='bool')

    def get_letter_idx(self, caption):
        """Given a string, returns a boolean numpy array with True for letter and digit characters and False for symbols.
        """
        symbols = set(string.printable) - set(string.letters) - set(string.digits)
        return np.array([c not in symbols for c in caption], dtype='bool')

    def generate_word_segment_data(self):
        """ Gets the index ranges of each word in the current page text
        """
        bin_array = self.get_letter_idx(self.current_page_caption)
        lengths = runLengthEncoding(bin_array, bin_array[0])
        segment_data = []
        curr_char_index = 0
        curr_word_index = 0
        for n in lengths:
            caption = self.current_page_caption[curr_char_index:curr_char_index+n]
            left_whitespace = len(caption) - len(caption.lstrip())
            #print left_whitespace
            stripped_caption_length = len(caption.strip())
            start = curr_char_index + left_whitespace
            end = start + stripped_caption_length
            if end-start > 0:
                segment_data.append({"start": start, "end": end})
                curr_word_index += 1
            curr_char_index += n
        return segment_data

    def generate_line_segment_data(self):
        """ Gets the index ranges of each line in the current page text
        """
        nlines = self.text_layout.get_line_count()
        segment_data = []
        for i in range(nlines):
            start = self.text_layout.get_line(i).start_index
            end = start + self.text_layout.get_line(i).length
            segment_data.append({"start": start, "end": end})
        return segment_data

    def stitch_ranges(self, caption, range_array, char_ltrb):
        """Takes a string, its respective bounding boxes, and ranges of all the substrings and provides bounding boxes
        of the substrings.

        :param caption: A string of length N
        :param range_array: A list of tuples containing tuples with the substring beginings and ends of length M
        :param char_ltrb: An int32 numpy array of size [N,4]
        :return: A tuple with a numpy array of size [M] of objects containing the substrings and an int32 numpy array
        of size [M,4] containing the LTRB bounding boxes of the respective substrings.
        """
        range_captions = np.empty(range_array.shape[0], dtype='object')
        range_ltrb = np.empty([range_array.shape[0], 4], dtype='int32')

        for n, ranges in enumerate(range_array):
            range_captions[n] = caption[ranges[0]:ranges[1]].strip()
            range_ltrb[n, :] = char_ltrb[ranges[0]:ranges[1], 0].min(), \
                               char_ltrb[ranges[0]:ranges[1], 1].min(), \
                               char_ltrb[ranges[0]:ranges[1], 2].max(), \
                               char_ltrb[ranges[0]:ranges[1], 3].max()

        return range_captions, range_ltrb

    def distort_bboxes(self, bboxes, x_factor=0.13, y_factor=0.07):
        """ Distorts the bounding boxes by randomly expanding or contracting them.
            Doesn't make sure that the bboxes don't go out of bounds.
        """
        x_disp = int(x_factor * self.letter_height)
        y_disp = int(y_factor * self.letter_height)
        random_disp = np.random.uniform(-1, 1, (len(bboxes), 4))
        random_disp[:, [0, 2]] *= x_disp
        random_disp[:, [1, 3]] *= y_disp
        return bboxes + random_disp

    def dilate_bboxes(self, bboxes, x_factor=0.1, y_factor=0.04):
        """ Dilates the bounding boxes by the specified factors.
            Doesn't make sure that the bboxes don't go out of bounds.
        """
        x_disp = int(x_factor * self.letter_height)
        y_disp = int(y_factor * self.letter_height)
        return bboxes + np.array([-x_disp, -y_disp, x_disp, y_disp])

    """ Given the segment bounding boxes, crops them and associates them with their corresponding text.
    param: gt_str_list: Custom labels for each segment. If None, the current labels will be used.
    param: constant_width: If >0, each letter of the segment will, on average, have the specified width,
                           assuming an image height of letter_height pixels.
    """
    def crop_page_boxes(self, gt_str_list=None, constant_width=0):

        if gt_str_list is None:
            gt_str_list = self.current_roi_captions

        img, bboxes = self.current_img, self.current_roi_ltrb.astype("int32")

        # makes sure no bounding boxes are out of bounds (can happen when dilating and distorting them)
        box_l = bboxes[:, 0]
        box_l[box_l < 0] = 0
        box_t = bboxes[:, 1]
        box_t[box_t < 0] = 0
        box_r = bboxes[:, 2]
        box_r[box_r >= img.shape[1]] = img.shape[1]
        box_b = bboxes[:, 3]
        box_b[box_b >= img.shape[0]] = img.shape[0]
        
        img_list = []
        caption_list = []
        byte_img = (img * 255).astype("uint8")
        
        for n, ltrb in enumerate(bboxes):
            word = byte_img[box_t[n]:box_b[n], box_l[n]:box_r[n]]
            if constant_width:
                chars = len(gt_str_list[n])
                width_per_char = word.shape[1] / float(chars)
                ratio = constant_width / float(width_per_char)
                # print "wpc:", width_per_char
                # print "ratio 1:", ratio
                # takes vertical resizing into account
                ratio *= word.shape[0]/float(self.letter_height)
                # print "ratio 2:", ratio
                word = cv2.resize(word, (int(word.shape[1]*ratio), word.shape[0]), interpolation=cv2.INTER_NEAREST)
                # print "wpc:", word.shape[1] / float(chars)
            img_list.append(word)
            caption_list.append(gt_str_list[n])

        return img_list, caption_list

class CorporaSynthesizer(Synthesizer):
    """ Class which synthesizes a set amount of pages consuming text from the provided corpora.
    """
    def __init__(self, letter_height, words, lines, out_path, bg_paths, font_list, constant_width, distort_bboxes,
                       page_width=0, chars_per_page=1000):
        super(CorporaSynthesizer, self).__init__(letter_height, words, lines, out_path, bg_paths, font_list, constant_width, distort_bboxes)
        if page_width:
            self.image_width = page_width
        else:
            self.image_width = letter_height*30
        self.corpora = OcrCorpus.create_iliad_corpus(lang='eng')
        self.chars_per_page = chars_per_page

    def render_page_text(self):
        """ Implements method of base class.
        """
        return render_text(text=self.current_page_caption,
                           font_name=self.current_font_name,
                           font_height=self.letter_height,
                           image_width=self.image_width,
                           crop_edge_ltrb=self.crop_edge_ltrb,
                           alignment=self.font_alignment,
                           adjust_image_width=False,
                           adjust_image_height=True
                           )

    def generate_random_page(self):
        # reders page
        font = np.random.choice(self.font_list)
        text = self.corpora.read_str(self.chars_per_page)
        self.generate_page(font, text)

        # saves page
        form_id = "form_{}".format(self.page_count)
        save_image_float(1 - self.current_img, os.path.join(self.forms_path, "{}.png".format(form_id)))
        # save groundtruth
        with open_txt(os.path.join(self.form_gt_path, "{}.gt.txt".format(form_id)), "w") as f:
            f.write(text)
        # saves parameters
        with open_txt(os.path.join(self.params_path, "{}.params.txt".format(form_id)), "w") as f:
            f.write("Font:\t{}".format(font))

        # optionally saves segments
        if self.words:
            self.save_segments(self.generate_segments(self.generate_word_segment_data(), suffix="w"), self.words_path, self.gt_words_file)
        if self.lines:
            self.save_segments(self.generate_segments(self.generate_line_segment_data(), suffix="l"), self.lines_path, self.gt_lines_file)



        self.page_count += 1


class DatasetSynthesizer(Synthesizer):
    """
    Class which synthesizes predetermined forms imitating different author styles.
    The segmentation of words and lines must also be predetermined.
    Useful to make synthetic clones of datasets.
    """
    def __init__(self, letter_height, words, lines, out_path, bg_paths, font_list, constant_width, distort_bboxes, dataset_path):
        super(DatasetSynthesizer, self).__init__(letter_height, words, lines, out_path, bg_paths, font_list, constant_width, distort_bboxes)
        self.load_forms(dataset_path)
        self.generate_author_parameters()

    def render_page_text(self):
        """ Implements method of base class.
        """
        return render_text(text=self.current_page_caption,
                           font_name=self.current_font_name,
                           font_height=self.letter_height,
                           crop_edge_ltrb=self.crop_edge_ltrb,
                           alignment=self.font_alignment,
                           adjust_image_width=True,
                           adjust_image_height=True
                           )

    def load_forms(self, data_path):
        """ Loads all forms and all segmentation information associated with them.
        """
        self.author_list = []
        for author in os.listdir(data_path):
            author_obj = Author(author)
            self.author_list.append(author_obj)
            for file in os.listdir(os.path.join(data_path, author)):
                if file.endswith(".form"):
                    form_id = remove_extension(file)
                    with open_txt(os.path.join(data_path, author, file), "r") as form_file:
                        form = Form(form_id, form_file.read())
                        author_obj.add_form(form)
                        # loads word data
                        with open_txt(os.path.join(data_path, author, form_id + ".words"), "r") as words_file:
                            word_data = []
                            for line in words_file:
                                _id, start, end = line.rstrip().split(",")
                                word_data.append({"id": _id, "start": int(start), "end": int(end)})
                            form.word_data = word_data
                        # loads line data
                        with open_txt(os.path.join(data_path, author, form_id + ".lines"), "r") as lines_file:
                            line_data = []
                            for line in lines_file:
                                _id, start, end = line.rstrip().split(",")
                                line_data.append({"id": _id, "start": int(start), "end": int(end)})
                            form.line_data = line_data

    def generate_author_parameters(self):
        """ Generates and saves to a file all the parameters associated with each author.
        """

        # assigns a random font to each author
        # guarantees each font will be used at least once (if authors >= fonts)
        author_font_list = np.array(self.font_list)
        np.random.shuffle(author_font_list)

        diff = len(self.author_list) - len(self.font_list)

        if diff > 0:
            extra_fonts = np.random.choice(self.font_list, diff)
            author_font_list = np.concatenate((author_font_list, extra_fonts))

        for author, font in zip(self.author_list, author_font_list.tolist()):
            author.font = font

        # saves the parameters for future reference
        for author in self.author_list:
            with open_txt(os.path.join(self.params_path, "{}.params.txt".format(author.id)), "w") as f:
                f.write("Font:\t{}".format(author.font))

    def generate_author_forms(self, author):   
        """ Generates all forms of a given author
        """
        # makes output directories
        forms_path = os.path.join(self.forms_path, author.id)
        mkdir_p(forms_path)
        if self.words:
            words_path = os.path.join(self.words_path, author.id)
            mkdir_p(words_path)
        if self.lines:
            lines_path = os.path.join(self.lines_path, author.id)
            mkdir_p(lines_path)

        print "Generating forms of author {}...".format(author.id)

        for form in author.forms:
            print "\t{}".format(form.id)
            # generates the form
            self.generate_page(author.font, form.text)
            save_image_float(1 - self.current_img, os.path.join(forms_path, form.id + ".png"))
            with open_txt(os.path.join(self.form_gt_path, "{}.gt.txt".format(form.id)), "w") as f:
                f.write(form.text)
            # optionally saves form segments
            if self.words:
                self.save_segments(self.generate_segments(form.word_data), words_path, self.gt_words_file)
            if self.lines:
                self.save_segments(self.generate_segments(form.line_data), lines_path, self.gt_lines_file)
            self.page_count += 1


def common_init(load_last_seed, out_path):
    # gets file dir
    src_dir, _ = os.path.split(__file__)

    # creates output dir if it doesn't exist    
    mkdir_p(out_path)

    # mechanism which allows to save the random numpy seed or load the last one
    if load_last_seed:
        print "Loading last seed."
        with open(os.path.join(out_path, "numpy_seed.pkl"), "r") as s:
            np.random.set_state(pickle.load(s))
    else:
        with open(os.path.join(out_path, "numpy_seed.pkl"), "w") as s:
            pickle.dump(np.random.get_state(), s)

    # loads font list
    font_list = []
    with open_txt(os.path.join(src_dir, "data", "font_names.txt"), "r") as f:
        for line in f:
            font_list.append(line.rstrip('\n'))

    # uncomment for font debug
    """
    available_fonts = get_system_fonts()
    self.font_list = list(set(font_names).intersection(set(available_fonts)))

    print "Fonts which were not found:", len(font_names) - len(self.font_list)
    for font in font_names:
        if font not in available_fonts:
            print font
    print set(font_names)
    print set(self.font_list)
    """

    # backgrounds which will be used for pages
    bg_base_path = os.path.join(src_dir, "data", "backgrounds", "white")
    bg_paths = []
    for img in os.listdir(bg_base_path):
        bg_paths.append(os.path.join(bg_base_path, img))

    return font_list, bg_paths


def clone_dataset(letter_height, words, lines, out_path, load_last_seed, constant_width, distort_bboxes, dataset_path):
    font_list, bg_paths = common_init(load_last_seed, out_path)
    # creates synth
    synth = DatasetSynthesizer(letter_height, words, lines, out_path, bg_paths, font_list, constant_width, distort_bboxes, dataset_path)
    # generates all images of all authors
    for author in synth.author_list:
        synth.generate_author_forms(author)
    
    synth.finalize()
    print "Finished!"

def generate_pages(letter_height, words, lines, out_path, load_last_seed, constant_width, distort_bboxes, num_pages):
    font_list, bg_paths = common_init(load_last_seed, out_path)

    synth = CorporaSynthesizer(letter_height, words, lines, out_path, bg_paths, font_list, constant_width, distort_bboxes)

    while synth.page_count < num_pages:
        print "Rendering page {}/{}...".format(synth.page_count+1, num_pages)
        synth.generate_random_page()

    synth.finalize()
    print "Finished!"
