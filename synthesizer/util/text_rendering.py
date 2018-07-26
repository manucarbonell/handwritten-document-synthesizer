import pango
import pangocairo
import cairo
import numpy as np
import string

def get_system_fonts():
    """ Retrieves all fonts found in the system
    """
    return [f.get_name() for f in pangocairo.cairo_font_map_get_default().list_families()]

def get_text_size(text, font_name, font_height, layout_width, alignment=pango.ALIGN_LEFT, wrap_word=True):
    """ Returns width and height of the text in pixels
    """
    surf = cairo.ImageSurface(cairo.FORMAT_A8, 1, 1)
    context = cairo.Context(surf)
    pangocairo_context = pangocairo.CairoContext(context)
    layout = pangocairo_context.create_layout()
    layout.set_width(layout_width * pango.SCALE)
    font = pango.FontDescription(font_name)
    font.set_absolute_size(font_height * pango.SCALE)
    layout.set_font_description(font)
    layout.set_alignment(alignment)
    if wrap_word:
        layout.set_wrap(pango.WRAP_WORD)
    layout.set_text(text)
    context.set_source_rgb(1, 0, 1.0)
    pangocairo_context.update_layout(layout)
    pangocairo_context.show_layout(layout)

    return layout.get_pixel_size()

def render_text(text=np.array(list("Hello\nWorld")), font_name='times', font_height=64, image_width=1000, image_height=1000,
                crop_edge_ltrb=np.array([10,10,10,10]), alignment=pango.ALIGN_LEFT,
                adjust_font_height=True, adjust_image_width=False, adjust_image_height=True):
    """ Renders the specified text on a canvas.
    Parameters
    ----------
    text : str
        The text to be rendered.
    font_name: str
        Alias for the font to be used to render the text
    font_height: int
        The height, in pixels, of the each line of text
    image_width, image_height: int
        The initial width and height of the page
    crop_edge_ltrb: [int, int, int, int]
        Margins of the page (left, top, right, bottom)
    alignment: pango.ALIGN_*
        Alignment of the text (left, center, right)
    adjust_font_height: bool
        Makes sure that the line height of the rendered text is font_height (Pango apparently doesn't guarantee that)
    adjust_image_width: bool
        If true, the page's width is adjusted so that it fits the text width, maintaining the margins
    adjust_image_height: bool
        If true, the page's height is adjusted so that it fits the text height, maintaining the margins

    Returns
    -------
    numpy.ndarray
        A grayscale image containing the rendered text.
    str
        The text which was rendered.
    numpy.ndarray
        The bounding boxes of each rendered character.
    int
        The height of the canvas in which the text was rendered.
    int
        The width of the canvas in which the text was rendered.
    layout
        A layout object containing information about the rendered text.
    """

    if isinstance(text,str) or isinstance(text,unicode):
        text = np.array(list(text))
    text = u"".join(text.tolist())

    text_width = image_width - (crop_edge_ltrb[0] + crop_edge_ltrb[2])
    text_height = image_height - (crop_edge_ltrb[1] + crop_edge_ltrb[3])
    canvas_width = image_width
    canvas_height = image_height

    if adjust_font_height:
        # sets proper, constant font height (pango should already do this, but it doesn't)
        w, h = get_text_size(string.ascii_letters, font_name, font_height, text_width, wrap_word=False)
        ratio = font_height / float(h)
        font_height = int(font_height*ratio)
        #w, h = get_text_size(string.ascii_letters, font_name, font_height, text_width, wrap_word=False)

    if adjust_image_width:
        max_width = 9999
        w, _ = get_text_size(text, font_name, font_height, max_width, wrap_word=True)
        text_width = w+2
        canvas_width = text_width + (crop_edge_ltrb[0] + crop_edge_ltrb[2])
        # I have no idea why this has to be done, it fixes an error which sometimes comes up
        # I think it always works if width is multiple of 100, but not sure, there's no documentation about this
        canvas_width += 100 - (canvas_width % 100)

    if adjust_image_height:
        _, h = get_text_size(text, font_name, font_height, text_width, wrap_word=True)
        text_height = h+2
        canvas_height = text_height + (crop_edge_ltrb[1] + crop_edge_ltrb[3])

    #print canvas_width, canvas_height

    # renders text
    surf = cairo.ImageSurface(cairo.FORMAT_A8, canvas_width, canvas_height)
    context = cairo.Context(surf)

    context.translate(crop_edge_ltrb[0], crop_edge_ltrb[1])
    pangocairo_context = pangocairo.CairoContext(context)
    layout = pangocairo_context.create_layout()
    layout.set_width(text_width * pango.SCALE)
    font = pango.FontDescription(font_name)
    # print font.get_family()
    font.set_absolute_size(font_height * pango.SCALE)
    layout.set_font_description(font)
    #print "FONT DESCRIPTION:", repr(font.get_family())
    layout.set_alignment(alignment)
    layout.set_wrap(pango.WRAP_WORD)
    layout.set_text(text)
    # print u"".join(text.tolist())
    context.set_source_rgb(1, 0, 1.0)
    pangocairo_context.update_layout(layout)
    pangocairo_context.show_layout(layout)
    # pangocairo_context.paint()
    surf.flush()

    buf = surf.get_data()
    np_img = np.frombuffer(buf, np.uint8).reshape(
        [canvas_height, canvas_width])

    char_ltwh = np.array(
        [layout.index_to_pos(k) for k in range(len(text))]) / pango.SCALE
    char_ltrb = char_ltwh.copy()
    char_ltrb[:, 2:] += char_ltwh[:,:2]
    char_ltrb[:, [0, 2]] += (crop_edge_ltrb[0])
    char_ltrb[:, [1, 3]] += (crop_edge_ltrb[1])

    w, h = layout.get_pixel_size()
    
    if h > text_height:
        print(h, text_height)
        raise RuntimeError("Page is too small for text. Text: {t}, Page: {p}".format(t=h, p=text_height))
        # inside_height = text_height >= char_ltrb[:,3]
        # if not inside_height.any():
        #     raise RuntimeError("Page is too small for text.")
        # text = text[inside_height]
    
    #print w, h
    #print image_width, image_height    
    
    return 1-(np_img/255.0), text, char_ltrb, canvas_height, canvas_width, layout

