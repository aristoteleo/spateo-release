import sys

import cv2 as cv
import numpy as np


def line_mode(x, y):
    """Click with the left mouse button to draw lines
       on the original image and the mask to distinguish regions

    Args:
        x: x_coordinate where the left mouse button is clicked
        y: y_coordinate where the left mouse button is clicked
    Returns:
        None
    Raises:
        None
    """
    global start, img_copy, line_ls, mask, mask_extend, mask_id, mask_flood, mask_flood_1, img_copy_2
    start = (x, y)
    cv.circle(img_copy, (x, y), 2, (0, 0, 255), -1)  # Draw trace points on the original img
    line_ls.append((x, y))
    for i in range(len(line_ls) - 1):
        cv.line(mask, line_ls[i], line_ls[i + 1], (255, 255, 255), 2)
        cv.line(mask_extend, line_ls[i], line_ls[i + 1], (255, 255, 255), 2)
        cv.line(mask_id, line_ls[i], line_ls[i + 1], 255, 2)
        cv.line(mask_flood, line_ls[i], line_ls[i + 1], 255, 2)
        cv.line(mask_flood_1, line_ls[i], line_ls[i + 1], 255, 2)
        cv.line(img_copy, line_ls[i], line_ls[i + 1], (0, 0, 255), 2)
        if num == "2":
            cv.line(img_copy_2, line_ls[i], line_ls[i + 1], (0, 0, 255), 2)


def drag_mode(x, y):
    """Hold down the left mouse button and move the mouse to draw lines
       on the original image and the mask to distinguish regions
    Args:
        x: x_coordinate where the left mouse button is clicked
        y: y_coordinate where the left mouse button is clicked
    Returns:
        None
    Raises:
        None
    """
    global start, last_pts, mask, mask_extend, mask_id, mask_flood, mask_flood_1, img_copy, img_copy_2, last_pts
    if last_pts == (-1, -1):
        start = (x, y)  # save start point
        last_pts = (0, 0)
        return
    else:
        pt = (x, y)  # current point
        cv.line(mask, start, pt, (255, 255, 255), 2)
        cv.line(mask_extend, start, pt, (255, 255, 255), 2)
        cv.line(mask_id, start, pt, 255, 2)
        cv.line(mask_flood, start, pt, 255, 2)
        cv.line(img_copy, start, pt, (0, 0, 255), 2)
        if num == "2":
            cv.line(mask_flood_1, start, pt, 255, 2)
            cv.line(img_copy_2, start, pt, (0, 0, 255), 2)
        start = pt  # save last point


def mask_fill(x, y, fill_mode):
    """Add different labels and colors to the divided areas according to fill_mode on the mask
    Args:
        x: x_coordinate where the left mouse button is clicked
        y: y_coordinate where the left mouse button is clicked
        fill_mode: The way the region is filled with color ,
                   optional mode : manual(Click a region to fill it),auto(Fill all areas with one mouse click)
    Returns:
        None
    Raises:
        None
    """
    global mask_id, img_copy_2, mask, mask_flood, mask_flood_1, color_index, fill, labels
    ret, thresimg = cv.threshold(mask_id, 0, 255, cv.THRESH_BINARY_INV)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(thresimg)
    if fill_mode == "manual":
        seed = (x, y)
        cv.floodFill(mask, mask_flood, seed, color[color_index], cv.FLOODFILL_MASK_ONLY)
        if num == "2":
            cv.floodFill(
                img_copy_2, mask_flood_1, seed, color[color_index], (50, 50, 50), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE
            )
        if color_index == (len(color) - 1):
            color_index = 0
        else:
            color_index = color_index + 1

    elif fill_mode == "auto":
        for i in range(centroids.shape[0]):
            x = int(centroids[i][0])
            y = int(centroids[i][1])
            if x < mask.shape[0] and y < mask.shape[1]:
                cv.floodFill(mask, mask_flood, (x, y), color[color_index], cv.FLOODFILL_MASK_ONLY)
                if num == "2":
                    cv.floodFill(
                        img_copy_2,
                        mask_flood_1,
                        (x, y),
                        color[color_index],
                        (150, 150, 150),
                        (150, 150, 150),
                        cv.FLOODFILL_FIXED_RANGE,
                    )
            if color_index == (len(color) - 1):
                color_index = 0
            else:
                color_index = color_index + 1
    row, col = mask.shape[0], mask.shape[1]
    mask_flood = np.zeros([row + 2, col + 2], np.uint8)
    if num == "2":
        mask_flood_1 = np.zeros([row + 2, col + 2], np.uint8)
    fill = False  # Automatically return to line mode after filling


def mouse_event(event, x, y, flags, param):
    """
    Mouse event in response to the "drawing" window
    """
    global fill, draw_mode, fill_mode
    if event == cv.EVENT_LBUTTONDOWN:
        if fill == False and draw_mode == "line":  # line mode
            line_mode(x, y)
        elif fill == True:  # fill mode
            mask_fill(x, y, fill_mode)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        if draw_mode == "drag":  # drag mode
            drag_mode(x, y)


def readData(filepath="/home/liaoxiaoyan/opencv_test/img_segement/"):
    global num, draw_mode, fill_mode

    if num == "1":
        filename = sys.argv[2]
        img = cv.imread(filepath + filename)
        filename_mask = sys.argv[3]  # contour images
        img_mask = cv.imread(filepath + filename_mask, 0)
        draw_mode = sys.argv[4]
        fill_mode = sys.argv[5]
        return img, img_mask

    elif num == "2":
        filename = sys.argv[2]
        img = cv.imread(filepath + filename)
        filename_2 = sys.argv[3]
        img_2 = cv.imread(filepath + filename_2)
        filename_mask = sys.argv[4]
        img_mask = cv.imread(filepath + filename_mask, 0)
        draw_mode = sys.argv[5]
        fill_mode = sys.argv[6]
        return img, img_2, img_mask


def draw_init(img, img_2, img_mask):
    global num, img_copy, img_copy_2, mask, mask_id, mask_extend, mask_flood, mask_flood_1
    contours_all, hierarchy = cv.findContours(img_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    img_copy = img.copy()
    cv.drawContours(img_copy, contours_all, -1, (0, 0, 255), 2)
    if num == "2":
        img_copy_2 = img_2.copy()
        cv.drawContours(img_copy_2, contours_all, -1, (0, 0, 255), 2)

    row, col = img.shape[0], img.shape[1]
    mask = np.zeros([row, col, 3], np.uint8)
    mask_extend = mask.copy()
    cv.drawContours(mask, contours_all, -1, (255, 255, 255), 2)
    mask_id = np.zeros([row, col], np.uint8)
    cv.drawContours(mask_id, contours_all, -1, 255, 2)
    mask_flood = np.zeros([row + 2, col + 2], np.uint8)
    mask_flood_1 = np.zeros([row + 2, col + 2], np.uint8)
    return contours_all


def img_segmentation(img):
    global labels, mask_id, img_c
    if type(labels) == int:  # no labeled yet
        ret, thresimg = cv.threshold(mask_id, 0, 255, cv.THRESH_BINARY_INV)
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(thresimg)
    else:  # labeled already
        b, g, r = cv.split(img)
        row, col = mask_id.shape[0], mask_id.shape[1]
        for i in range(labels.max()):
            mask_copy = np.zeros([row, col], np.uint8)
            mask_copy[labels == i + 1] = 1
            # cv.imwrite(filepath+"mask_"+str(i+1)+".png",mask_copy)
            bi = b * mask_copy
            gi = g * mask_copy
            ri = r * mask_copy
            img_c = cv.merge([bi, gi, ri])
            cv.imwrite("img_segement" + str(i + 1) + ".png", img_c)
    cv.imshow("img_segement", img_c)


def extend_contours():
    global mask_extend, mask, img_copy, img_copy_2
    mask_gray = cv.cvtColor(mask_extend, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(mask_gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            # EXTEND_WIDTH=10
            x = contours[i][j][0][0]
            y = contours[i][j][0][1]
            cv.circle(mask_gray, (x, y), EXTEND_WIDTH, 255, -1)

    contours_2, hierarchy = cv.findContours(mask_gray, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    cv.drawContours(mask, contours_2, -1, (255, 0, 0), 2)
    cv.drawContours(img_copy, contours_2, -1, (255, 0, 0), 2)
    if num == "2":
        cv.drawContours(img_copy_2, contours_2, -1, (255, 0, 0), 2)


def fill_mask_color():
    """
    fill mask with different color in different region
    """
    global fill, line_ls
    fill = True
    line_ls = []


def save_draw():
    global mask
    cv.imwrite("mask.png", mask)


def add_contours(img):
    global line_ls, mask_flood, mask_flood_1, fill, last_pts
    line_ls = []
    row, col = img.shape[0], img.shape[1]
    mask_flood = np.zeros([row + 2, col + 2], np.uint8)
    mask_flood_1 = np.zeros([row + 2, col + 2], np.uint8)
    fill = False
    last_pts = (-1, -1)


def clear(img, img_2, contours_all):
    global line_ls, fill, img_copy, img_copy_2, mask, mask_extend, mask_id, mask_flood, mask_flood_1, last_pts, img_c
    line_ls = []
    fill = False
    last_pts = (-1, -1)

    img_copy = img.copy()
    cv.drawContours(img_copy, contours_all, -1, (0, 0, 255), 2)
    if num == "2":
        img_copy_2 = img_2.copy()
        cv.drawContours(img_copy_2, contours_all, -1, (0, 0, 255), 2)
    row, col = img.shape[0], img.shape[1]
    img_c = np.zeros([row, col, 3], np.uint8)
    mask = np.zeros([row, col, 3], np.uint8)
    mask_extend = mask.copy()
    cv.drawContours(mask, contours_all, -1, (255, 255, 255), 2)
    mask_id = np.zeros([row, col], np.uint8)
    cv.drawContours(mask_id, contours_all, -1, 255, 2)
    mask_flood = np.zeros([row + 2, col + 2], np.uint8)
    mask_flood_1 = np.zeros([row + 2, col + 2], np.uint8)


# global variable
color = [
    (192, 182, 255),
    (203, 192, 255),
    (60, 20, 220),
    (245, 240, 255),
    (147, 112, 219),  # b,g,r
    (180, 105, 255),
    (147, 20, 255),
    (133, 21, 199),
    (214, 112, 218),
    (216, 191, 216),
    (221, 160, 221),
    (238, 130, 238),
    (255, 0, 255),
    (255, 0, 255),
    (139, 0, 139),
    (128, 0, 128),
    (211, 85, 186),
    (211, 0, 148),
    (204, 50, 153),
    (130, 0, 75),
    (226, 43, 138),
    (219, 112, 147),
    (238, 104, 123),
    (205, 90, 106),
    (139, 61, 72),
    (250, 230, 230),
    (255, 248, 248),
    (255, 0, 0),
    (205, 0, 0),
    (112, 25, 25),
    (139, 0, 0),
    (128, 0, 0),
    (225, 105, 65),
    (237, 149, 100),
    (222, 196, 176),
    (153, 136, 119),
    (144, 128, 112),
    (255, 144, 30),
    (255, 248, 240),
    (180, 130, 70),
    (250, 206, 135),
    (235, 206, 135),
    (255, 191, 0),
    (230, 216, 173),
    (230, 224, 176),
    (160, 158, 95),
    (255, 255, 240),
    (255, 255, 225),
    (238, 238, 175),
    (255, 255, 0),
    (209, 200, 0),
    (79, 79, 47),
    (139, 139, 0),
    (128, 128, 0),
    (204, 209, 72),
    (170, 178, 32),
    (208, 224, 64),
    (170, 255, 127),
    (154, 250, 0),
    (127, 255, 0),
    (250, 255, 245),
    (113, 179, 60),
    (87, 139, 46),
    (240, 255, 240),
    (144, 238, 144),
    (152, 251, 152),
    (143, 188, 143),
    (50, 205, 50),
    (0, 255, 0),
    (34, 139, 34),
    (80, 12, 0),
    (0, 100, 0),
    (0, 255, 127),
    (0, 252, 124),
    (47, 255, 173),
    (47, 107, 85),
    (220, 245, 245),
    (210, 250, 250),
    (240, 255, 255),
    (224, 255, 255),
    (0, 255, 255),
    (0, 128, 128),
    (107, 183, 189),
    (205, 250, 255),
    (170, 232, 238),
    (140, 230, 240),
    (0, 215, 255),
    (220, 248, 255),
    (32, 165, 218),
    (240, 250, 255),
    (230, 245, 253),
    (179, 222, 245),
    (181, 228, 255),
    (0, 165, 255),
    (213, 239, 255),
    (205, 235, 255),
    (173, 222, 255),
    (215, 235, 250),
    (140, 180, 210),
    (135, 184, 222),
    (196, 228, 255),
    (0, 140, 255),
    (230, 240, 250),
    (63, 133, 205),
    (185, 218, 255),
    (96, 164, 244),
    (30, 105, 210),
    (19, 69, 139),
    (238, 245, 255),
    (45, 82, 160),
    (122, 160, 255),
    (80, 127, 255),
    (0, 69, 255),
    (122, 150, 233),
    (71, 99, 255),
    (225, 228, 255),
    (114, 128, 250),
    (250, 250, 255),
    (128, 128, 240),
    (143, 143, 188),
    (92, 92, 205),
    (0, 0, 255),
    (42, 42, 165),
    (34, 34, 178),
    (0, 0, 139),
    (0, 0, 128),
]
"""
    EXTEND_WIDTH : Track expansion width, pixel value
    line_ls : Save the mouse click point, the point coordinates(a tuple of coordinate points)
    fill : Distinguish between drawing or filling flags when the mouse is clicked. {'True':fill_mode , 'False':line_mode
    color_index : Index when filling the mask with a color, an iterator
    draw_mode : The way draw trajectories. optional: {'line':line_mode,'drag':drag_mode}
    fill_mode : The way fill mask. optional:{'manual':A single area is manually filled. 'auto': All area is auto filled} 
    mask : The area divided
    mask_extend : The area divided, used when extending the contours
    mask_id : The area divided, use when labeling areas
    mask_flood : Use when fill mask with different color
    mask_flood_1 : Use when fill img_2 with different color
    labels : A diagram of the result after each area has been labeled with different integer
"""
EXTEND_WIDTH = 10
start = last_pts = (-1, -1)
line_ls = []
fill = False
color_index = 0
draw_mode = fill_mode = 0
num = 0
img_copy = img_copy_2 = img_c = 0
mask = mask_extend = mask_id = mask_flood = mask_flood_1 = labels = 0

cv.namedWindow("drawing", 0)
cv.setMouseCallback("drawing", mouse_event)
cv.namedWindow("mask", 0)


def main():
    global num, img_copy, img_copy_2, mask, img_c

    num = sys.argv[1]  # Is the number of input images, excluding contour images
    if num == "2":
        cv.namedWindow("img_2", 0)
        img, img_2, img_mask = readData()  # change num,draw_mode,fill_mode
        contours_all = draw_init(
            img, img_2, img_mask
        )  # init img_copy,img_copy_2,mask,mask_id,mask_extend,mask_flood,mask_flood_1
    elif num == "1":
        img, img_mask = readData()
        contours_all = draw_init(img, 0, img_mask)
    row, col = img.shape[0], img.shape[1]
    img_c = np.zeros([row, col, 3], np.uint8)

    while 1:
        cv.imshow("drawing", img_copy)
        if num == "2":
            cv.imshow("img_2", img_copy_2)
        cv.imshow("mask", mask)

        k = cv.waitKey(1) & 0xFF
        if k == ord("q"):  # press 'q' to exit
            break
        elif k == ord("s"):  # press 's' to save mask
            save_draw()
        elif k == ord("i"):  # press 'i' to split the original image
            cv.namedWindow("img_segement", 0)
            # cv.imshow("img_segement",img_c)
            img_segmentation(img)
        elif k == ord("p"):  # press 'p' to extend contours
            extend_contours()
        elif k == ord("r"):  # press 'r' to fill mask with different colors and labels
            fill_mask_color()
        elif k == ord("b"):  # press 'b' to Continue drawing tracks on the same mask
            add_contours(img)
        elif k == ord("c"):  # press 'c' to clear all
            if num == "2":
                clear(img, img_2, contours_all)
            elif num == "1":
                clear(img, 0, contours_all)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
