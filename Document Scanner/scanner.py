import cv2

import numpy as np

read_img = cv2.imread("doc18.jpg")

if read_img.shape[0] > read_img.shape[1]:

    img_width = 2480

    img_height = 3508

elif read_img.shape[0] < read_img.shape[1]:

    img_width = 3508

    img_height = 2480


def pre_process(image):

    img_gy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_br = cv2.GaussianBlur(img_gy, (5, 5), 3)

    img_cy = cv2.Canny(img_br,100,200)
    
    kernel = np.ones((4, 4))
   
    img_di = cv2.dilate(img_cy, kernel, iterations=2)

    img_th = cv2.erode(img_di, kernel, iterations=1)

    return img_th
 
 
def counters(image):

    biggest_cnt = np.array([])

    max_area = 0

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 2000:

            pm = cv2.arcLength(cnt, True)

            corner = cv2.approxPolyDP(cnt, 0.02*pm, True)

            if area > max_area and len(corner) == 4:

                biggest_cnt = corner

                max_area = area

    t = cv2.drawContours(img_cnt, biggest_cnt, -1, (0, 0, 255), 150)

    return biggest_cnt


def reorder(c_pts):

    c_pts = c_pts.reshape((4, 2))

    new_c_pts = np.zeros((4, 1, 2), np.int32)

    add = c_pts.sum(1)

    new_c_pts[0] = c_pts[np.argmin(add)]

    new_c_pts[3] = c_pts[np.argmax(add)]

    diff = np.diff(c_pts, axis=1)

    new_c_pts[1] = c_pts[np.argmin(diff)]

    new_c_pts[2] = c_pts[np.argmax(diff)]

    print(new_c_pts)

    return new_c_pts


def warp(image, biggest_cnt):

    biggest_cnt = reorder(biggest_cnt)

    pts1 = np.float32(biggest_cnt)

    pts2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])

    print(pts2)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    img_trs = cv2.warpPerspective(image, matrix, (img_width, img_height))

    img_crp = img_trs[40:img_trs.shape[0]-20, 10:img_trs.shape[1]-10]

    return img_crp



img_cnt = read_img.copy()

img_rt1 = pre_process(read_img)

big_cnt = counters(img_rt1)

img_crop = warp(read_img, big_cnt)

img_f_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

thresh, scan_img = cv2.threshold(img_f_gray, 120, 255, cv2.THRESH_BINARY)

cv2.imwrite("result1.png", scan_img)

cv2.imwrite("result2.png", img_rt1)

cv2.imwrite("result3.png", img_cnt)

cv2.imwrite("result4.png", img_crop)


cv2.waitKey(0)

