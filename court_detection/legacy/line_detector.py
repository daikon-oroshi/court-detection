import cv2
import numpy as np
import itertools
from typing import Tuple

def luminance_(img :np.array) -> np.array:
    print(img[0,0])
    #perception = np.array([0.114, 0.587, 0.299])
    perception = np.array([0.364, 0.087, 0.549]) # greenの重みを減らす
    ret = np.einsum("ijk,k->ij", img, perception).astype(np.uint8)
    print(ret[0,0])
    return ret

def luminance(img :np.array) -> np.array:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,1]

def whiteline_detection(
        lum_img: np.array,
        val_th: int,
        diff_th:int,
        rng: int
    ) -> np.array:
    def tmp(arr, x:int, y:int , val_th, diff_th, rng) -> bool:
        if arr[x,y] > val_th:
            #print("higher than " + str(val_th))
            if arr[x,y] - arr[x-rng,y] > diff_th and arr[x,y] - arr[x+rng,y] > diff_th:
                #print("たて")
                return True
            elif arr[x,y] - arr[x,y-rng] > diff_th and arr[x,y] - arr[x,y+rng] > diff_th:
                #print("横")
                return True
            else:
                return False
        else:
            return False

    ret = np.zeros(lum_img.shape, dtype=np.uint8)
    w = lum_img.shape[0]
    h = lum_img.shape[1]
    for i, j in itertools.product(range(rng, w-rng), range(rng, h-rng)):
        ret[i,j] = int(tmp(lum_img, i, j, val_th, diff_th, rng)) * 255
    return ret

def hough(white):
    return cv2.HoughLines(white,1,np.pi/180,250)

def draw_lines(img, lines):
    for l in lines:
        rho,theta = l[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1200*(-b))
        y1 = int(y0 + 1200*(a))
        x2 = int(x0 - 1200*(-b))
        y2 = int(y0 - 1200*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    return img

if __name__ == "__main__":
    import sys
    finename = sys.argv[1]
    img = cv2.imread(finename)
    width,height=1280,720
    img = cv2.resize(img,(width,height))
    img = cv2.bilateralFilter(img,15, 20, 20)
    img = cv2.bilateralFilter(img,15, 20, 20)
    img = cv2.bilateralFilter(img,15, 20, 20)
    #img = cv2.bilateralFilter(img,15, 20, 20)
    #lum_ = luminance_(img)
    lum = luminance(img)

    print(img.shape)
    print(lum.shape)

    white = whiteline_detection(lum, 150, 25, 3)
    hough_lines = hough(white)
    hough_ = draw_lines(img, hough_lines)

    for i in [lum, white, hough_]:

        cv2.imshow('image', i)

        #キーボード入力を受け付ける
        key = cv2.waitKey(0)

        if key == 27:            #escの処理
            cv2.destroyAllWindows()

    image,contours, hierarchy = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_ = np.zeros((height, width), dtype=np.uint8)
    #輪郭の中で面積が最大となる輪郭を検出
    for c in contours:
        epsilon = 0.001 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(img_, [approx], -1, (255, 255, 255), 2)

    #hough_lines = hough(img_)
    #hough_ = draw_lines(img, hough_lines)

    #キーボード入力を受け付ける
    #cv2.imshow('image', hough_)
    #key = cv2.waitKey(0)
    #if key == 27:            #escの処理
    #    cv2.destroyAllWindows()
