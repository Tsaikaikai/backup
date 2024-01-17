import cv2
import numpy as np

def distortion(distorted_img):
    camera_matrix = np.array([[360, 0, 320],[0, 360, 240],[0, 0, 1]])
    distortion_coefficients = np.array([-440/10000, -300/10000, 0, 0, 0])
    undistorted_img = cv2.undistort(distorted_img, camera_matrix, distortion_coefficients)
    return undistorted_img

cap = cv2.VideoCapture(0)

while True:
    ret, distorted_img = cap.read()

    # 校正前的圖像
    cv2.imshow('distorted Image', distorted_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    undistorted_img = distortion(distorted_img)

    # 校正後的圖像
    cv2.imshow('Undistorted Image', undistorted_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
