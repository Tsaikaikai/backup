import cv2
import numpy as np
import time

class Camera:
    def __init__(self, para):
        self._index = para['index']
        self._count = 0
        self._anchors = []

        self._live = para['live']

        self._mean_error_min = para['mean_error_min']
        self._mean_error_max = para['mean_error_max']

        self._anchor_count = para['anchor_count']
        self._min_angle = para['min_angle']
        self._gap = para['gap']

        self._frame_width = para['frame_width']
        self._frame_height = para['frame_height']
        self._image_width = para['image_width']
        self._image_height = para['image_height']

        self._open()

    def release(self):
        self._close()

    def get_result(self, reset=False, save_path=''):
        error_code = 1
        pos = 0
        mean = 0.0
        focus = 0.0
        ang = 0.0
        if not self.cap.isOpened():
            self._close()
            self._open()
            return self._return_result(code=-1)

        ret, frame = self.cap.read()
        if ret is False:
            self._close()
            self._open()
            return self._return_result(code=-1)

        self._count = (self._count + 1) % 1000
        #distortion_frame = self._distortion(frame)
        source = cv2.resize(frame,
                            (self._image_width, self._image_height),
                            interpolation = cv2.INTER_AREA)
        if save_path != '':
            cv2.imwrite(save_path + '_s.jpg', source)

        mean = source.mean()

        if reset is True:
            self._anchors.clear()
            return self._return_result(code=2, reset=reset)

        if mean < self._mean_error_min:
            return self._return_result(code=6, mean=mean)

        if mean > self._mean_error_max:
            return self._return_result(code=7, mean=mean)

        processed, lines, focus = self._find_lines(source)
        if save_path != '':
            cv2.imwrite(save_path + '_p.jpg', processed)

        error_code, pos, line_x, ang = self._find_position(lines)

        output = source.copy()
        if error_code == 1:
            cv2.line(output,
                     (pos, 0), (pos, self._image_height),
                     (255, 0, 255), 2)
            temp_length = int(self._image_height / 2)
            cv2.line(output,
                     (line_x, temp_length - 100), (line_x, temp_length + 100),
                     (0, 0, 255), 2)

        if save_path != '':
            cv2.imwrite(save_path + '_o.jpg', output)

        if self._live:
            cv2.imshow('o', output)
            cv2.waitKey(1)

        del source
        del processed
        del output

        return self._return_result(
            code=error_code, location=pos, mean=mean,
            focus=focus, reset=reset, angle=ang)

    def _open(self):
        self.cap = cv2.VideoCapture(self._index)
        print(self._index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)

    def _close(self):
        self.cap.release()
        time.sleep(0.1)

    def _distortion(self, img):
        camera_matrix = np.array([[320, 0, 320],[0, 320, 180],[0, 0, 1]])
        distortion_coefficients = np.array([-580/10000, -365/10000, 0, 0, 0])
        undistorted_img = cv2.undistort(img, camera_matrix, distortion_coefficients)
        return undistorted_img

    def _find_lines(self, img):
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize = (5, 1))
        output = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 1), 0)
        opened = cv2.morphologyEx(blur, cv2.MORPH_OPEN, open_kernel)
        sobel = cv2.Sobel(opened, -1, 1, 0, 3, 3)
        ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img2 = binary.copy()
        skel = binary.copy()
        skel[:, :] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        x = []

        focus = sobel.mean()

        while True:
            eroded = cv2.morphologyEx(img2, cv2.MORPH_ERODE, kernel)
            dilate = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            sub = cv2.subtract(img2, dilate)
            skel = cv2.bitwise_or(skel, sub)
            img2[:, :] = eroded[:, :]

            if cv2.countNonZero(img2) == 0:
                break

        lines = cv2.HoughLines(skel, 1, np.pi/180, 120)
        if lines is None:
            return output, 0, focus
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 1)
                x.append((x1, y1, x2, y2))

        del gray
        del sobel
        del blur
        del eroded
        del binary
        del img2
        del skel

        return output, x, focus

    def _find_position(self, lines):
        if lines is 0:
            return 4, 0, 0

        anchors_search = self._get_anchors()

        min_distance = 9999999
        select_line = 0
        angle = 0
        for i in range(len(lines)):
            line = lines[i]
            x = int((line[0] + line[2]) / 2)
            angle = self._angle(line)
            distance = abs(x - anchors_search)

            if distance < min_distance and abs(angle) < self._min_angle:
                min_distance = distance
                select_line = line
                select_angle = angle

        if not select_line:
            return 5, 0, 0, 0

        pos = int((select_line[0] + select_line[2]) / 2)
        line_x = pos

        if len(self._anchors) > self._anchor_count:
            del self._anchors[0]

        jump_dist = anchors_search - pos
        jump_count = round(jump_dist / self._gap)

        pos += jump_count * self._gap
        self._anchors.append(pos)

        return 1, pos, line_x, select_angle

    def _get_anchors(self):
        if not len(self._anchors):
            return self._image_width / 2

        return sum(self._anchors) / len(self._anchors)

    def _angle(self, line):
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        if x1 == x2:
            return 0
        if y1 == y2:
            return 90

        k = (y2 - y1) / (x2 - x1)
        ang = 90 - (np.arctan(k) * 180.0 / np.pi)
        if ang > 90:
            ang = ang - 180
        if ang < -90:
            ang = ang + 180

        return round(ang,3)

    def _return_result(self,
                       code=-1, location=0, mean=0, focus=0, reset=False, angle=0):
        result = dict()
        result['status'] = code
        result['frame'] = self._count
        result['location'] = location
        result['angle'] = angle
        result['data0'] = mean
        result['data1'] = focus
        result['data2'] = '0'
        result['data3'] = '0'
        result['data4'] = '0'
        result['data5'] = reset

        return result
