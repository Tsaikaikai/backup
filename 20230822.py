import sensor, image
from pyb import Pin


def DoImageProcess(frame, anchors, pi7, pi6):
    standard_length = 300
    kernel_size = 1
    kernel = [1, 0, -1, \
                      2, 0, -2, \
                      1, 0, -1]
    anchor_count = 5
    min_angle = 30

    img = sensor.snapshot()

    if (img.get_statistics().mean() < 10):
        print(frame, 6, -1, 0, 0, 0, 0, 0, 0, 0)
        return img, anchors

    resetFlag = pi6.value()
    if(resetFlag == 1):
        print(frame, 2, -1, 0, 0, 0, 0, 0, 0, 0)
        anchors = []
        return img, anchors

    #jumpFlag = pi7.value()
    jumpFlag = 0
    if (jumpFlag == 1):
        print(frame, 3, -1, 0, 0, 0, 0, 0, 0, 0)
        return img, anchors

    #img.morph(kernel_size, kernel)
    img.erode(2)
    img.lens_corr(1.5)
    all_lines = img.find_lines(threshold = 500)
    if (len(all_lines) < 1):
        print(frame, 4, -1, 0, 0, 0, 0, 0, 0, 0)
        return img, anchors

    if (len(anchors) >= anchor_count) :
        anchors_x, anchors_angle, anchors_m, anchors_lsq = GetAnchors(anchors, standard_length)
        anchors_x = int(anchors_x)
        select_line = SelectLine(all_lines, anchors_x, img.width(), min_angle)
        if select_line == -1:
            print(frame, 5, -1, 0, 0, 0, 0, 0, 0, 0)
            return img, anchors

        del anchors[0]
        select_x = int((select_line.x1() + select_line.x2()) / 2)
        angle = select_line.theta()
        if (angle > 90):
            angle -= 180
        distance = abs(anchors_x - select_x)
        if ( anchors_x > select_x):
            count = round(distance / standard_length)
        else:
            count = -round(distance / standard_length)
        anchors.append((select_x, angle , count))
        select_x += count *standard_length
        print(frame, 1, select_x, angle, count, anchors_m, anchors_lsq, 0, 0, 0)

        img.draw_line(anchors_x, 0, anchors_x, img.height() - 1, color = (255, 0, 0), thickness=3)
        img.draw_line(select_line.x1(), 0, select_line.x2(), img.height() - 1, color = (255, 0, 255),thickness=1)
    else:
        anchors_search = 0
        if (len(anchors) < anchor_count / 2):
            anchors_search = img.width() / 2
        else:
            anchors_search, anchors_angle, anchors_m, anchors_lsq = GetAnchors(anchors, standard_length)
            anchors_search = int(anchors_search)
        select_line = SelectLine(all_lines, anchors_search, img.width(), min_angle)
        if select_line != -1:
            x = int((select_line.x1() + select_line.x2()) / 2)
            angle = select_line.theta()
            if (angle > 90):
                angle -= 180
            anchors.append((x, angle, 0))
            img.draw_line(select_line.line(), color = (0, 255, 0),thickness=3)
        print(frame, 0, -1, 0, 0, 0, 0, 0, 0, 0)
    return img, anchors

def SelectLine(lines, anchors_x, width, min_angle):
    min_distance = 999999
    select_line = -1
    for i in range(len(lines)):
        l = lines[i]
        x = int(( l.x1() + l.x2()) / 2)
        distance = abs(x - anchors_x)
        if (l.theta() > 90):
            angle = 180 - l.theta()
        else:
            angle = l.theta()
        if (distance < min_distance) and (angle < min_angle)  :
            min_distance = distance
            select_line = l
    return select_line

def GetAnchors(anchors, standard_length):
    if len(anchors) < 1:
        return 0, 0
    x = 0
    angle = 0
    avg_x = 0
    avg_y = 0
    avg_xy = 0
    avg_xx = 0
    for i in range(len(anchors)):
        temp = anchors[i][0] + standard_length * anchors[i][2]
        x += temp
        angle += anchors[i][1]
        avg_x += i
        avg_y += temp
        avg_xy += i * temp
        avg_xx += i * i
    x /= len(anchors)
    angle /= len(anchors)
    avg_x /= len(anchors)
    avg_y /= len(anchors)
    avg_xy /= len(anchors)
    avg_xx /= len(anchors)

    m = (avg_xy - avg_x*avg_y) / (avg_xx - avg_x * avg_x)
    b = avg_y - m * avg_x

    lsq = 0
    for i in range(len(anchors)):
        temp = anchors[i][0] + standard_length * anchors[i][2]
        lsq += abs((m * i + b) - temp)
    return x, angle, m, lsq

if __name__ == '__main__':
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.VGA)
    sensor.set_auto_exposure(False, 100000)
    sensor.set_auto_gain(False)
    sensor.set_auto_whitebal(False)
    sensor.set_windowing((540, 320))

    pi7 = Pin("P7",  Pin.IN)
    pi6 = Pin("P6",  Pin.IN)

    frame = -1
    anchors = []
    while (True):
        frame = (frame + 1) % 1000
        img, anchors = DoImageProcess(frame, anchors, pi7, pi6 )
