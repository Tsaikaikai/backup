import sensor, image, time
import math

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.VGA)
sensor.set_auto_exposure(False, 100000)
sensor.set_auto_whitebal(False)

frame = -1

kernel_size = 1
kernel = [1, 0, -1, \
                 2, 0, -2, \
                 1, 0, -1]

hasAnchor = False
anchor_count = 5
anchors = []

min_angle = 30

while(True):
    frame += 1
    frame %= 1000

    img = sensor.snapshot()
    img.morph(kernel_size, kernel)

    img.lens_corr(1.8)

    all_lines = img.find_lines(threshold = 1000)

    if (len(all_lines) < 1):
        print(frame, -1, 0)
        continue

    if (hasAnchor) :
        anchors_x = 0
        anchors_angle = 0
        for i in range(len(anchors)):
            anchors_x += (anchors[i].x1() + anchors[i].x2()) / 2
            anchors_angle += anchors[i].theta()
        anchors_x /= len(anchors)
        anchors_angle /= len(anchors)

        min_distance = img.width() + 1
        select_line = -1
        for i in range(len(all_lines)):
            l = all_lines[i]
            x = int(( l.x1() + l.x2()) / 2)
            distance = abs(x - anchors_x)

            if (l.theta() > 90):
                angle = 180 - l.theta()
            else:
                angle = l.theta()

            if (distance < min_distance) and (angle < min_angle)  :
                min_distance = distance
                select_line = l

        anchors_x = int(anchors_x)
        img.draw_line(anchors_x, 0, anchors_x, img.height() - 1, color = (255, 0, 0),thickness=3)
        print(frame, anchors_x, anchors_angle)

        if (select_line != -1):
            anchors.append(select_line)
            del anchors[0]

    else:
        anchor_search = 0
        if (len(anchors) < anchor_count / 3):
            anchor_search = img.width() / 2
        else:
            for i in range(len(anchors)):
                anchor_search += (anchors[i].x1() + anchors[i].x2()) / 2
            anchor_search /= len(anchors)
            anchor_search = int(anchor_search)

        min_distance = img.width() + 1
        select_line = -1
        for i in range(len(all_lines)):
            l = all_lines[i]
            x = int(( l.x1() + l.x2()) / 2)
            distance = abs(x - anchor_search)

            if (l.theta() > 90):
                angle = 180 - l.theta()
            else:
                angle = l.theta()

            if (distance < min_distance) and (angle < min_angle)  :
                min_distance = distance
                select_line = l

        img.draw_line(select_line.line(), color = (0, 255, 0),thickness=3)
        anchors.append(select_line)

        print(select_line)

        if (len(anchors) > anchor_count):
            hasAnchor = True

