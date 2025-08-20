# uncompyle6 version 3.9.2
# Python bytecode version base 3.5 (3350)
# Decompiled from: Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)]
# Embedded file name: E:\Project\Proj-17-03\x64\Release\deeplearning.py
# Compiled at: 2017-11-21 10:39:27
# Size of source mod 2**32: 29540 bytes
import os, math, random, numpy as np, tensorflow as tf
from PIL import Image
from shutil import copyfile

class DeepLearning(object):

    def __init__(self, size_x=150, size_y=150, channel=1, class_num=2):
        tf.Session(tf.reset_default_graph())
        self._IMAGE_SIZE_X = size_x
        self._IMAGE_SIZE_Y = size_y
        self._IMAGE_CHANNEL = channel
        self._CLASS_NUM = class_num
        self._learning_rate = 0.0001
        self._batch_num = 100
        self._epoch_num = 150
        self._xs = tf.placeholder(tf.float32, [
         None, self._IMAGE_SIZE_Y,
         self._IMAGE_SIZE_X, self._IMAGE_CHANNEL])
        self._ys = tf.placeholder(tf.float32, [None, self._CLASS_NUM])
        self._show_detial = False
        self._save_image = False
        self._model_builded = False
        self._ckpt_loaded = False
        self._sess = tf.Session()

    @property
    def rate(self):
        return self._learning_rate

    @rate.setter
    def rate(self, value):
        self._learning_rate = value

    @property
    def batch(self):
        return self._batch_num

    @batch.setter
    def batch(self, value):
        self._batch_num = value

    @property
    def epoch(self):
        return self._epoch_num

    @epoch.setter
    def epoch(self, value):
        self._epoch_num = value

    @property
    def save_image(self):
        return self._save_image

    @save_image.setter
    def save_image(self, value):
        self._save_image = value

    def configinfo(self):
        print(">>>>>>>> CONFIG INFO")
        print("IMAGE")
        print("    SIZE_X: %d" % self._IMAGE_SIZE_X)
        print("    SIZE_X: %d" % self._IMAGE_SIZE_Y)
        print("    CHANNEL: %d" % self._IMAGE_CHANNEL)
        print("TRAINING")
        print("    LEARNING RATE: %.5f" % self._learning_rate)
        print("    BATCH NUMBER: %d" % self._batch_num)
        print("    EPOCH NUMBER: %d" % self._epoch_num)
        print("RESULT")
        print("    CLASS NUMBER: %d" % self._CLASS_NUM)
        print(">>>>>>>>>>>>>>>>>>>>>")
        print("\n")

    def __load_training_data(self, dir):
        filefilter = ('.bmp', '.png', '.jpg', '.tif')
        filename = []
        dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        for i in range(len(dirs)):
            for f in os.listdir(os.path.join(dir, dirs[i])):
                if f.lower().endswith(filefilter):
                    filename.append(f)

        img_num = len(filename)
        data = np.empty([
         img_num,
         self._IMAGE_SIZE_Y,
         self._IMAGE_SIZE_X,
         self._IMAGE_CHANNEL], dtype="float32")
        label = np.zeros([img_num, self._CLASS_NUM], dtype="uint8")
        mean = np.zeros([img_num, self._IMAGE_CHANNEL], dtype="float32")
        cnt_file = 0
        for i in range(len(dirs)):
            files = os.listdir(os.path.join(dir, dirs[i]))
            for j in range(len(files)):
                img_name = os.path.join(dir, dirs[i], files[j])
                if not img_name.lower().endswith(filefilter):
                    pass
                else:
                    img = Image.open(img_name)
                    width, height = img.size
                    if width != self._IMAGE_SIZE_X or height != self._IMAGE_SIZE_Y:
                        img = img.resize((self._IMAGE_SIZE_X, self._IMAGE_SIZE_Y), Image.BICUBIC)
                    if self._IMAGE_CHANNEL == 1:
                        img = img.convert("L")
                    tmp_data = np.asarray(img)
                    if self._IMAGE_CHANNEL == 1:
                        tmp_data = tmp_data.reshape([self._IMAGE_SIZE_Y,
                         self._IMAGE_SIZE_X,
                         1])
                tmp_data = tmp_data / 255.0
                tmp_mean = np.mean(tmp_data, axis=(0, 1))
                mean[cnt_file, :] = tmp_mean
                tmp_normalize = tmp_data - tmp_mean
                data[cnt_file, :, :, :] = tmp_normalize
                label[(cnt_file, i)] = 1
                cnt_file += 1

        return (
         data, label, mean, filename)

    def __load_prediction_data(self, dir):
        filefilter = ('.bmp', '.png', '.jpg', '.tif')
        filename = []
        for f in os.listdir(dir):
            if f.lower().endswith(filefilter):
                filename.append(f)

        img_num = len(filename)
        data = np.empty([
         img_num,
         self._IMAGE_SIZE_Y,
         self._IMAGE_SIZE_X,
         self._IMAGE_CHANNEL], dtype="float32")
        mean = np.zeros([img_num, self._IMAGE_CHANNEL], dtype="float32")
        for i in range(img_num):
            img = Image.open(os.path.join(dir, filename[i]))
            width, height = img.size
            if width != self._IMAGE_SIZE_X or height != self._IMAGE_SIZE_Y:
                img = img.resize((
                 self._IMAGE_SIZE_X, self._IMAGE_SIZE_Y), Image.BICUBIC)
            if self._IMAGE_CHANNEL == 1:
                img = img.convert("L")
            tmp_data = np.asarray(img)
            if self._IMAGE_CHANNEL == 1:
                tmp_data = tmp_data.reshape([self._IMAGE_SIZE_Y,
                 self._IMAGE_SIZE_X,
                 1])
            tmp_data = tmp_data / 255.0
            tmp_mean = np.mean(tmp_data, axis=(0, 1))
            mean[i, :] = tmp_mean
            tmp_normalize = tmp_data - tmp_mean
            data[i, :, :, :] = tmp_normalize

        return (
         data, mean, filename)

    def __write_classresult(self, filefullname, images, labels, names):
        resultfile = open(filefullname, "w")
        text = "name, result, label"
        for cnt_output in range(self._CLASS_NUM):
            text += ", class_%02d" % cnt_output

        resultfile.write(text + "\n")
        ys_class = np.argmax(labels, axis=1)
        for cnt_batch in range(0, len(images), self._batch_num):
            idx_st = cnt_batch
            if idx_st + self._batch_num > len(images):
                idx_ed = len(images)
            else:
                idx_ed = idx_st + self._batch_num
            y_pre = self._sess.run(self._softmax, feed_dict={(self._xs): (images[idx_st:idx_ed]),
             (self._ys): (labels[idx_st:idx_ed])})
            y_class = np.argmax(y_pre, axis=1)
            for cnt_file in range(idx_ed - idx_st):
                text = "%s, %d, %d" % (
                 names[cnt_file + idx_st],
                 y_class[cnt_file],
                 ys_class[cnt_file + idx_st])
                for cnt_output in range(self._CLASS_NUM):
                    text += ", %.5f" % y_pre[(cnt_file, cnt_output)]

                resultfile.write(text + "\n")

        resultfile.close()

    def __write_predictionresult(self, predict_path, images, names):
        filefullname = "%s/classResult_prediction.txt" % predict_path
        resultfile = open(filefullname, "w")
        text = "name, result"
        for cnt_output in range(self._CLASS_NUM):
            text += ", class_%02d" % cnt_output

        resultfile.write(text + "\n")
        if self._save_image:
            for cnt_output in range(self._CLASS_NUM):
                dir = "%s/class_%02d" % (predict_path, cnt_output)
                if not os.path.exists(dir):
                    os.mkdir(dir)

        for cnt_batch in range(0, len(images), self._batch_num):
            idx_st = cnt_batch
            if idx_st + self._batch_num > len(images):
                idx_ed = len(images)
            else:
                idx_ed = idx_st + self._batch_num
            y_pre = self._sess.run(self._softmax, feed_dict={(self._xs): (images[idx_st:idx_ed])})
            y_class = np.argmax(y_pre, axis=1)
            for cnt_file in range(idx_ed - idx_st):
                text = "%s, %d" % (
                 names[cnt_file + idx_st],
                 y_class[cnt_file])
                for cnt_output in range(self._CLASS_NUM):
                    text += ", %.5f" % y_pre[(cnt_file, cnt_output)]

                resultfile.write(text + "\n")
                if self._save_image:
                    src = "%s/%s" % (predict_path, names[cnt_file + idx_st])
                    dest = "%s/class_%02d/%s" % (
                     predict_path,
                     y_class[cnt_file],
                     names[cnt_file + idx_st])
                    copyfile(src, dest)

        resultfile.close()

    def __layer_convolution(self, name, input_layer, filter_x, filter_y, stride_x, stride_y, input_num, output_num):
        if self._show_detial:
            print("  Convolution")
            print("  Name: %s" % name)
            print("  Filter X: %d" % filter_x)
            print("  Filter Y: %d" % filter_y)
            print("  Stride X: %d" % stride_x)
            print("  Stride Y: %d" % stride_y)
            print("  Input Num: %d" % input_num)
            print("  Output Num: %d" % output_num)
            print("  Shape Input: %s" % input_layer.shape)
        with tf.name_scope(name):
            weight = tf.get_variable(name="%s_%s" % (name, "weight"), shape=[
             filter_x, filter_y, input_num, output_num], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name="%s_%s" % (name, "bias"), shape=[
             output_num], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input_layer, weight, strides=[
             1, stride_x, stride_y, 1], padding="SAME", name="%s_%s" % (name, "conv"))
            x = tf.nn.bias_add(x, bias, name="add")
            x = tf.nn.relu(x, name="relu")
            return x

    def __layer_inception(self, name, input_layer, input_num, output_num_1x1, output_num_3x3, output_num_5x5, output_num_pool):
        if self._show_detial:
            print("  Inception")
            print("  Name: %s" % name)
            print("  Input Num: %d" % input_num)
            print("  Output 1x1 Num: %d" % output_num_1x1)
            print("  Output 3x3 Num: %d" % output_num_3x3)
            print("  Output 5x5 Num: %d" % output_num_5x5)
            print("  Output pool Num: %d" % output_num_pool)
        with tf.name_scope(name):
            weight_1x1 = tf.get_variable(name="%s_%s" % (name, "weight_1x1"), shape=[
             1, 1, input_num, output_num_1x1], initializer=tf.contrib.layers.xavier_initializer())
            bias_1x1 = tf.get_variable(name="%s_%s" % (name, "bias_1x1"), shape=[
             output_num_1x1], initializer=tf.constant_initializer(0.0))
            x_1x1 = tf.nn.conv2d(input_layer, weight_1x1, strides=[
             1, 1, 1, 1], padding="SAME", name="%s_%s" % (name, "conv_1x1"))
            x_1x1 = tf.nn.bias_add(x_1x1, bias_1x1, name="%s_%s" % (name, "add_1x1"))
            x_1x1 = tf.nn.relu(x_1x1, name="%s_%s" % (name, "relu_1x1"))
            weight_reduce3x3 = tf.get_variable(name="%s_%s" % (name, "weight_reduce3x3"), shape=[
             1, 1, input_num, output_num_1x1], initializer=tf.contrib.layers.xavier_initializer())
            bias_reduce3x3 = tf.get_variable(name="%s_%s" % (name, "bias_reduce3x3"), shape=[
             output_num_1x1], initializer=tf.constant_initializer(0.0))
            x_3x3 = tf.nn.conv2d(input_layer, weight_reduce3x3, strides=[
             1, 1, 1, 1], padding="SAME", name="%s_%s" % (name, "conv_reduce3x3"))
            x_3x3 = tf.nn.bias_add(x_3x3, bias_reduce3x3, name="%s_%s" % (name, "add_reduce3x3"))
            x_3x3 = tf.nn.relu(x_3x3, name="%s_%s" % (name, "relu_reduce3x3"))
            weight_3x3 = tf.get_variable(name="%s_%s" % (name, "weight_3x3"), shape=[
             1, 1, output_num_1x1, output_num_3x3], initializer=tf.contrib.layers.xavier_initializer())
            bias_3x3 = tf.get_variable(name="%s_%s" % (name, "bias_3x3"), shape=[
             output_num_3x3], initializer=tf.constant_initializer(0.0))
            x_3x3 = tf.nn.conv2d(x_3x3, weight_3x3, strides=[
             1, 1, 1, 1], padding="SAME", name="%s_%s" % (name, "conv_3x3"))
            x_3x3 = tf.nn.bias_add(x_3x3, bias_3x3, name="%s_%s" % (name, "add_3x3"))
            x_3x3 = tf.nn.relu(x_3x3, name="%s_%s" % (name, "relu_3x3"))
            weight_reduce5x5 = tf.get_variable(name="%s_%s" % (name, "weight_reduce5x5"), shape=[
             1, 1, input_num, output_num_1x1], initializer=tf.contrib.layers.xavier_initializer())
            bias_reduce5x5 = tf.get_variable(name="%s_%s" % (name, "bias_reduce5x5"), shape=[
             output_num_1x1], initializer=tf.constant_initializer(0.0))
            x_5x5 = tf.nn.conv2d(input_layer, weight_reduce5x5, strides=[
             1, 1, 1, 1], padding="SAME", name="%s_%s" % (name, "conv_reduce5x5"))
            x_5x5 = tf.nn.bias_add(x_5x5, bias_reduce5x5, name="%s_%s" % (name, "add_reduce5x5"))
            x_5x5 = tf.nn.relu(x_5x5, name="%s_%s" % (name, "relu_reduce5x5"))
            weight_5x5 = tf.get_variable(name="%s_%s" % (name, "weight_5x5"), shape=[
             1, 1, output_num_1x1, output_num_5x5], initializer=tf.contrib.layers.xavier_initializer())
            bias_5x5 = tf.get_variable(name="%s_%s" % (name, "bias_5x5"), shape=[
             output_num_5x5], initializer=tf.constant_initializer(0.0))
            x_5x5 = tf.nn.conv2d(x_5x5, weight_5x5, strides=[
             1, 1, 1, 1], padding="SAME", name="%s_%s" % (name, "conv_5x5"))
            x_5x5 = tf.nn.bias_add(x_5x5, bias_5x5, name="%s_%s" % (name, "add_5x5"))
            x_5x5 = tf.nn.relu(x_5x5, name="%s_%s" % (name, "relu_5x5"))
            x_pool = tf.nn.max_pool(input_layer, ksize=[
             1, 3, 3, 1], strides=[
             1, 1, 1, 1], padding="SAME", name="%s_%s" % (name, "maxpool"))
            weight_pool = tf.get_variable(name="%s_%s" % (name, "weight_pool"), shape=[
             1, 1, input_num, output_num_pool], initializer=tf.contrib.layers.xavier_initializer())
            bias_pool = tf.get_variable(name="%s_%s" % (name, "bias_pool"), shape=[
             output_num_pool], initializer=tf.constant_initializer(0.0))
            x_pool = tf.nn.conv2d(x_pool, weight_pool, strides=[
             1, 1, 1, 1], padding="SAME", name="%s_%s" % (name, "conv_pool"))
            x_pool = tf.nn.bias_add(x_pool, bias_pool, name="%s_%s" % (name, "add_pool"))
            x_pool = tf.nn.relu(x_pool, name="%s_%s" % (name, "relu_pool"))
            x = tf.concat([x_1x1, x_3x3, x_5x5, x_pool], 3)
            return x

    def __layer_pooling(self, name, input_layer, method, filter_x, filter_y, stride_x, stride_y):
        if self._show_detial:
            print("  Pooling")
            print("  Name: %s" % name)
            print("  Method: %s" % method)
            print("  Filter X: %d" % filter_x)
            print("  Filter Y: %d" % filter_y)
            print("  Stride X: %d" % stride_x)
            print("  Stride Y: %d" % stride_y)
            print("  Shape Input: %s" % input_layer.shape)
        with tf.name_scope(name):
            if method == "Max":
                x = tf.nn.max_pool(input_layer, ksize=[
                 1, filter_x, filter_y, 1], strides=[
                 1, stride_x, stride_y, 1], padding="SAME", name="%s_%s" % (name, "maxpool"))
            return x

    def __layer_fullconnect(self, name, input_layer, is_reshape, input_num, output_num):
        if self._show_detial:
            print("  FullConnect")
            print("  Name: %s" % name)
            print("  Reshape: %s" % is_reshape)
            print("  Input Num: %d" % input_num)
            print("  Output Num: %d" % output_num)
            print("  Shape Input: %s" % input_layer.shape)
        if is_reshape == "True":
            x = tf.reshape(input_layer, [-1, input_num])
        else:
            x = input_layer
        with tf.name_scope(name):
            weight = tf.get_variable(name="%s_%s" % (name, "weight"), shape=[
             input_num, output_num], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name="%s_%s" % (name, "bias"), shape=[
             output_num], initializer=tf.constant_initializer(0.0))
            x = tf.matmul(x, weight, name="%s_%s" % (name, "mul"))
            x = tf.nn.bias_add(x, bias, name="%s_%s" % (name, "add"))
            return x

    def __layer_output(self, name, input_layer, input_num):
        if self._show_detial:
            print("  Output")
            print("  Name: %s" % name)
            print("  Input Num: %d" % input_num)
            print("  Shape Input: %s" % input_layer.shape)
        with tf.name_scope(name):
            weight = tf.get_variable(name="%s_%s" % (name, "weight"), shape=[
             input_num, self._CLASS_NUM], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name="%s_%s" % (name, "bias"), shape=[
             self._CLASS_NUM], initializer=tf.constant_initializer(0.0))
            x = tf.matmul(input_layer, weight, name="%s_%s" % (name, "mul"))
            x = tf.nn.bias_add(x, bias, name="%s_%s" % (name, "add"))
            return x

    def build_model(self, model_cmd):
        print(">>>>>>>> BUILD MODEL")
        if self._model_builded:
            print("Model Builded.\n")
            return
        print("Build Model.")
        x = self._xs
        input_num = self._IMAGE_CHANNEL
        feature_size_x = self._IMAGE_SIZE_X
        feature_size_y = self._IMAGE_SIZE_Y
        lines = model_cmd.splitlines()
        for cnt_line in range(len(lines)):
            if self._show_detial:
                print("%d > %s" % (cnt_line, lines[cnt_line]))
            tokens = lines[cnt_line].split(",")
            if tokens[0] == "Convolution":
                x = self._DeepLearning__layer_convolution(tokens[1], x, int(tokens[2]), int(tokens[3]), int(tokens[4]), int(tokens[5]), input_num, int(tokens[6]))
                input_num = int(tokens[6])
                feature_size_x = math.ceil(feature_size_x / float(tokens[4]))
                feature_size_y = math.ceil(feature_size_y / float(tokens[5]))
                if self._show_detial:
                    print("  feature size x: %d" % feature_size_x)
                    print("  feature size y: %d" % feature_size_y)
            elif tokens[0] == "Inception":
                output_1x1 = int(tokens[2])
                output_3x3 = int(tokens[3])
                output_5x5 = int(tokens[4])
                output_pool = int(tokens[5])
                x = self._DeepLearning__layer_inception(tokens[1], x, input_num, output_1x1, output_3x3, output_5x5, output_pool)
                input_num = output_1x1 + output_3x3 + output_5x5 + output_pool
            elif tokens[0] == "Pooling":
                x = self._DeepLearning__layer_pooling(tokens[1], x, tokens[2], int(tokens[3]), int(tokens[4]), int(tokens[5]), int(tokens[6]))
                feature_size_x = math.ceil(feature_size_x / float(tokens[5]))
                feature_size_y = math.ceil(feature_size_y / float(tokens[6]))
                if self._show_detial:
                    print("  feature size x: %d" % feature_size_x)
                    print("  feature size y: %d" % feature_size_y)
            elif tokens[0] == "FullConnect":
                if tokens[2] == "True":
                    input_num = input_num * feature_size_x * feature_size_y
                x = self._DeepLearning__layer_fullconnect(tokens[1], x, tokens[2], input_num, int(tokens[3]))
                input_num = int(tokens[3])
            elif tokens[0] == "Output":
                x = self._DeepLearning__layer_output(tokens[1], x, input_num)
                input_num = self._CLASS_NUM
            elif tokens[0] == "Residual":
                print("res")
            else:
                print("other")

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._ys, logits=x))
        train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(cross_entropy)
        prediction = tf.equal(tf.argmax(x, 1), tf.argmax(self._ys, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        softmax = tf.nn.softmax(x)
        init = tf.global_variables_initializer()
        self._train_step = train_step
        self._prediction = prediction
        self._accuracy = accuracy
        self._softmax = softmax
        self._sess.run(init)
        self._model_builded = True
        print(">>>>>>>> Build Finished\n")

    def load_ckpt(self, ckptfullname):
        print(">>>>>>>> LOAD SAVER")
        if not self._model_builded:
            print("Model not Builded.\n")
            return
        print("Open Saver.")
        saver = tf.train.Saver()
        saver.restore(self._sess, ckptfullname)
        self._ckpt_loaded = True
        print(">>>>>>>> LOADSAVER Finish\n")

    def train_model(self, train_data_path, test_data_path, output_path):
        print(">>>>>>>> TRAIN MODEL")
        if not self._model_builded:
            print("Model not Builded.\n")
            return
        print("Train Model.")
        if self._show_detial:
            print("Train Data: %s" % train_data_path)
            print("Test Data: %s" % test_data_path)
            print("Output: %s" % output_path)
        print("  Open Log.", end="")
        logfile = open("%s/TraingEpoch.log" % output_path, "w")
        print(" - Finished.")
        print("  Load Data", end="")
        train_image, train_labels, train_mean, train_name = self._DeepLearning__load_training_data(train_data_path)
        test_image, test_labels, test_mean, test_name = self._DeepLearning__load_training_data(test_data_path)
        print(" - Finished.")
        saver = tf.train.Saver()
        for cnt_epoch in range(self._epoch_num):
            shuffle_list = np.arange(len(train_image))
            random.shuffle(shuffle_list)
            tmp_trainvalue = 0
            for cnt_batch in range(0, len(train_image), self._batch_num):
                idx_st = cnt_batch
                if idx_st + self._batch_num > len(train_image):
                    idx_ed = len(train_image)
                else:
                    idx_ed = idx_st + self._batch_num
                self._sess.run(self._train_step, feed_dict={(self._xs): (train_image[shuffle_list[idx_st:idx_ed]]),
                 (self._ys): (train_labels[shuffle_list[idx_st:idx_ed]])})
                train_value = self._sess.run(self._accuracy, feed_dict={(self._xs): (train_image[shuffle_list[idx_st:idx_ed]]),
                 (self._ys): (train_labels[shuffle_list[idx_st:idx_ed]])})
                progressbar = idx_ed * 100 / len(train_image)
                text = "\rEpoch %5d - %03d%%" % (cnt_epoch, progressbar)
                print(text, end="")
                tmp_trainvalue += train_value * (idx_ed - idx_st)

            tmp_trainvalue /= len(train_image)
            tmp_testvalue = 0
            for cnt_batch in range(0, len(test_image), self._batch_num):
                idx_st = cnt_batch
                if idx_st + self._batch_num > len(test_image):
                    idx_ed = len(test_image)
                else:
                    idx_ed = idx_st + self._batch_num
                test_value = self._sess.run(self._accuracy, feed_dict={(self._xs): (test_image[idx_st:idx_ed]),
                 (self._ys): (test_labels[idx_st:idx_ed])})
                tmp_testvalue += test_value * (idx_ed - idx_st)

            tmp_testvalue /= len(test_image)
            text = "Epoch %5d - testValue = %.5f, trainValue = %.5f" % (
             cnt_epoch, tmp_testvalue, tmp_trainvalue)
            print("\r    " + text)
            logfile.write(text + "\n")
            if cnt_epoch == self._epoch_num - 1:
                ckptfullname = "%s/data_30712.ckpt" % output_path
                saver.save(self._sess, ckptfullname, write_meta_graph=False)
            elif cnt_epoch % 100 == 99:
                ckptfullname = "%s/data_%05d.ckpt" % (output_path, cnt_epoch)
                saver.save(self._sess, ckptfullname, write_meta_graph=False)

        print("  Output classResult.", end="")
        filefullname = "%s/classResult_train.txt" % output_path
        self._DeepLearning__write_classresult(filefullname, train_image, train_labels, train_name)
        filefullname = "%s/classResult_test.txt" % output_path
        self._DeepLearning__write_classresult(filefullname, test_image, test_labels, test_name)
        print(" - Finished.")
        print("  Close.", end="")
        logfile.close()
        print(" - Finished.")
        print(">>>>>>>> Train Finish\n")

    def predict_model(self, predict_data_path):
        print(">>>>>>>> PREDICT MODEL")
        if not self._model_builded:
            print("Model not Builded.\n")
            return
        if not self._ckpt_loaded:
            print("SAVER not Opened.\n")
            return
        print("Predict Model.")
        if self._show_detial:
            print("Predict Data: %s" % predict_data_path)
        print("  Load Data", end="")
        predict_image, predict_mean, predict_name = self._DeepLearning__load_prediction_data(predict_data_path)
        print(" - Finished.")
        self._DeepLearning__write_predictionresult(predict_data_path, predict_image, predict_name)
        print(">>>>>>>> Prediction Finish\n")

    def free_model(self):
        print(">>>>>>>> FREE MODEL")
        self._model_builded = False
        self._ckpt_loaded = False
        tf.Session(tf.reset_default_graph())
        print(">>>>>>>> Free Finish\n")