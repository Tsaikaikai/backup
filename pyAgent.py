# uncompyle6 version 3.9.2
# Python bytecode version base 3.5 (3350)
# Decompiled from: Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)]
# Embedded file name: E:\Project\Proj-17-03\x64\Release\pyAgent.py
# Compiled at: 2017-11-20 14:48:10
# Size of source mod 2**32: 3976 bytes
import os
from deeplearning import DeepLearning
from txtWriter import TextCrypto

class TrainAndPredict(object):

    def __init__(self):
        self._textcrypto = TextCrypto()

    def train(self, output_path, traindata_path, testdata_path):
        config_path = os.path.join(output_path, "data01.auo")
        config = self._textcrypto.read_file(config_path)
        print(config)
        model_path = os.path.join(output_path, "data02.auo")
        model = self._textcrypto.read_file(model_path)
        imagewidth = 100
        imageheight = 100
        imagechannel = 3
        outputclass = 2
        learningrate = 1e-05
        learningbatch = 300
        learningepoch = 300
        lines = config.splitlines()
        for cnt_line in range(len(lines)):
            tokens = lines[cnt_line].split(":", 1)
            if tokens[0] == "ImageWidth":
                imagewidth = int(tokens[1])
            elif tokens[0] == "ImageHeight":
                imageheight = int(tokens[1])
            elif tokens[0] == "ImageChannel":
                if tokens[1] == "Gray":
                    imagechannel = 1
            elif tokens[0] == "OutputClass":
                outputclass = int(tokens[1])
            elif tokens[0] == "LearningRate":
                learningrate = float(tokens[1])
            elif tokens[0] == "LearningBatch":
                learningbatch = int(tokens[1])
            elif tokens[0] == "LearningEpoch":
                learningepoch = int(tokens[1])

        obj_deeplearning = DeepLearning(imagewidth, imageheight, imagechannel, outputclass)
        obj_deeplearning.rate = learningrate
        obj_deeplearning.batch = learningbatch
        obj_deeplearning.epoch = learningepoch
        obj_deeplearning.configinfo()
        obj_deeplearning.build_model(model)
        obj_deeplearning.train_model(traindata_path, testdata_path, output_path)

    def predict(self, output_path, predictdata_path, batch_num=0):
        config_path = os.path.join(output_path, "data01.auo")
        config = self._textcrypto.read_file(config_path)
        model_path = os.path.join(output_path, "data02.auo")
        model = self._textcrypto.read_file(model_path)
        imagewidth = 100
        imageheight = 100
        imagechannel = 3
        outputclass = 2
        learningrate = 1e-05
        learningbatch = 300
        learningepoch = 300
        lines = config.splitlines()
        for cnt_line in range(len(lines)):
            tokens = lines[cnt_line].split(":", 1)
            if tokens[0] == "ImageWidth":
                imagewidth = int(tokens[1])
            elif tokens[0] == "ImageHeight":
                imageheight = int(tokens[1])
            elif tokens[0] == "ImageChannel":
                if tokens[1] == "Gray":
                    imagechannel = 1
            elif tokens[0] == "OutputClass":
                outputclass = int(tokens[1])
            elif tokens[0] == "LearningRate":
                learningrate = float(tokens[1])
            elif tokens[0] == "LearningBatch":
                learningbatch = int(tokens[1])
            elif tokens[0] == "LearningEpoch":
                learningepoch = int(tokens[1])

        ckptfullname = "%s/data_30712.ckpt" % output_path
        obj_deeplearning = DeepLearning(imagewidth, imageheight, imagechannel, outputclass)
        obj_deeplearning.rate = learningrate
        obj_deeplearning.epoch = learningepoch
        if batch_num > 0:
            obj_deeplearning.batch = batch_num
        else:
            obj_deeplearning.batch = learningbatch
        obj_deeplearning.configinfo()
        obj_deeplearning.build_model(model)
        obj_deeplearning.load_ckpt(ckptfullname)
        obj_deeplearning.predict_model(predictdata_path)
