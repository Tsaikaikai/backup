import cv2
import numpy as np
import os
import json
import math
import shutil
import sys

class Aging_detect():
    def __init__(self, image_path, config_path):
        
        self.image_path = image_path
        self.config_path = config_path
        self.output_path = path + "/result"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        self.kernel = np.ones((7, 7), np.uint8)
        
        with open(config_path) as f:
            self.data = json.load(f)
            
        self.obj_count = len(self.data["obj_list"])
        
        # find all img file
        self.file_list = [path + "\\" + i for i in os.listdir(path) if '.jpeg' in i]
        self.imgsize = cv2.imread(self.file_list[0]).shape
        #偵測產品輪廓用
        self.img_obj = np.zeros(self.imgsize[:-1], np.int64)

        #有亮沒亮用
        self.black_img = 0
        self.detect_type = self.data['detect_list']
        self.img_type_list = [np.zeros(self.imgsize[:-1], np.int64) for i in range(len(self.detect_type))]
        
        self.roi_img = np.zeros(self.imgsize[:-1], np.uint8)
        self.roi_img = cv2.fillPoly(self.roi_img, [np.array(self.data['detect_roi'])], 255)
        
        self.list_obj_img = []
        # 時序判定用
        self.keep_cnt = 2
        self.img_judge_list = []
        self.pattern_result = {}
        for tp in self.data["detect_list"]:
            self.img_judge_list.append({
                "ctn": 0,  #計算目前第幾張
                "img_seq": []  #連續N幀的影像，避免亮一半被抓到
            })
            self.pattern_result[tp["type_name"]] = []
            
        self.judge_detail = self.data["obj_list"]
    
    def Find_obj_area(self):
        list_gray = []
        obj_gray_spec = self.data['obj_gray_spec']

        for img_path in self.file_list:

            img = cv2.imread(img_path)

            # to gray 抓需要辨識的frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #gray_blur = cv2.blur(gray, (7, 7), 2)
            gray_blur = np.array(np.power(gray, 0.7),dtype=np.uint8)

            #gray_blur = cv2.equalizeHist(gray_blur)
            gray_blur = cv2.bitwise_and(gray_blur, gray_blur, mask = self.roi_img)

            #取得當下ROI區域的灰階平均值
            gray_value = gray_blur[self.roi_img == 255].mean()
            if (gray_value < obj_gray_spec[0] or gray_value > obj_gray_spec[1]):
                continue
            list_gray.append(gray_value)

            #切割當下物件區域
            ret2, th_otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                          
            th_otsu = cv2.erode(th_otsu, self.kernel, iterations=3)
            th_otsu = cv2.dilate(th_otsu, self.kernel, iterations=3)
            
            self.img_obj[th_otsu != 0] += 1
        
        # 找所有物件的候選區域
        obj_occur_ratio = self.data['obj_occur_ratio']
        valid_count = len(list_gray)
        obj_detect_th = valid_count * obj_occur_ratio
        img_sheet = np.zeros(self.imgsize[:-1], np.uint8)
        img_sheet[self.img_obj > obj_detect_th] = 255
        self.img_sheet = img_sheet
        
    
    def Check_obj_info(self):
        # 分析每一個obj info      
        for obj_config in self.data["obj_list"]:

            obj_no = obj_config["obj_no"]
            area_spec = obj_config['area_spec']
            width_spec = obj_config['width_spec']
            height_spec = obj_config['height_spec']

            # ROI框中心點
            cent_x,cent_y = np.array(obj_config["roi_pox"]).mean(axis=0)

            roi_temp = np.zeros(self.imgsize[:-1], np.uint8)
            roi_temp = cv2.fillPoly(roi_temp, [np.array(obj_config["roi_pox"])], 255)
            img_obj = cv2.bitwise_and(self.img_sheet, self.img_sheet, mask = roi_temp)
            contours, hierarchy = cv2.findContours(img_obj, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            min_d = -1
            find_c = None
            find_c_area = 0

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = cv2.contourArea(c)

                #符合obj spec 要找到離中心最近的
                if (area >= area_spec[0]) and (
                    area <= area_spec[1]) and (
                    w >= width_spec[0]) and (
                    w <= width_spec[1]) and (
                    h >= height_spec[0]) and (
                    h <= height_spec[1]):


                    M = cv2.moments(c)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        d = math.sqrt((cx - cent_x)**2 + (cy - cent_y)**2)
                        if ( min_d == -1 ) or ( d < min_d ):
                            min_d = d
                            find_c = c
                            find_c_area = area

            #只留找到最近的Contour obj
            img_findobj = np.zeros(self.imgsize[:-1], np.uint8)

            if (min_d!=-1):
                img_findobj = cv2.drawContours(img_findobj, [find_c], -1, 255, thickness=cv2.FILLED)
                img_findobj = cv2.bitwise_and(img_findobj, img_findobj, mask = roi_temp)

            self.list_obj_img.append({"obj_img":img_findobj,"area":find_c_area})
            
            
    def Check_HSV_Info(self):
        obj_gray_spec = self.data['obj_gray_spec']
        for n, img_path in enumerate(self.file_list):

            img = cv2.imread(img_path)

            # to gray 抓需要辨識的frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.blur(gray, (7, 7), 2)
            gray_blur = cv2.bitwise_and(gray_blur, gray_blur, mask = self.roi_img)
            #取得當下ROI區域的灰階平均值
            gray_value = gray_blur[self.roi_img == 255].mean()

            bypass = False

            #灰階不符的卡掉
            if (gray_value < obj_gray_spec[0] or gray_value > obj_gray_spec[1]):
                self.black_img = self.black_img + 1
                bypass = True
            
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            is_rainbow = self.check_rainbow(img_hsv)
            if is_rainbow :
                #print(img_path + "rainbow bypass")
                bypass = True

            for i, config_detect in enumerate(self.detect_type):

                detect_result = []
                hsv_lower = np.array(config_detect['HSV_LOWER'])
                hsv_upper = np.array(config_detect['HSV_UPPER'])

                #紅色 Hue 會跨255 所以分段處理
                if (hsv_upper[0] <= 255):
                    detect_mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)

                else:

                    LowerH1 = hsv_lower[0]
                    UpperH1 = 255
                    LowerH2 = 0
                    UpperH2 = hsv_upper[0] - 255

                    lower1 = np.array([LowerH1, hsv_lower[1], hsv_lower[2]])
                    upper1 = np.array([UpperH1, hsv_upper[1], hsv_upper[2]])
                    lower2 = np.array([LowerH2, hsv_lower[1], hsv_lower[2]])
                    upper2 = np.array([UpperH2, hsv_upper[1], hsv_upper[2]])

                    mask1 = cv2.inRange(img_hsv, lower1, upper1)
                    mask2 = cv2.inRange(img_hsv, lower2, upper2)
                    detect_mask = cv2.bitwise_or(mask1, mask2)

                self.img_judge_list[i]["ctn"] += 1
                self.img_judge_list[i]["img_seq"].append(detect_mask)

                #超過keep_cnt才開始辨識
                if self.img_judge_list[i]["ctn"] >= self.keep_cnt:

                    img_judge = np.zeros(self.imgsize[:-1], np.uint8)
                    
                    #連續影像取OR
                    for img_temp in self.img_judge_list[i]["img_seq"]:
                        img_judge = cv2.bitwise_or(img_judge, img_temp)

                    self.img_judge_list[i]["img_seq"].pop(0)

                    for obj_img in self.list_obj_img:
                        img_judge_and = cv2.bitwise_and(img_judge, img_judge, mask = obj_img["obj_img"])

                        if np.count_nonzero(img_judge_and) > obj_img["area"] * 0.5:
                            detect_result.append(config_detect["type_name"])
                        else:
                            detect_result.append("NG")
                    
                    if bypass:
                        detect_result = ["Null"] * len(self.list_obj_img)
                else:
                    detect_result = ["Null"] * len(self.list_obj_img)

                self.pattern_result[config_detect["type_name"]].append(detect_result)

    def Check_judge_Info(self):
        judge_seq = [[] for i in range(self.obj_count)]
        ng_type = [self.detect_type[i]['type_name'] for i in range(len(self.detect_type)) if self.detect_type[i]['Judge'] == "NG"]
        ok_type = [self.detect_type[i]['type_name'] for i in range(len(self.detect_type)) if self.detect_type[i]['Judge'] == "OK"]
        
        # create judge_detail 裝每個obj判斷的資訊
        self.ng_desc = ["Not_Found", "Sequence_Error"] + ["NG_" + i for i in ng_type]
        
        for n in range(len(self.file_list)):
            for obj_i in range(self.obj_count):

                #每個pattern裡第obj_i個取出
                temp = [self.pattern_result[pattern][n][obj_i] for key_i, pattern in enumerate(self.pattern_result.keys())]

                add = False
                tp = "Null"
                
                for t in temp:
                    if t in ng_type:
                        tp = t
                        break

                    if t!= "Null" and t!= "NG":
                        if add:
                            tp = "Trans"
                        else:
                            tp = t
                            add = True
                    elif t == "NG" and not add:
                        tp = t

                judge_seq[obj_i].append(tp)
                
        for i in range(self.obj_count):
            self.judge_detail[i]["judge"] = ""
            temp = {}
            for t in self.ng_desc:
                temp[t] = {"cnt":0,"detail":""}    
            self.judge_detail[i]["judge_detail"] = temp
        
    
        for obj_no, judge_list in enumerate(judge_seq):

            start_idx = -1
            for img_no, judge in enumerate(judge_list):

                tp = ""
                
                if judge in ["Null","Trans"]:
                    continue

                elif judge in ["NG"]:
                    tp = "Not_Found"
                    start_idx = -1  #NG之後重新判斷sequence

                elif judge in ng_type:
                    tp = "NG_" + judge
                else:
                    now_idx = ok_type.index(judge)
                    if start_idx == -1:
                        start_idx = now_idx
                    elif now_idx == (start_idx+1) % len(ok_type):
                        start_idx = (start_idx+1) % len(ok_type)
                    elif now_idx != start_idx:
                        tp = "Sequence_Error" 
                        
                if tp != "" :
                    if(self.judge_detail[obj_no]["judge_detail"][tp]["detail"]==""):
                        self.judge_detail[obj_no]["judge_detail"][tp]["detail"] += str(img_no)
                    else:
                        self.judge_detail[obj_no]["judge_detail"][tp]["detail"] += "," + str(img_no)
                    self.judge_detail[obj_no]["judge_detail"][tp]["cnt"] += 1

    def Result_Output(self):
        img_draw_all = np.zeros(self.imgsize, np.uint8)

        for i in range(len(self.judge_detail)):
            result = "OK"
            
            # 超過95%沒有pattern亮起，代表從頭到尾都沒亮過->可能沒放，可能放壞的
            # 寫no pattern 佛哥比較好認，不一定要寫No pattern
            
            if self.judge_detail[i]["judge_detail"]["Not_Found"]["cnt"] > (len(self.file_list) - self.black_img)*0.95:
                result = "No Pattern" 

            elif self.black_img == len(self.file_list):
                result = "All Black" 
        
            else:
                #檢查Judge Detail->cnt>0 即 NG
                for k, key in enumerate(self.judge_detail[i]["judge_detail"].keys()):
                    if (self.judge_detail[i]["judge_detail"][key]["cnt"] > 0):
                        result = "NG"

            self.judge_detail[i]["judge"] = result
            
            if result != "OK":
                color = (0, 0, 255) #red
            else:
                color = (0, 255, 0) #green

            img_draw_all = cv2.polylines(img_draw_all, pts=[np.array(self.judge_detail[i]['roi_pox'], np.int32)], isClosed=True, color=color, thickness=3)
            img_draw_all[self.list_obj_img[i]["obj_img"] > 0] = color  

            img_draw = np.zeros(self.imgsize, np.uint8)
            img_draw[self.list_obj_img[i]["obj_img"] > 0] = (255,255,255)
            img_draw = cv2.polylines(img_draw, pts=[np.array(self.judge_detail[i]['roi_pox'], np.int32)], isClosed=True, color=(255,0,0), thickness=3)
            cv2.imwrite(self.output_path + "/" + self.judge_detail[i]['obj_no'] + ".jpg", img_draw)
            
        cv2.imwrite(self.output_path + "/result.jpg", img_draw_all)

        self.data_transfer(self.judge_detail)

        with open(self.output_path + '/result.json', 'w') as fp:
            json.dump(self.judge_detail, fp, indent=4, separators=(',', ': '))

    def Save_NG_Images(self):

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
    
        for obj in self.judge_detail:
            (x, y, w, h) = cv2.boundingRect(np.array(obj["roi_pox"]))

            for pattern in obj['judge_detail']:

                for img_num in pattern['detail'].split(','):

                    "NG filename"
                    if img_num == "": continue
                    if len(str(img_num)) <= 3 : 
                        for i in range(3 - len(str(img_num))):
                            img_num = "0" + str(img_num)

                    src_path = os.path.join(self.image_path ,"IMG_" + str(img_num) + ".jpeg")
                    src_img = cv2.imread(src_path)
                    crop_img = src_img[y:y+h, x:x+w]
                    #save format -> obj_2_NG_G_001.jpeg
                    output_filename = "obj_" + str(obj['obj_no']) + "_"  + str(pattern["type"]) + "_" + str(img_num) + ".jpeg"
                    dst_path = os.path.join(self.output_path , output_filename)

                    if os.path.isfile(src_path):
                        cv2.imwrite(dst_path, crop_img)

    def data_transfer(self, input_data):
        for roi in input_data:
            judge_detail_dic = []
            for pattern in roi["judge_detail"]:
                pattern_dic = {}
                pattern_dic["type"] = pattern
                pattern_dic["cnt"] = roi["judge_detail"][pattern]["cnt"]
                pattern_dic["detail"] = roi["judge_detail"][pattern]["detail"]
                judge_detail_dic.append(pattern_dic)

            roi.pop("judge_detail")
            roi["judge_detail"] = judge_detail_dic


    def check_rainbow(self, img_hsv):

        R1_hsv_lower = np.array([0,50,50])
        R1_hsv_upper = np.array([5,255,255])
        R2_hsv_lower = np.array([140,50,50])
        R2_hsv_upper = np.array([179,255,255])

        G_hsv_lower = np.array([50,50,50])
        G_hsv_upper = np.array([85,255,255])

        B_hsv_lower = np.array([100,50,50])
        B_hsv_upper = np.array([140,255,255])

        mask_red1 = cv2.inRange(img_hsv, R1_hsv_lower, R1_hsv_upper)
        mask_red2 = cv2.inRange(img_hsv, R2_hsv_lower, R2_hsv_upper)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_green = cv2.inRange(img_hsv, G_hsv_lower, G_hsv_upper)
        mask_blue = cv2.inRange(img_hsv, B_hsv_lower, B_hsv_upper)

        # 計算紅色區域的平均x座標
        red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_x_coordinates = []
        for contour in red_contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if w > 80 or h > 80 :
                red_x_coordinates.append(x)
        if len(red_x_coordinates) > 0:
            red_x_avg = sum(red_x_coordinates) / len(red_x_coordinates)
        else:
            return False

        # 計算綠色區域的平均x座標
        green_contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_x_coordinates = []
        for contour in green_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 80 or h > 80 :
                green_x_coordinates.append(x)
        if len(green_x_coordinates) > 0:
            green_x_avg = sum(green_x_coordinates) / len(green_x_coordinates)
        else:
            return False

        # 計算藍色區域的平均x座標
        blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_x_coordinates = []
        for contour in blue_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 80 or h > 80 :
                blue_x_coordinates.append(x)
        if len(blue_x_coordinates) > 0:
            blue_x_avg = sum(blue_x_coordinates) / len(blue_x_coordinates)
        else:
            return False

        if (green_x_avg < red_x_avg) and (red_x_avg < blue_x_avg) and (blue_x_avg-green_x_avg) > 100:
            is_rainbow = True
        else:
            is_rainbow = False

        return is_rainbow

    def run(self):
        self.Find_obj_area() #找物件輪廓後選區
        self.Check_obj_info() #確定物件區域
        self.Check_HSV_Info() #找pattern
        self.Check_judge_Info() #合併pattern Judge資訊
        self.Result_Output() #繪圖輸出
        self.Save_NG_Images() #根據結果儲存NG圖

'''
if __name__ == '__main__':
    
    path = sys.argv[1]
    config_path = sys.argv[2]
    obj = Aging_detect(path, config_path)
    obj.run()
    print("Done!")

'''

if __name__ == '__main__':

    path = "dataset/123456_ABCDEFG/NG5_test"
    config_path = 'config/test.json'
    obj = Aging_detect(path, config_path)
    obj.run()
    print("Done!")

