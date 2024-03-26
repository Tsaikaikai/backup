        private void MatchNotch2HObject(out HObject HO_img_raw, out HObject HO_img, out int notch_idx, HObject[] hobj_list, HTuple modelId, int total_w, int total_h, String cam_name)
        {
            //merge原圖
            HObject concatImage = null;
            HOperatorSet.GenEmptyObj(out concatImage);
            foreach (HObject hobj_temp in hobj_list)
                HOperatorSet.ConcatObj(concatImage, hobj_temp, out concatImage);
            HObject concatImage_raw;
            HOperatorSet.TileImages(concatImage, out concatImage_raw, 1, "vertical");

            //輸出拼接原圖(Notch在原始位置)
            HObject CropImage_raw;
            HOperatorSet.GetImageSize(concatImage_raw, out HTuple w, out HTuple h);
            HOperatorSet.CropPart(concatImage_raw, out CropImage_raw, 0, 0, total_w, (int)h - total_h);
            
            HO_img_raw = CropImage_raw;
            HO_img = CropImage_raw;
            notch_idx = 0;

            try
            {
                concatImage = null;
                HOperatorSet.GenEmptyObj(out concatImage);
                for (int j = 0; j < 2; j++)
                {
                    HOperatorSet.ConcatObj(concatImage, CropImage_raw, out concatImage);
                }

                //拼接兩次原圖
                HObject concatImage_all;
                HOperatorSet.TileImages(concatImage, out concatImage_all, 1, "vertical");

                // 讀取模型文件找最靠近中間的notch
                HTuple cent_y, score;
                HalconFunction.FindCenterNotch(concatImage_all, modelId, 0.6, out cent_y, out score);

                int mid_y = Convert.ToInt32((double)cent_y);

                Console.WriteLine("notch_idx : " + mid_y.ToString());

                notch_idx = mid_y;
                int start = mid_y - total_h / 2;


                if (start < 0)
                {
                    start = 0;
                }

                int x1 = 0;
                int y1 = start;
                int x2 = total_w;
                int y2 = start + total_h;

                // 裁剪notch置中影像區域
                HObject reducedImage, result;
                HOperatorSet.GenRectangle1(out HObject retangle, y1, x1, y2, x2);
                HOperatorSet.ReduceDomain(concatImage_all, retangle, out reducedImage);
                HOperatorSet.CropDomain(reducedImage, out result);
                HO_img = result;
                
            }
            catch(Exception ee)
            {
                MessageBox.Show(ee.Message);  //20230601
            }
        }