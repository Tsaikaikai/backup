        private List<int> CatchAreaScanIndex(int linescanResolution, int notchLocation, int areaTotalNum, int catchNum, double notchShiftDegreeAngle)
        {
            if (catchNum > areaTotalNum)
            {
                new Exception("Catching number more than number of area scan images.");
            }
            int lineGap = linescanResolution / catchNum;
            int areaGap = linescanResolution / areaTotalNum;
            int shiftLine = Convert.ToInt32(linescanResolution * (notchShiftDegreeAngle / 360));
            int index = Convert.ToInt32(notchLocation + shiftLine);

            List<int> imgNumber = new List<int>();
            while (imgNumber.Count() < catchNum)
            {
                int captureNum = Convert.ToInt32(index / areaGap);

                if(Math.Abs((captureNum + 1) * areaGap - index) < Math.Abs((captureNum) * areaGap - index))
                {
                    captureNum++;
                }
                imgNumber.Add(captureNum);
                index = index + lineGap;
                if (index > linescanResolution)
                {
                    index = index - linescanResolution;
                }
            }
            return imgNumber;
        }
