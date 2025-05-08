//---------------------------------------------------------------------------

#include <vcl.h>
#include <fstream>
#pragma hdrstop

#include "SSO.h"
#include "math.h"
#include "TTLAStorage.h"
#include "TTLATifIO.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)
#pragma resource "*.dfm"
TForm1 *Form1;
//---------------------------------------------------------------------------
__fastcall TForm1::TForm1(TComponent* Owner)
        : TForm(Owner)
{
        DisplayGamma=1.0;
        MaskParameters[0]=2;
        MaskParameters[1]=2;
        MaskScale=0.3;
        MaskWeight=0;
        ScanBeta=5 ;     //Pooling exponent to obtain Total JND
        pi=3.1415926;

        TabSheet1->Show();
}
//---------------------------------------------------------------------------
void __fastcall TForm1::Btn_SSOClick(TObject *Sender)
{
    //Create csf filter
    CSFFilter();

    //Create mask filter
    MFilter();

    //create aperture
    Aperture();

    //Create border image
    SSOBorder();

    //Compute difference images
    Subtract2Img();      //ok

    //Convert images to luminance
    Convert2Lumin();     //ok

    //Create local luminance images or scalar mean luminance
    TempImg();           //ok

    //Create contrast images
    CCI();               //ok

    //Attenuate contrast images with border image
    Attenuate();         //ok

    //Filter contrast images with csf
    FourierCSF();        //ok

    //Use aperture to compute jnd images
    //Compute max of each jnd image
    ComputeJnd();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::ComputeJnd()
{
    //use aperture to compute jnd image
    //compute max of each jnd imag
    std::vector<float> temp;
    float mjnd=0.0, finshjnd, hjndtemp, hjndsum=0.0, sjnd, totjnd;

    Beta=StrToFloat(Edit_Beta->Text);
    // ConfinedConvolve[ fovapsmall, Abs[#]^beta]
    for(int i=0; i<actH; i++){
        for(int j=0; j<actW; j++)
        {
            //abs
            if(fcon_test[i*actW+j]<0.0)
                temp.push_back(pow((-fcon_test[i*actW+j]), Beta));
            else
                temp.push_back(pow(fcon_test[i*actW+j], Beta));
        }
    }

    jnd=ConfinedConvolve(fovapsmall, temp, actW, actH);
    //chop and find max.
    for(int i=0; i<actH; i++){
        for(int j=0; j<actW; j++)
        {
            finshjnd=jnd[i*actW+j];
            finshjnd=pow((finshjnd*dpp),(1/Beta));

            if((finshjnd<0.0000000001)&&(finshjnd>-0.0000000001))
                jnd[i*actW+j]=0.0;
            else
                jnd[i*actW+j]=finshjnd;

            if(finshjnd>mjnd)  //find the max jnd
                mjnd=finshjnd;

            //hjnd
            hjndtemp=temp[i*actW+j]*dpp;
            hjndsum+=hjndtemp;
        }
    }

    sjnd=pow(hjndsum, (1/Beta));    //Single jnd

    //Minkowski
    totjnd=Minkowski(jnd, 4);       //Total jnd

    JNDImgC->Height=400*((float)actH/(float)actW);
    ShowColorJND(jnd, actW, actH);
    maxJND->Caption=FloatToStrF(mjnd, ffNumber, 7, 3);
    tolJND->Caption=FloatToStrF(totjnd, ffNumber, 7, 3);
    sinJND->Caption=FloatToStrF(sjnd, ffNumber, 7, 3);
    Shape9->Brush->Color=clLime;
    TabSheet5->Show();
    SaveJNDBtn->Enabled=true;       //enable the button of save JND map
}
//---------------------------------------------------------------------------

float __fastcall TForm1::Minkowski(std::vector<float> list, int beta)
{
    //MinkowskiRealC=Compile[{{array,_Real,1},beta},(Plus @@ ((Abs[#]^beta)& /@  array))^(1/beta)];
    float minkow, total=0.0;
    int count;

    count=actH*actW;
    for(int i=0; i<count; i++){
        total+=(list[i]*list[i]*list[i]*list[i]);
    }
    minkow=pow(total, ((float)1/beta));

    return minkow;
}
//----------------------------------------------------------------------------

void __fastcall TForm1::Subtract2Img()
{
    //Compute difference images
    int count=actH*actW;
    diff.clear();   //samper
    for(int i=0; i<count; i++)
        diff.push_back(testImg[i]-referImg[i]);

    Shape8->Brush->Color=clLime;
    Application->ProcessMessages();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::FourierCSF()
{
    //Filter contrast image with CSF
    float temp;
    int size;
    //FFT
    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * actH * actW);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * actH * actW);
    size=actH*actW;
    //for test image============================
    //assign value to test fftw_complex *in
    for(int i=0; i<size; i++){
        in[i][0] = Att_test[i];
        in[i][1] = 0.0;
    }

    p = fftw_plan_dft_2d(actH, actW, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    //CSF*GRe
    for(int i=0; i<size; i++){
        in[i][0]=out[i][0]*csf[i];
        in[i][1]=out[i][1]*csf[i];
    }

    p = fftw_plan_dft_2d(actH, actW, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fcon_test.clear();  //samper
    //real part chop and times to contrast
    for(int i=0; i<size; i++){
         temp=out[i][0]/size;
         if((temp<0.0000000001)&&(temp>-0.0000000001))
             temp=0.0;
         fcon_test.push_back(temp);
    }

    //show image
    FFT_TestImg->Height=150*((float)actH/(float)actW);    //assign the height value to display
    //release memory
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    ShowFilterImg(fcon_test, actW, actH, 6);
    Shape7->Brush->Color=clLime;
    Application->ProcessMessages();
}
//----------------------------------------------------------------------------

void __fastcall TForm1::Attenuate()
{
    //attenuate contrast images with border image
    Att_test.clear();       //samper
    for(int i=0; i<actH; i++)
        for(int j=0; j<actW; j++)
            Att_test.push_back(border[i*actW+j]*ContrastTestImg[i*actW+j]);

    //show image
    Bod_TestImg->Height=150*((float)actH/(float)actW);    //assign the height value to display

    ShowFilterImg(Att_test, actW, actH, 5);
    Shape6->Brush->Color=clLime;
    Application->ProcessMessages();
}
//----------------------------------------------------------------------------

void __fastcall TForm1::Convert2Lumin()
{
    //convert image to luminance,
    //the value between 0~100
    GammaParameters=StrToInt(Edit_GammaPara->Text);
    Lum_ref =ToLuminance(referImg, GammaParameters);        //for mean luminance
    Lum_test=ToLuminance(diff, GammaParameters);

    Lum_TestImg->Height=150*((float)actH/(float)actW);    //assign the height value to display

    //ShowFilterImg(Lum_test, actW, actH, 2);
    ShowFilterImg(diff, actW, actH, 2);
}
//---------------------------------------------------------------------------

int __fastcall TForm1::intRound(float temp)
{
    int rTemp;
    float temp1;

    temp1=temp-(int)temp;
    rTemp=(temp1>0.5)?((int)temp+1):(int)temp;

    return rTemp;
}

//---------------------------------------------------------------------------
void __fastcall TForm1::MFilter()
{
    int mpixels, temp;
    float mdegrees;
    MaskFilter.clear();     //samper
    if(MaskWeight > 0 )
    {
        //Round
        temp=intRound(MaskScale/dpp);

        mpixels = 2*temp;
        mdegrees = mpixels*dpp;

        GaussianAperture(mpixels, mdegrees, MaskScale, true);
        for(int i=0; i<mpixels; i++)
            for(int j=0; j<mpixels; j++)
                MaskFilter.push_back(MaskWeight*GATable[i*mpixels+j]);
    }

}
//---------------------------------------------------------------

void __fastcall TForm1::GaussianAperture(int pixels, float degrees, float scale, bool normalize)
{
    float GATemp, value, tempi, tempj;
    float total=0.0, expvalue;;

    GATable.clear();
    value= (degrees/pixels)/scale;
    for(int i=0; i<pixels; i++) {
        for(int j=0; j<pixels; j++)
        {
            tempi=i-(pixels/2);
            tempj=j-(pixels/2);
            //-Pi(Plus @@ (({y, x} degrees/pixels/scale)^2 ))
            GATemp=exp((-3.1415926)*((tempi*tempi+tempj*tempj)*(value*value)));
            GATable.push_back(GATemp);

            //Chop
            if(GATable[i*pixels+j]<0.0000000001)
                GATable[i*pixels+j]=0.0;
            
            total+=GATable[i*pixels+j];
        }
    }

    if(normalize)
    {
        for(int i=0; i<pixels; i++)
            for(int j=0; j<pixels; j++)
                GATable[i*pixels+j]/=total;
    }
}
//-----------------------------------------------------------------

void __fastcall TForm1::Aperture()
{
    /*newscale = fovscale/Sqrt[beta];
    appix = 2Round[newscale/dpp];
    fovapsmall = Chop[GaussianAperture[appix, appix dpp, newscale]];
    */
    float newscale, appix;

    //read parameters value
    FoveaScale=StrToFloat(Edit_FoveaScale->Text);
    Beta=StrToFloat(Edit_Beta->Text);

    newscale=FoveaScale/sqrt(Beta);
    appix = 2*intRound(newscale/dpp);

    GaussianAperture(appix, appix*dpp, newscale, false);
    fovapsmall.clear();     //samper
    for(int i=0; i<appix; i++) {
        for(int j=0; j<appix; j++)
        {
            //if the value is too small, set to 0.0
            if((GATable[i*appix+j]<0.0000000001)&&(GATable[i*appix+j]>-0.0000000001))
                fovapsmall.push_back(0.0);
            else
                fovapsmall.push_back(GATable[i*appix+j]);
        }
    }

    ShowFilterImg(fovapsmall, appix, appix, 1);
    Shape3->Brush->Color=clLime;
    Application->ProcessMessages();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::SSORCSFFilter(int pixelsW, int pixelsH, float degreesW, float degreesH)
{
    /*SSORCSFFilter[pixels_, degrees_, type_, parameters_] :=
        Module[{yhalf, xhalf},
        {yhalf, xhalf} = pixels/2;
        parameters[[1]] Wrap[
            Table[type[Sqrt[Plus @@ ( ({v, u}/degrees)^2)],
            Rest[parameters]], {v, -yhalf, yhalf - 1}, {u, -xhalf,
            xhalf - 1}]]]
    */
    int tempi, tempj;
    float csftemp, sech1, sech2;
    float f, f0, f1, loss, p;

    //read the parameters
    f=StrToFloat(Edit_f->Text);
    f0=StrToFloat(Edit_f0->Text);
    f1=StrToFloat(Edit_f1->Text);
    loss=StrToFloat(Edit_loss->Text);
    p=StrToFloat(Edit_p->Text);
    csfMask.clear();        //samper
    for(int i=0; i<pixelsH; i++){        //H
        for(int j=0; j<pixelsW; j++)     //W
        {
            tempi=i-(pixelsH/2);
            tempj=j-(pixelsW/2);
            //CSFHPmH[f_, f0_:4.1726, f1_:1.3625, loss_:0.849337, p_:0.77859] :=
            //    Sech[(f/f0)^p]-loss Sech[f/f1]
            //csftemp=sqrt((tempi*tempi+tempj*tempj)/(degrees*degrees));
            csftemp=sqrt(pow((tempi/degreesH),2)+pow((tempj/degreesW),2));
            //csftemp *= f;
            sech1=(float)2/(exp(pow((csftemp/f0),p))+exp(-pow((csftemp/f0),p)));
            sech2=(float)2/(exp(csftemp/f1)+exp(-(csftemp/f1)));

            csfMask.push_back((sech1-(loss*sech2))*f);
        }
    }
}

//----------------------------------------------------------------------------
void __fastcall TForm1::SSOObliqueFilter(int pixelsW, int pixelsH, float degreesW, float degreesH)
{
    /*
    SSOObliqueFilter[pixels_, degrees_, type_, parameters_] := 
        Module[{yhalf, xhalf},
        {yhalf, xhalf} = pixels/2;
        Wrap[Table[type[Sequence @@ ({v, u}/degrees), parameters], {v, -yhalf,
            yhalf - 1}, {u, -xhalf, xhalf - 1}]]]
    */
    float f, th;
    int tempi, tempj;
    float dtempi, dtempj;
    float fscale, corner, blitemp;

    //read parameters value
    fscale=StrToFloat(Edit_fscale->Text);
    corner=StrToFloat(Edit_corner->Text);
    bliMask.clear();        //samper
    for(int i=0; i<pixelsH; i++){           //H
        for(int j=0; j<pixelsW; j++)        //W
        {
            tempi=i-(pixelsH/2);
            tempj=j-(pixelsW/2);
            /*ObliqueEffect[u_, v_, fscale_:13.57149, corner_:3.481] :=
                Module[{tmp, f = Sqrt[u^2 + v^2], th = ArcTan[u, v]},
                If[f <= corner, 1, 1 -  (1 - Exp[-(f - corner)/fscale])(Sin[2th]^2)]]
            */
            dtempi=(float)tempi/degreesH;
            dtempj=(float)tempj/degreesW;

            f=sqrt(dtempi*dtempi+dtempj*dtempj);
            if(dtempi==0)
                th=0.0;
            else
                th=atan(dtempj/dtempi);
            if(f<=corner)
                blitemp=1;
            else
                blitemp=1-(1-exp(-(f-corner)/fscale))*(pow(sin(2*th), 2));

            bliMask.push_back(blitemp);
        }
    }
}

//-----------------------------------------------------------------------------
void __fastcall TForm1::CSFFilter()
{
    float temp, degreeW, degreeH;
    int count;

    degreeW=actW*dpp;
    degreeH=actH*dpp;

    SSORCSFFilter(actW, actH, degreeW, degreeH);
    SSOObliqueFilter(actW, actH, degreeW, degreeH);
    rcsf.clear();       //samper
    oef.clear();        //samper
    csf.clear();        //samper
    for(int i=0; i<actH; i++){
        for(int j=0; j<actW; j++)
        {
            count = i*actW+j;
            //rcsf = Chop[SSORCSFFilter[pixels, degrees, csftype, csfparams]]
            if((csfMask[count]<0.0000000001)&&(csfMask[count]>-0.0000000001))
                rcsf.push_back(0.0);
            else
                rcsf.push_back(csfMask[count]);

            //oef = Chop[SSOObliqueFilter[pixels, degrees, oetype, oeparams ]]
            if((bliMask[count]<0.0000000001)&&(bliMask[count]>-0.0000000001))
                oef.push_back(0.0);
            else
                oef.push_back(bliMask[count]);

            temp=rcsf[count]*oef[count];
            //csf = Chop[rcsf oef]
            if((temp<0.0000000001)&&(temp>-0.0000000001))
                csf.push_back(0.0);
            else
                csf.push_back(temp);
        }
    }
  
    CSFFilterImg->Height=150*((float)actH/(float)actW);
    ShowFilterImg(csf, actW, actH, 0);       //ShowFilterImg

    //warp
    float warptemp;
    int warpcount=(actH*actW)/2;
    int count1;
    //が传
    for(int i=0; i<warpcount; i++){
        count = i+warpcount;
        warptemp=csf[i];
        csf[i]=csf[count];
        csf[count]=warptemp;
    }
    //オが传
    for(int i=0; i<actH; i++){
        for(int j=0; j<(actW/2); j++)
        {
            count = i*actW+j;
            count1= count+(actW/2);
            warptemp=csf[count];
            csf[count]=csf[count1];
            csf[count1]=warptemp;
        }
    }
    //ShowFilterImg(csf, actW, actH, 0);       //ShowFilterImg
    Shape2->Brush->Color=clLime;
    Application->ProcessMessages();
}

//--------------------------------------------------------------------------
std::vector<float> TForm1::ToLuminance(std::vector<float> img, int lmax)
{
    //ToLuminance[g_, lmax_] := (lmax/2) (1 + (g - 128)/127)
    int width, height;
    std::vector<float> luminance;

    width=actW;
    height=actH;
    luminance.clear();      //samper
    for(int i=0; i<height; i++)
        for(int j=0; j<width; j++)
            luminance.push_back((lmax/2)*(1+(img[i*width+j]-128)/127));

    return  luminance;
}

//----------------------------------------------------------------------------
void __fastcall TForm1::Btn_ReadImgClick(TObject *Sender)
{
    Graphics::TBitmap *OriBitmap;
    OriImageR.clear();      //samper
    if (OpenPictureDialog1->Execute())
    {
        Image1->Picture->LoadFromFile(OpenPictureDialog1->FileName);

        OriBitmap = Image1->Picture->Bitmap;  //read the image information
        ImageW=OriBitmap->Width;
        ImageH=OriBitmap->Height;

        Image1->Height=400*((float)ImageH/(float)ImageW);

        //assign the orignal color value to OriBuf
        for(int i=0; i<ImageH; i++)
            for(int j=0; j<ImageW; j++)
                OriImageR.push_back(GetRValue(OriBitmap->Canvas->Pixels[j][i]));
        
        Btn_MakeRef->Enabled=true;
    }
}
//---------------------------------------------------------------------------

void __fastcall TForm1::CCI()
{
    //contrast = luminance/lmean - 1 = (luminance-lmean)/lmean
    ContrastTestImg.clear();        //samper
    for(int i=0; i<actH; i++)
        for(int j=0; j<actW; j++)
            ContrastTestImg.push_back(Lum_test[i*actW+j]/lmean);

    Contrast_TestImg->Height=150*((float)actH/(float)actW);    //assign the height value to display

    ShowFilterImg(ContrastTestImg, actW, actH, 3);
    Shape5->Brush->Color=clLime;
    Application->ProcessMessages();
}
//---------------------------------------------------------------------------

void __fastcall TForm1::SSOBorder()
{
    //pixels = dimensions of the image; degrees = height and width of the image in degrees;
    //scale = scale of the Gaussian in degrees; gain = maximum value of the image
    /*
    SSOBorder[pixels_, degrees_, scale_, gain_:1] := Module[{},
    {dppy, dppx} = degrees/pixels;
    Table[1 - gain Exp[- Pi (Min[dppx(x - 1) , dppy(y - 1) , dppx(pixels[[2]] - x) ,
         dppy(pixels[[1]] - y)]/scale)^2], {y, pixels[[1]]}, {x, pixels[[2]]}]]
    */
    float dppx, dppy;
    float scale, gain;
    float temp1, temp2, temp3, temp4, min;

    //read parameters value
    scale=StrToFloat(Edit_BoScale->Text);
    gain=StrToFloat(Edit_BoGain->Text);

    dppx=dpp;
    dppy=dpp;
    border.clear();     //samper
    for(int i=0; i<actH; i++){
        for(int j=0; j<actW; j++)
        {
            temp1=dppx*(i-1);
            temp2=dppy*(j-1);
            temp3=dppx*(actH-i);
            temp4=dppy*(actW-j);
            min=SSOMin(temp1, temp2, temp3, temp4);
            border.push_back(1-gain*exp(-pi*pow((min/scale),2)));
        }
    }

    BorderImg->Height=150*((float)actH/(float)actW);
    ShowFilterImg(border, actW, actH, 4);
    Shape4->Brush->Color=clLime;
    Application->ProcessMessages();
}
//--------------------------------------------------------------------------

float __fastcall TForm1::SSOMin(float t1, float t2, float t3, float t4)
{
    //search the minimal value and return
    float mini, temp1, temp2;

    temp1=(t1<t2)?t1:t2;
    temp2=(t3<t4)?t3:t4;
    mini=(temp1<temp2)?temp1:temp2;

    return mini;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::GaussianFilter(float scale)
{
    //pixels = dimensions of the image;
    //degrees = height and width of the image in degrees;
    //scale = scale of the Gaussian in degrees
    /*
    GaussianFilter[pixels_, degrees_, scale_] := Module[{yhalf, xhalf},
    {yhalf, xhalf} = pixels/2;
    Wrap[Table[Exp[-Pi(Plus @@ (({v, u} scale/degrees)^2 ))], {v, -yhalf,
          yhalf - 1}, {u, -xhalf, xhalf - 1}]]]
    */
    float tempi, tempj;
    float degreesW, degreesH, g;

    degreesW=actW*dpp;
    degreesH=actH*dpp;
    
    GauFilter.clear();      //samper
    for(int i=0; i<actH; i++){
        for(int j=0; j<actW; j++)
        {
            tempi=i-(actH/2);
            tempj=j-(actW/2);

            tempi=pow((tempi*(scale/degreesH)),2);
            tempj=pow((tempj*(scale/degreesW)),2);
            g=exp(-pi*(tempi+tempj));

            //warp ?
            GauFilter.push_back(g);
        }
    }
}
//--------------------------------------------------------------------------

float __fastcall TForm1::MeanImg()
{
    //compute the mean value of an image
    float mean, sam=0.0;

    for(int i=0; i<actH; i++)
        for(int j=0; j<actW; j++)
            sam+=Lum_ref[i*actW+j];    //luminance reference image

    mean=sam/(actW*actH);

    return mean;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::TempImg()
{
    /*
    (* Create local luminance images or scalar mean luminance*)
    lfilter = Chop[SSOLuminanceFilter[pixels, degrees, lumscale]] ;
    lmean = N[ Mean[Flatten[Last[#]]] &  /@ luminance];
    */
    int temp;
    GaussianFilter(0.1);
    //chop the gauFilter

    lfilter.clear();        //samper
    for(int i=0; i<actH; i++){
        for(int j=0; j<actW; j++)
        {
            if(GauFilter[i*actW+j]<0.0000000001)  //if the value is too small, set to 0.0
                lfilter.push_back(0.0);
            else
                lfilter.push_back(GauFilter[i*actW+j]);
        }
    }

    lmean=MeanImg();
    temp=lmean*1000;
    lmean=(float)temp/1000.0;
    meanlumin->Caption=lmean;
}
//----------------------------------------------------------------------------

std::vector<float> TForm1::ListConvolve(std::vector<float> ker, std::vector<float> list, int ww, int hh)
{
    // retrun a convolution result
    std::vector<float> ConRest, newMask;
//    int size, hhh, www, sizeh;
    int size, sizeh;
    int tempi, tempj;
    float temp, sum;
    //FFT
    fftw_complex *in, *out_mask, *out_image;
    fftw_plan p;

    size=sqrt(ker.size());
//    hhh=hh+size;
//    www=ww+size;
    //hhh=hh;    //test for no zero padding
    //www=ww;

/*    //do DFT
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * hhh * www);
    out_mask = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * hhh * www);
    out_image = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * hhh * www);

    //for mask ========================
    //make the mask size is the same as image, and initialize the mask
    if((size!=hhh)||(size!=www))
    {
    for(int i=0; i<hhh; i++){
        for(int j=0; j<www; j++)
        {
            if((i>=(hhh/2-size/2))&&(i<(hhh/2+size/2))&&(j>=(www/2-size/2))&&(j<(www/2+size/2))){
                tempi=(size/2)-((hhh/2)-i);
                tempj=(size/2)-((www/2)-j);
                newMask.push_back(ker[tempi*size+tempj]);
            }
            else
                newMask.push_back(0.0);
        }
    }
    }

    //warp   ==================================================
    float warptemp;
    int warpcount=(hhh*www)/2;
    int count, count1;
    //が传
    for(int i=0; i<warpcount; i++){
        count = i+warpcount;
        warptemp=newMask[i];
        newMask[i]=newMask[count];
        newMask[count]=warptemp;
    }
    //オが传
    for(int i=0; i<hhh; i++){
        for(int j=0; j<(www/2); j++)
        {
            count = i*www+j;
            count1= count+(www/2);
            warptemp=newMask[count];
            newMask[count]=newMask[count1];
            newMask[count1]=warptemp;
        }
    }
    //=========================================================

    //ShowFilterImg(newMask,www,hhh, 7);
    size=hhh*www;
    for(int i=0; i<size; i++){
        in[i][0] = (double)newMask[i];
        in[i][1] = 0.0;
    }
    p = fftw_plan_dft_2d(hhh, www, in, out_mask, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    //for image ========================
    //make the mask size is the same as image, and initialize the mask
    newMask.clear();
    if((hh!=hhh)||(ww!=www))
    {
    for(int i=0; i<hhh; i++){
        for(int j=0; j<www; j++)
        {
            if((i>=(hhh/2-hh/2))&&(i<(hhh/2+hh/2))&&(j>=(www/2-ww/2))&&(j<(www/2+ww/2))){
                tempi=(hh/2)-((hhh/2)-i);
                tempj=(ww/2)-((www/2)-j);
                newMask.push_back(list[tempi*ww+tempj]);
            }
            else
                newMask.push_back(0.0);
        }
    }
    }

    for(int i=0; i<size; i++){
        in[i][0] = (double)newMask[i];
        //in[i][0] = (double)list[i];      //test for no zero padding
        in[i][1] = 0.0;
    }
    p = fftw_plan_dft_2d(hhh, www, in, out_image, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    //out_mask * out_image
    for(int i=0; i<size; i++){
        in[i][0] = out_mask[i][0]*out_image[i][0]-out_mask[i][1]*out_image[i][1];
        in[i][1] = out_mask[i][1]*out_image[i][0]+out_mask[i][0]*out_image[i][1];
        //in[i][0] = out_image[i][0];
        //in[i][1] = out_image[i][1];
    }

    //back transfer
    p = fftw_plan_dft_2d(hhh, www, in, out_image, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    for(int i=0; i<size; i++){
        //ConRest.push_back((int)(out_image[i][0]/(www*hhh)));
        out_image[i][0]= (int)(out_image[i][0]/size);
    }

    for(int i=0; i<hhh; i++){
        for(int j=0; j<www; j++)
        {
            if((i>=(hhh/2-hh/2))&&(i<(hhh/2+hh/2))&&(j>=(www/2-ww/2))&&(j<(www/2+ww/2)))
                ConRest.push_back((int)out_image[i*www+j][0]);
        }
    }
*/
    //original convolution
    ConRest.clear();        //samper
    for(int i=0; i<hh; i++){
        for(int j=0; j<ww; j++)
        {
            sum=0.0;
            for(int k=0; k<size; k++){
                for(int l=0; l<size; l++)
                {
                    tempi=i+k-(size/2)+1;
                    tempj=j+l-(size/2)+1;
                    if((tempi>=hh)||(tempi<0)||(tempj>=ww)||(tempj<0))
                        temp=0.0;
                    else
                        temp=list[tempi*ww+tempj];

                    sum+=(ker[k*size+l]*temp);
                }
            }
            ConRest.push_back(sum);
        }
    }

    //release memory
    //fftw_destroy_plan(p);
    //fftw_free(in);
    //fftw_free(out_mask);
    //fftw_free(out_image);
    return ConRest;
}
//----------------------------------------------------------------------------

std::vector<float> TForm1::ConfinedConvolve(std::vector<float> kernel, std::vector<float> image, int W, int H)
{
    // ConfinedConvolve[kernel_, image_] := Module[{tmp, adjust, tot},
    // tmp = ListConvolve[kernel, image, {1, -1}, 0];
    // tot = Total[kernel, 2];
    // adjust = ListConvolve[kernel/tot, Array[1 &, Dimensions[image]], {1, -1}, 0];
    // tmp = tmp/adjust;
    // Take[tmp, Sequence @@ Transpose[{#/2 + 1, -#/2} &@Dimensions[kernel]]]]
    float tot=0.0;
    int kernelSize;
    std::vector<float> tmp, adjust, kernel1, idenImg;

    tmp = ListConvolve(kernel, image, W, H);

    kernelSize=kernel.size();
    for(int i=0; i<kernelSize; i++)
        tot+=kernel[i];

    //kernel/tot
    kernel1.clear();        //samper
    for(int i=0; i<kernelSize; i++)
        kernel1.push_back(kernel[i]/tot);

    //Array[1 &, Dimensions[image]]
    idenImg.clear();        //samper
    for(int i=0; i<W*H; i++)
        idenImg.push_back(1.0);

    adjust = ListConvolve( kernel1, idenImg, W, H);

    //tmp = tmp/adjust
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            if(adjust[i*W+j]!=0)
                tmp[i*W+j]/=adjust[i*W+j];
        }
    } 
    //ShowFilterImg(tmp,W,H, 7);
    return tmp;
}
//----------------------------------------------------------------------------

std::vector<float> TForm1::ListConvolve1D(std::vector<float> ker, std::vector<float> list)
{
    // retrun a convolution result
    std::vector<float> ConRest;
    int sizeker, sizelist;
    int tempi, tempj;
    float temp, sum;

    sizeker=ker.size();
    sizelist=list.size();
    ConRest.clear();        //samper
    for(int i=0; i<sizelist; i++){
        sum=0.0;
        for(int k=0; k<sizeker; k++){
            tempi=i+k-(sizeker/2)+1;
            if((tempi>=sizelist)||(tempi<0))
                temp=0.0;
            else
                temp=list[tempi];

            sum+=(ker[sizeker-k-1]*temp);
        }
        ConRest.push_back(sum);
    }

    return ConRest;
}
//----------------------------------------------------------------------------

std::vector<float> TForm1::ConfinedConvolve1D(std::vector<float> kernel, std::vector<float> image)
{
    float tot=0.0;
    int kernelSize, imagesize;
    std::vector<float> tmp, adjust, kernel1, idenImg;

    tmp = ListConvolve1D(kernel, image);

    kernelSize=kernel.size();
    for(int i=0; i<kernelSize; i++)
        tot+=kernel[i];

    //kernel/tot
    kernel1.clear();        //samper
    for(int i=0; i<kernelSize; i++)
        kernel1.push_back(kernel[i]/tot);

    //Array[1 &, Dimensions[image]]
    imagesize=image.size();
    idenImg.clear();        //samper
    for(int i=0; i<imagesize; i++)
        idenImg.push_back(1.0);

    adjust = ListConvolve1D(kernel1, idenImg);

    //tmp = tmp/adjust
    for(int i=0; i<imagesize; i++)
        tmp[i]/=adjust[i];

    return tmp;
}
//----------------------------------------------------------------------------

std::vector<float> TForm1::DownSampleF(std::vector<float> img, int down)
{
    float sum, ave;
    std::vector<float> downrest;
    
    downrest.clear();       //samper
    for(int i=0; i<ImageH; i=i+down){
        for(int j=0; j<ImageW; j=j+down)
        {
            sum=0.0;
            for(int k=0; k<down; k++)
                for(int l=0; l<down; l++)
                    sum+=img[(i+k)*ImageW+(j+l)];

            downrest.push_back(sum/(down*down));
        }
    }

    return downrest;
}
//-----------------------------------------------------------------------------

std::vector<float> TForm1::ActiveRectangle(std::vector<float> image, float cropmargin, int scale, int lengththreshold, int activeW, int activeH)
{
    //image = the image;
    //margin = a margin to crop inside the active region;
    //scale = the scale of a Gaussian derivative filter;
    //lengththreshold  (see FindEdges);
    float min, max,  temp;   //threshold,
    int size, half, num, Wp[2], Hp[2];
    std::vector<float> unitGau, unitGautmp, line, tmp0, boundar;

    /*size=image.size();
    for(int i=0; i<size; i++)
    {
        if(image[i]<min)
            min=image[i];
        if(image[i]>max)
            max=image[i];
    }
    threshold = -0.1*(min-max); */
    EdgeScale=StrToFloat(Edit_EdgeScale->Text);
    //edge filter
    half=2*EdgeScale;
    for(int i=0; i<2*half; i++)
    {
        num=i-half;
        temp=2*exp(-pi*(num*num)/(EdgeScale*EdgeScale))*pi*num;
        if(temp==0)
            unitGautmp.push_back(0);
        else
            unitGautmp.push_back(-temp/(EdgeScale*EdgeScale));
    }

    //show the profile of UnitGaussianDerivative
    //for(int i=0; i<2*half; i++)
    //    Chart1->Series[0]->AddXY(i, unitGau[i]);

    //warp the UnitGaussianDerivative
    for(int i=0; i<2*half; i++){
        num=(i+half);
        num= (num>=(2*half))?(num-2*half):num;
        unitGau.push_back(unitGautmp[num]);
    }

    //mean of downsample image (vertical) ====================================
    for(int i=0; i<activeH; i++)
    {
        temp=0.0;
        for(int j=0; j<activeW; j++)
        {
            temp+=image[i*activeW+j]/activeW;
        }
        line.push_back(temp); //average
    }

    tmp0=ConfinedConvolve1D(unitGau, line);

    //find the position of min & max
    min=65535.0;  max=-65535.0;
    for(int i=0; i<activeH; i++)
    {
        if(tmp0[i]<min){
            min=tmp0[i];
            Hp[0]=i;        //position of min
        }
        if(tmp0[i]>max){
            max=tmp0[i];
            Hp[1]=i;         //position of max
        }
    }

    //mean of downsample image (horizontal) =================================
    line.clear();
    tmp0.clear();
    for(int j=0; j<activeW; j++)
    {
        temp=0.0;
        for(int i=0; i<activeH; i++)
        {
            temp+=image[i*activeW+j]/activeH;
        }
        line.push_back(temp);
    }

    tmp0=ConfinedConvolve1D(unitGau, line);

    //find the position of min & max
    min=65535.0;  max=-65535.0;
    for(int i=0; i<activeW; i++)
    {
        if(tmp0[i]<min){
            min=tmp0[i];
            Wp[0]=i;        //position of min
        }
        if(tmp0[i]>max){
            max=tmp0[i];
            Wp[1]=i;         //position of max
        }
    }

    //chech the size of horizontal and vertical is even or odd
    num=Wp[1]-Wp[0];
    if(!(num%2))        //if the size is odd
        Wp[1]=Wp[1]+1;  //make the size is even

    num=Hp[1]-Hp[0];
    if(!(num%2))        //if the size is odd
        Hp[1]=Hp[1]+1;  //make the size is even

    //save the position value
    boundar.push_back(Hp[0]+CropMargin);
    boundar.push_back(Hp[1]-CropMargin);
    boundar.push_back(Wp[0]+CropMargin);
    boundar.push_back(Wp[1]-CropMargin);
    //boundar.push_back(Hp[0]+CropMargin);
    //boundar.push_back(Hp[1]-CropMargin);

    return boundar;
}
//----------------------------------------------------------------------------

void __fastcall TForm1::Btn_MakeRefClick(TObject *Sender)
{
    float ppixels, pdegrees, temp;
    float rpixels, rdegrees, odpp;
    std::vector<float> pflt, ar, dar, rfilter, range;
    int newsizeW, newsizeH;

    //set the parameters
    PreScale=StrToFloat(Edit_PreScale->Text);
    DownSample=StrToInt(Edit_DownSample->Text);
    ReferenceScale=StrToFloat(Edit_RefScale->Text);
    CropMargin=StrToFloat(Edit_Margin->Text);
    dpp=1/(StrToFloat(Edit_dpp->Text));

    //Create Pre-filter
    //ppixels=2 Round(pscale/dpp)
    odpp=dpp;
    ppixels=PreScale/dpp;
    temp=ppixels-(int)ppixels;
    ppixels=(temp>0.5)?((int)ppixels+1):(int)ppixels;
    ppixels*=2;

    pdegrees = ppixels*dpp;

    GaussianAperture(ppixels, pdegrees, PreScale, True);

    for(int i=0; i<(ppixels*ppixels); i++)
         pflt.push_back(GATable[i]);

    //Filter Image
    ar=ConfinedConvolve(pflt, OriImageR, ImageW, ImageH);
    //ShowImg(ar, ImageW, ImageH);
    
    //Downsample Image
    dar=DownSampleF(ar, DownSample);
    //ShowImg(dar, ImageW/DownSample, ImageH/DownSample);

    dpp = dpp*DownSample;
    newsizeW = ImageW/DownSample;
    newsizeH = ImageH/DownSample;

    //Extract Active Rectangle
    range=ActiveRectangle(dar, CropMargin, 8, 5, ImageW/DownSample, ImageH/DownSample);
    //range.push_back(0);    //for ModelFest
    //range.push_back(ImageH-1);
    //range.push_back(0);
    //range.push_back(ImageW-1);

    actW=range[3]-range[2]+1;
    actH=range[1]-range[0]+1;
    ROI_Img->Height=200*((float)actH/(float)actW);    //assign the height value to display
    Ref_ROI_Img->Height=ROI_Img->Height;
    
    ar.clear();
    testImg.clear();    //samper
    for(int i=0; i<newsizeH; i++){
        for(int j=0; j<newsizeW; j++)
        {
            if((j>=range[2])&&(j<=range[3])&&(i>=range[0])&&(i<=range[1]))
                testImg.push_back(dar[i*newsizeW+j]);
        }
    }

    //just for show the extract active rectangle ------------
    for(int i=0; i<newsizeH; i++){
        for(int j=0; j<newsizeW; j++)
        {
            //if((j==range[2])||(j==range[3])||(i==range[0])||(i==range[1]))
            //    dar[i*newsizeW+j]=255;
            if((j<=range[2]+2)&&(j>=range[2]-2))
                dar[i*newsizeW+j]=255;
            if((j<=range[3]+2)&&(j>=range[3]-2))
                dar[i*newsizeW+j]=255;
            if((i<=range[0]+2)&&(i>=range[0]-2))
                dar[i*newsizeW+j]=255;
            if((i<=range[1]+2)&&(i>=range[1]-2))
                dar[i*newsizeW+j]=255;
        }
    } //------------------------------------------------------

    //Create Reference Filter
    //rpixels = 2 Round[rscale/dpp];
    rpixels=ReferenceScale/dpp;
    temp=rpixels-(int)rpixels;
    rpixels=(temp>0.5)?((int)rpixels+1):(int)rpixels;
    rpixels*=2;

    rdegrees=rpixels*dpp;

    GaussianAperture(rpixels, rdegrees, ReferenceScale, True);

    for(int i=0; i<(rpixels*rpixels); i++)
         rfilter.push_back(GATable[i]);

    //Filter Image
    referImg=ConfinedConvolve(rfilter, testImg, actW, actH);
    ShowImg(testImg, actW, actH, 1);
    ShowImg(referImg, actW, actH, 2);

    //show the information of the make reference
    TabSheet2->Show();
    ShowImg(dar, newsizeW, newsizeH, 0);

    RefSheet_OriW->Caption=ImageW;
    RefSheet_OriH->Caption=ImageH;
    RefSheet_OriDpp->Caption=odpp;
    RefSheet_DownW->Caption= newsizeW;
    RefSheet_DownH->Caption= newsizeH;
    RefSheet_DownDpp->Caption= dpp;
    RefSheet_FilW->Caption=rpixels;
    RefSheet_FilH->Caption=rpixels;
    RefSheet_FilDpp->Caption=rdegrees;
    RefSheet_OriMarge->Caption=CropMargin;
    FinImgW->Caption=actW;
    FinImgH->Caption=actH;
    Shape1->Brush->Color=clLime;
    Btn_SSO->Enabled=true;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::ShowImg(std::vector<float> img, int W, int H, int type)
{
    //show image in the display
    Graphics::TBitmap *TheBitmap, *TempBitmap;

    TheBitmap=Image1->Picture->Bitmap;
    TempBitmap= new Graphics::TBitmap();
    TempBitmap->Assign(TheBitmap);
    TempBitmap->Height=H;
    TempBitmap->Width=W;

    for(int i=0; i<H; i++)
        for(int j=0; j<W; j++)
            TempBitmap->Canvas->Pixels[j][i] = TColor(RGB(img[i*W+j], img[i*W+j], img[i*W+j]));

    switch(type)
    {
    case 0:
        RefImg->Height=256*((float)H/(float)W);
        RefImg->Picture->Bitmap=TempBitmap;
        break;
    case 1:
        ROI_Img->Picture->Bitmap=TempBitmap;
        break;
    case 2:
        Ref_ROI_Img->Picture->Bitmap=TempBitmap;
        break;
    }
}
//----------------------------------------------------------------------------

void __fastcall TForm1::ShowFilterImg(std::vector<float> img, int W, int H, int type)
{
    //show image in the display
    Graphics::TBitmap *TheBitmap, *TempBitmap;

    TheBitmap=Image1->Picture->Bitmap;
    TempBitmap= new Graphics::TBitmap();
    TempBitmap->Assign(TheBitmap);
    TempBitmap->Height=H;
    TempBitmap->Width=W;

    float min=1000000.0, max=-1000000.0, temp, temp1;

    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++)
        {
            if(img[i*W+j]<min)
                min=img[i*W+j];
            if(img[i*W+j]>max)
                max=img[i*W+j];
        }
    }

    temp=max-min;
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++)
        {
            temp1=255*((img[i*W+j]-min)/temp);
            TempBitmap->Canvas->Pixels[j][i] = TColor(RGB(temp1, temp1, temp1));
        }
    }

    switch(type)
    {
    case 0:
        CSFFilterImg->Picture->Bitmap=TempBitmap;
        break;
    case 1:
        ApertureImg->Picture->Bitmap=TempBitmap;
        break;
    case 2:
        Lum_TestImg->Picture->Bitmap=TempBitmap;
        break;
    case 3:
        Contrast_TestImg->Picture->Bitmap=TempBitmap;
        break;
    case 4:
        BorderImg->Picture->Bitmap=TempBitmap;
        break;
    case 5:
        Bod_TestImg->Picture->Bitmap=TempBitmap;
        break;
    case 6:
        FFT_TestImg->Picture->Bitmap=TempBitmap;
        break;
    case 7:
        TempBitmap->Height=H;
        TempBitmap->Width=W;
        JNDImgC->Picture->Bitmap=TempBitmap;
        break;
    }
}
//----------------------------------------------------------------------------

void __fastcall TForm1::ShowColorJND(std::vector<float> img, int W, int H)
{
    //show image in the display
    Graphics::TBitmap *TheBitmap, *TempBitmap;
    float jndMax=0.0, jndMin=1000.0, jndCen;
//    float jndThd1=1.5, jndThd2=2.0;

    TheBitmap=Image1->Picture->Bitmap;
    TempBitmap= new Graphics::TBitmap();
    TempBitmap->Assign(TheBitmap);
    TempBitmap->Height=H;
    TempBitmap->Width=W;
    BYTE colorR, colorG, colorB;

    //find the max and min value
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++)
        {
            if(img[i*W+j]>=jndMax)
                jndMax=img[i*W+j];
            if(img[i*W+j]<=jndMin)
                jndMin=img[i*W+j];
        }
    }
    jndCen=(jndMax+jndMin)/2;

    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++)
        {
            if((img[i*W+j]>jndCen)&&(img[i*W+j]<=jndMax))
//            if((img[i*W+j]>jndThd1)&&(img[i*W+j]<jndThd2))
            {
                colorR=255*((img[i*W+j]-jndCen)/(jndMax-jndCen));
                colorG=255*((jndMax-img[i*W+j])/(jndMax-jndCen));
                colorB=0;
            }

            else if(img[i*W+j]<=jndCen)
//            else if(img[i*W+j]>=jndThd2)
            {
                colorR=0;
                colorG=255*((img[i*W+j]-jndMin)/(jndCen-jndMin));
                colorB=0;
            }
            
            TempBitmap->Canvas->Pixels[j][i] = TColor(RGB(colorR, colorG, colorB));
        }
    }

    JNDImgC->Picture->Bitmap=TempBitmap;
}
//----------------------------------------------------------------------------

void __fastcall TForm1::Btn_ReadTiffClick(TObject *Sender)
{
    Mura::TwoDStorage<unsigned short> buf;
    std::string header;
//    int result;
    //Graphics::TBitmap *TheBitmap, *TempBitmap;

    //TheBitmap=Image2->Picture->Bitmap;
    //TempBitmap= new Graphics::TBitmap();
    //TempBitmap->Assign(TheBitmap);

    //BYTE colorR, colorG, colorB;

    if(OpenDialog1->Execute())
        Mura::TiffIO<unsigned short>::InputSimpleGrayTiff(OpenDialog1->FileName.c_str(), buf, header);
//        result = Mura::TiffIO<unsigned short>::InputSimpleGrayTiff(OpenDialog1->FileName.c_str(), buf, header);

    ImageW = buf.Ncols();
    ImageH = buf.Nrows();

    //Image1->Width = ImageW / 7;
    //Image1->Height = ImageH / 7;

    Image1->Height=400*((float)ImageH/(float)ImageW);
    //TempBitmap->Height=Image1->Height;
    //TempBitmap->Width=400;
    //assign the orignal color value to OriBuf
    OriImageR.clear();  //samper
    for(int i=0; i<ImageH; i++)
    for(int j=0; j<ImageW; j++)
    {
        if(i%7==0 && j%7==0)
        {
            unsigned char c = buf[i][j] / 16;
            Image1->Canvas->Pixels[j/7][i/7] = (TColor)(c*256*256 + c*256 + c);
            //TempBitmap->Canvas->Pixels[j/7][i/7]= c*256*256 + c*256 + c;
        }
        //Image1->Canvas->Pixels[j][i] = buf[i][j];
        OriImageR.push_back((float)buf[i][j]);
    }

    //Image1->Picture->Bitmap=TempBitmap;
    Btn_MakeRef->Enabled=true;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::SaveJNDBtnClick(TObject *Sender)
{
    if(SavePictureDialog1->Execute())
    {
        JNDImgC->Picture->SaveToFile(SavePictureDialog1->FileName);
    }
}

//---------------------------------------------------------------------------
bool TForm1::BatchProcess(std::string inFileName, std::string outFileName)
{
    Mura::TwoDStorage<unsigned short> buf;
    Mura::TwoDStorage<float> JNDmap;
    std::string header;

    if(0!=Mura::TiffIO<unsigned short>::InputSimpleGrayTiff(inFileName.c_str(), buf, header))
        return false;

    ImageW = buf.Ncols();
    ImageH = buf.Nrows();
    Image1->Height=400*((float)ImageH/(float)ImageW);

    OriImageR.clear();  //samper
    for(int i=0; i<ImageH; i++)
    for(int j=0; j<ImageW; j++)
    {
        if(i%7==0 && j%7==0)
        {
            unsigned char c = buf[i][j] / 16;
            Image1->Canvas->Pixels[j/7][i/7] = (TColor)(c*256*256 + c*256 + c);
        }
        OriImageR.push_back((float)buf[i][j]);
    }

    Btn_MakeRefClick(this);
    Btn_SSOClick(this);

    JNDmap.Resize(actH, actW);
    for(int i=0; i<actH; i++)
    for(int j=0; j<actW; j++)
        JNDmap[i][j] = jnd[i*actW+j];
    if(0!=Mura::TiffIO<float>::OutputSimpleGrayTiff(outFileName.c_str(), JNDmap, 32, ""))
        return false;

    return true;
}

//---------------------------------------------------------------------------
void __fastcall TForm1::Btn_ReadBatClick(TObject *Sender)
{
    std::ifstream BatchIn;
    std::string buf1, buf2;
    String msg;

    if(OpenDialog1->Execute())
    {
        BatchIn.open(OpenDialog1->FileName.c_str());
        while(BatchIn >> buf1 && BatchIn >> buf2)
        {
            if(BatchProcess(buf1, buf2))
                Caption = String("Process Complete: ") + String(buf1.c_str());
            else
                Caption = String("Error Occurred: ") + String(buf1.c_str());
        }
    }
}

//---------------------------------------------------------------------------
