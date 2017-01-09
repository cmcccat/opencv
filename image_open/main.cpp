#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<math.h>
using namespace cv;
using namespace std;
int Abs(int a)
{
    if(a>=0)
        return a;
    else
        return -a;
}
int GlobalThread(Mat &src)
{
    int channels = 0;
    MatND dstHist;
    int histSize[]={256};
    float midRanges[]={0,256};
    const float *ranges[]={midRanges};
    calcHist(&src,1,&channels,Mat(),dstHist,1,histSize,ranges,true,false);
    normalize(dstHist,dstHist,1.0,0.0,NORM_MINMAX);
    int i,T,T1=0;
    double z = 0,z1 = 0,z2 = 0;
    double mg = 0,m1 = 0,m2 = 0;
    for (i=0;i<256;i++)
    {
       z  += dstHist.at<float>(i);
       mg += dstHist.at<float>(i)*i;
    }
    T=(int)(mg/z);
    do
    {
        T1=T;
        z1=0;
        z2=0;
        for (i=0;i<=T1;i++)
        {
            z1 += dstHist.at<float>(i);
            m1 += dstHist.at<float>(i)*i;
        }
        z2 = z-z1;
        m2 = mg-m1;
        if(z1)
           m1=m1/z1;
        else
           m1=0;
        if(z2)
            m2=m2/z2;
        else
            m2=0;
        T=(int)((m1+m2)/2);
    } while(T1!=T);
    return T1;
}
void gradientGray(Mat &src,Mat &mag)
{
    const int H=src.rows,W=src.cols;
    Mat Ix(H,W,CV_32S),Iy(H,W,CV_32S);
    Mat GRD(H,W,CV_32SC3);
    for(int y=0;y<H;y++)
    {
        Ix.at<int>(y,0)=abs(src.at<int>(y,1)-src.at<int>(y,0))*2;
        for(int x=1;x<W-1;x++)
            Ix.at<int>(y,x) = abs(src.at<char>(y,x+1)-src.at<char>(y,x-1));
        Ix.at<int>(y,W-1) = abs(src.at<char>(y,W-1)-src.at<char>(y,W-2))*2;
    }
    for(int x=0;x<W;x++)
    {
        Iy.at<int>(0,x)=abs(src.at<char>(1,x)-src.at<char>(0,x))*2;
        for(int y=1;y<H-1;y++)
            Iy.at<int>(y,x) = abs(src.at<char>(y+1,x)-src.at<char>(y-1,x));
        Iy.at<int>(H-1,x) = abs(src.at<char>(H-1,x)-src.at<char>(H-2,x))*2;
    }
    convertScaleAbs(min(Ix+Iy,255),mag);
}
void test(Mat &src,int m)
{
    Mat k;
    int H=src.rows,W=src.cols;
     blur(src,k,Size(m,m));
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {

                if(k.at<char>(i,j)<1.35*src.at<char>(i,j))
                src.at<char>(i,j)=k.at<char>(i,j);
        }
    }
}
void junzhi(Mat &src,Mat &out,int block)
{
    int H=src.rows,W=src.cols;
    int sum,p,q;
    int block2=block*block;
    int R=(block+1)/2;
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {
            sum=0;
            for(int k=-R+1;k<R;k++)
            {
                for(int z=-R+1;z<R;z++)
                {
                    p=i+k;
                    q=j+z;
                    if(p<0||p>=H||q<0||q>=W)
                    sum+=0;
                    else
                    sum+=src.at<char>(p,q);
                }
                out.at<char>(i,j)=(char)(sum/block2);
            }
        }
    }

}
void fangcha(Mat &src,Mat &blur,Mat &out,int block)
{
    int H=src.rows,W=src.cols;
    int sum,p,q;
    int block2=block*block;
    int R=(block+1)/2;
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {
            sum=0;
            for(int k=-R+1;k<R;k++)
            {

                for(int z=-R+1;z<R;z++)
                {
                    p=i+k;
                    q=j+z;
                    if(p<0||p>=H||q<0||q>=W)
                    sum+=(0-blur.at<char>(i,j))*(0-blur.at<char>(i,j));
                    else
                    sum+=(src.at<char>(p,q)-blur.at<char>(p,q))*(src.at<char>(p,q)-blur.at<char>(p,q));
                }
                out.at<int>(i,j)=(sum/block2);
            }
        }
    }

}
void White(Mat &src,int m)
{
    Mat k(src.rows,src.cols,CV_8UC1);
    int H=src.rows,W=src.cols;
    blur(src,k,Size(m,m));
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {
            if(k.at<char>(i,j)<1.03*src.at<char>(i,j))
            {
                src.at<char>(i,j)=255;
            }
            else
                src.at<char>(i,j)=0;
        }
    }
}

void on(Mat &src)
{
    int H=src.rows,W=src.cols;
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {
                if(src.at<char>(i,j)<0)
                src.at<char>(i,j)=255;
        }
    }
}
void Sauvola(Mat &src,int m)
{
    Mat k(src.rows,src.cols,CV_8UC1);
    Mat k1(src.rows,src.cols,CV_32SC1);
    int H=src.rows,W=src.cols;
    int thread;
    blur(src,k,Size(m,m));
    fangcha(src,k,k1,m);
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {
            if(k1.at<int>(i,j)>100)
            {

                int z1=k1.at<int>(i,j)/128-1;
                int z=1+0.5*z1;
                thread=k.at<char>(i,j)+z;
                if(src.at<char>(i,j)<thread)
                    src.at<char>(i,j)=0;
            }
            else
            src.at<char>(i,j)=255;
        }
    }
}
void drawrow(Mat &src)
{
    int H=src.rows,W=src.cols;
    double sum1=0,sum2=0;
    Mat dr=Mat::zeros(H,1,CV_32SC1);
    Mat ds=Mat::zeros(1,W,CV_32SC1);
    int max1=0,max2=0;
    Mat drawImage1 = Mat::zeros(H,W,CV_8UC3);
    Mat drawImage2 = Mat::zeros(H,W,CV_8UC3);
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {
           if(src.at<char>(i,j)==0)
           {
            dr.at<int>(i,0)++;
            ds.at<int>(0,j)++;
           }
        }
    }
     for(int i=0;i<H;i++)
     {
         if(max1<= dr.at<int>(i,0))
            max1=dr.at<int>(i,0);
            sum1+=dr.at<int>(i,0);
     }
     for(int j=0;j<W;j++)
     {
         if(max2<= ds.at<int>(0,j))
            max2=ds.at<int>(0,j);
            sum2+=ds.at<int>(0,j);
     }
     for(int i=0;i<H;i++)
     {
         int value1 = cvRound(dr.at<int>(i,0)*256*0.9/max1);
         line(drawImage1,Point(drawImage1.cols-1,i),Point(drawImage1.cols-1-value1,i),Scalar(255,255,255));
     }
     for(int j=0;j<W;j++)
     {
         int value2 = cvRound(ds.at<int>(0,j)*256*0.9/max2);
         line(drawImage2,Point(j,drawImage2.rows-1),Point(j,drawImage2.rows-1-value2),Scalar(255,255,255));

     }
     namedWindow("行累计值",WINDOW_NORMAL);
     namedWindow("列累计值",WINDOW_NORMAL);
     imshow("行累计值",drawImage1);
     imshow("列累计值",drawImage2);
     sum1/=H;
     sum2/=W;
     cout<<"max1"<<max1<<endl;
     cout<<"max2"<<max2<<endl;
     cout<<"sum1"<<sum1<<endl;
     cout<<"sum2"<<sum2<<endl;
     for(int i=0;i<H;i++)
    {
            if(dr.at<int>(i,0)<sum1*0.05)
            for(int j=0;j<W;j++)
            {
                src.at<char>(i,j)=255;
            }
    }
     for(int i=0;i<W;i++)
    {

            if(ds.at<int>(0,i)<sum2*0.3)
            for(int j=0;j<H;j++)
            {
                src.at<char>(j,i)=255;
            }

    }
}
int ostu(Mat &src)
{
    int channels = 0;
    MatND dstHist;
    int histSize[]={256};
    float midRanges[]={0,256};
    const float *ranges[]={midRanges};
    calcHist(&src,1,&channels,Mat(),dstHist,1,histSize,ranges,true,false);
    int Threshold = 0;
    double delta = 0;
    double mg = 0,Pm=0;
    for (int i=0;i<256;i++)
    {
       Pm  +=   dstHist.at<float>(i);
       mg += dstHist.at<float>(i)*i;
    }
    if(Pm)
        mg=mg/Pm;
    else
        mg=0;
    double m = 0, P1 = 0;
    for(int k=0;k<256;k++)
    {
        if(Pm)
        {
            P1 +=  dstHist.at<float>(k)/Pm;
            m +=  dstHist.at<float>(k)*k/Pm;
        }
        else;
        double t = mg*P1-m;
        double delta_tmp = t*t/(P1*(1-P1));
        if(delta_tmp>delta)
        {
            delta = delta_tmp;
            Threshold = k;
        }
    }
    return Threshold;
}
int erzhi1(Mat &src,int P)
{
    int P1=ostu(src);
    threshold(src,src,P1*0.8+P*0.2,255,CV_THRESH_BINARY);
}
int erzhi(Mat &src,int block,int m)
{

        Mat tmp2=src;
        int k=0;
        for(int i=0;i<block;i++)
    {
        for(int j=0;j<block;j++)
    {

        int px_sta=(int)tmp2.cols/block*i;
        int px_end=(int)tmp2.cols/block*(i+1)-1;
        int py_sta=(int)tmp2.rows/block*j;
        int py_end=(int)tmp2.rows/block*(j+1)-1;
        if(px_end>=tmp2.cols)
            px_end=tmp2.cols-1;
        if(py_end>=tmp2.rows)
            py_end=tmp2.rows-1;
        int Length=px_end-px_sta+1;
        int Height=py_end-py_sta+1;
        if(Length<50&&Height<50)
            return 0;
        Mat ROI=tmp2(Rect(px_sta,py_sta,Length,Height));
        Mat tmp_m,tmp_sd;
        meanStdDev(ROI,tmp_m,tmp_sd);
        //cout<<"均值"<<tmp_m.at<double>(0,0)<<endl;
        //cout<<"方差"<<tmp_sd.at<double>(0,0)<<endl;
        if(tmp_m.at<double>(0,0)<m)
        {

            ROI+=(char)(m/tmp_m.at<double>(0,0)-1)*ROI;
            on(ROI);
            erzhi(ROI,block,m);
            k++;
        }
        else if(tmp_m.at<double>(0,0)<2*m)
        {

            ROI-=(char)(1-m/tmp_m.at<double>(0,0))*ROI;
            on(ROI);
            erzhi(ROI,block,m);
            k++;
        }
        else
            k++;

    }
    }
    if(k==block*block)
        return 0;

}
void zhifngtu(Mat &src)
{
        int channels = 0;
        MatND dstHist;
        int histSize[]={256};
        float midRanges[]={0,256};
        const float *ranges[]={midRanges};
        calcHist(&src,1,&channels,Mat(),dstHist,1,histSize,ranges,true,false);
        Mat drawImage = Mat::zeros(Size(256,256),CV_8UC3);
        double g_dHistMaxValue;
        minMaxIdx(dstHist,0,&g_dHistMaxValue,0,0);
        for(int i=0;i<256;i++)
        {
            int value = cvRound(dstHist.at<float>(i)*256*0.9/g_dHistMaxValue);
            line(drawImage,Point(i,drawImage.rows-1),Point(i,drawImage.rows-1-value),Scalar(255,255,255));

        }
        imshow("直方图",drawImage);
}
int main()
{
    Mat image = imread("chun.jpg");
    if(!image.data)
    {
        cout<<"3.jpg 不存在！"<<endl;
        return -1;
    }
    Mat gray;
    cvtColor(image,gray,CV_RGB2GRAY);
    namedWindow("灰度图",WINDOW_NORMAL);
    imshow("灰度图",gray);
    Mat tmp,tmp1,tmp2;
    Mat tmp_m,tmp_sd;
    tmp = gray.clone();
    meanStdDev(tmp,tmp_m,tmp_sd);
    cout<<"均值"<<tmp_m.at<double>(0,0)<<endl;
    cout<<"方差"<<tmp_sd.at<double>(0,0)<<endl;
    int m=(int)tmp_m.at<double>(0,0);
    int m1=tmp_sd.at<double>(0,0);
    int block=3;
    double sum=0,sum1=0;
    for(int i=0;i<block;i++)
    {
        for(int j=0;j<block;j++)
        {
            int px_sta=(int)tmp.cols/block*i;
            int px_end=(int)tmp.cols/block*(i+1)-1;
            int py_sta=(int)tmp.rows/block*j;
            int py_end=(int)tmp.rows/block*(j+1)-1;
            if(px_end>=tmp.cols)
                px_end=tmp.cols-1;
            if(py_end>=tmp.rows)
                py_end=tmp.rows-1;
            int Length=px_end-px_sta+1;
            int Height=py_end-py_sta+1;
            Mat ROI=tmp(Rect(px_sta,py_sta,Length,Height));
            meanStdDev(ROI,tmp_m,tmp_sd);
            cout<<"均值"<<tmp_m.at<double>(0,0)<<endl;
            sum+=Abs(m-(int)tmp_m.at<double>(0,0));
            sum1+=Abs(m1-(int)tmp_sd.at<double>(0,0));
        }
    }
    sum/=9;
    sum1/=9;
    cout<<"sum:"<<sum<<endl;
    cout<<"sum1:"<<sum1<<endl;
    if(sum<9&&m>128&&sum1<9)
    {
        Mat out;
        int P=ostu(tmp);
        threshold(tmp,out,P,255,CV_THRESH_BINARY);
        namedWindow("均衡图—二值",WINDOW_NORMAL);
        imshow("均衡图—二值",out);
        imwrite("处理结果.jpg",out);

    }
    else
    {
        if(m<128)
        {
            int k=25,n=5;
            for(int i=0;i<n;i++)
                {
                    Mat element = getStructuringElement(MORPH_RECT,Size(k,k));
                    morphologyEx(tmp,tmp1,MORPH_BLACKHAT,element);
                    tmp2 =gray-tmp1;
                    if(i>n-3)
                    test(tmp2,3);
                    tmp = tmp2;
                    k=9;
                }
                namedWindow("差",WINDOW_NORMAL);
                imshow("差",tmp);
                erzhi(tmp,2,128);
                namedWindow("差",WINDOW_NORMAL);
                imshow("差",tmp);
                Mat sumblack=Mat::zeros(tmp.rows,tmp.cols,CV_8UC1);
                Mat element1 = getStructuringElement(MORPH_RECT,Size(3,3));
                Mat out,out1;
                sumblack=tmp;
                dilate(sumblack,out,element1);
                drawrow(out);
                for(int j=0;j<5;j++)
                    {
                        erode(out,out1,element1);
                        out=out1+sumblack;
                    }

                threshold(out,out,128,255,CV_THRESH_BINARY);
                namedWindow("低对比图—二值",WINDOW_NORMAL);
                imshow("低对比图—二值",out);
                imwrite("处理结果.jpg",out);
        }
        else
        {
            int k=25,n=5;
            for(int i=0;i<n;i++)
                {
                    Mat element = getStructuringElement(MORPH_RECT,Size(k,k));
                    morphologyEx(tmp,tmp1,MORPH_BLACKHAT,element);
                    tmp2 =gray-tmp1;
                    tmp = tmp2;
                    k=7;
                }
                tmp+=(char)(m/128)*tmp;
                on(tmp);
                namedWindow("差",WINDOW_NORMAL);
                imshow("差",tmp);
                Mat out;
                threshold(tmp,out,128,255,CV_THRESH_BINARY);
                namedWindow("高灰度-二值",WINDOW_NORMAL);
                imshow("高灰度-二值",out);
                imwrite("处理结果.jpg",out);
        }
    }
    waitKey();
    return 0;
}
