//
//  DlibWrapper.m
//  DisplayLiveSamples
//
//  Created by Luis Reisewitz on 16.05.16.
//  Copyright © 2016 ZweiGraf. All rights reserved.
//

#import "DlibWrapper.h"
#import <UIKit/UIKit.h>

#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <math.h>


struct point{
    double x;
    double y;
};

struct roots{
    long double root1;
    long double root2;
};

@interface DlibWrapper ()

@property (assign) BOOL prepared;

+ (std::vector<dlib::rectangle>)convertCGRectValueArray:(NSArray<NSValue *> *)rects;
int smallestint (int*,int);
long double smallestdouble (long double*,int);
roots rootsabc(long double , long double , long double );

@end
@implementation DlibWrapper {
    dlib::shape_predictor sp;
}


- (instancetype)init {
    self = [super init];
    if (self) {
        _prepared = NO;
    }
    return self;
}

- (void)prepare {
    NSString *modelFileName = [[NSBundle mainBundle] pathForResource:@"shape_predictor_68_face_landmarks" ofType:@"dat"];
    std::string modelFileNameCString = [modelFileName UTF8String];
    
    dlib::deserialize(modelFileNameCString) >> sp;
    
    // FIXME: test this stuff for memory leaks (cpp object destruction)
    self.prepared = YES;
}

- (void)doWorkOnSampleBuffer:(CMSampleBufferRef)sampleBuffer inRects:(NSArray<NSValue *> *)rects {
    
    if (!self.prepared) {
        [self prepare];
    }
    
    // Declare local variables
    
    dlib::array2d<dlib::bgr_pixel> img;
    
    int partno, x_canthus_eyeleft,y_canthus_eyeleft,x_canthus_eyeright,y_canthus_eyeright;
    int fh1,fh2,fh3,fh4;
    int x_noselist[4],y_noselist[4],flag_noselist_same;
    long double m_canthus,c_canthus,m_noselist, c_noselist;
    int xfh1,xfh2;
    long double xfh1_1,xfh1_2,xfh2_1,xfh2_2;
    int yfh1,yfh2;
    long double yfh1_uncorr,yfh2_uncorr;
    long double yfh1_1,yfh1_2,yfh2_1,yfh2_2;
    int xinter,yinter;
    long double xinter_uncorr,yinter_uncorr;
    int xfh_top_left,yfh_top_left,xfh_top_right,yfh_top_right,xfh_bottom_left,yfh_bottom_left,xfh_bottom_right,yfh_bottom_right;
    long double distcan;
    long double afh1,bfh1,cfh1;
    long double afh2,bfh2,cfh2;
    long double afh_top,bfh_top,cfh_top;
    long double afh_bottom,bfh_bottom,cfh_bottom;
    double m_fh_top,c_fh_top,m_fh_bottom,c_fh_bottom;
    
    double xmean,ymean;
    
    int pixelcount = 0,pixelhuesum,pixelredsum,pixelgreensum,pixelhue,pixelred;
    int rval,gval,bval,hval,sval,vval;
    
    double avggreenarray[300],avggreen;
    
    // MARK: magic
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    char *baseBuffer = (char *)CVPixelBufferGetBaseAddress(imageBuffer);
    
    // set_size expects rows, cols format
    img.set_size(height, width);
    
    // copy samplebuffer image data into dlib image format
    img.reset();
    long position = 0;
    while (img.move_next()) {
        dlib::bgr_pixel& pixel = img.element();

        // assuming bgra format here
        long bufferLocation = position * 4; //(row * width + column) * 4;
        char b = baseBuffer[bufferLocation];
        char g = baseBuffer[bufferLocation + 1];
        char r = baseBuffer[bufferLocation + 2];
        //        we do not need this
        //        char a = baseBuffer[bufferLocation + 3];
        
        dlib::bgr_pixel newpixel(b, g, r);
        pixel = newpixel;
        
        position++;
    }
    
    // unlock buffer again until we need it again
    CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    
    // convert the face bounds list to dlib format
    std::vector<dlib::rectangle> convertedRectangles = [DlibWrapper convertCGRectValueArray:rects];
    
    // for every detected face
    for (unsigned long j = 0; j < convertedRectangles.size(); ++j)
    {
        dlib::rectangle oneFaceRect = convertedRectangles[j];
        
        // detect all landmarks
        dlib::full_object_detection shape = sp(img, oneFaceRect);
        
        // and draw them into the image (samplebuffer)
        for (unsigned long k = 0; k < shape.num_parts(); k++) {
            dlib::point p = shape.part(k);
            // draw_solid_circle(img, p, 3, dlib::rgb_pixel(0, 255, 255));
        }

        draw_solid_circle(img, shape.part(27), 3, dlib::rgb_pixel(0, 255, 255));
        draw_solid_circle(img, shape.part(28), 3, dlib::rgb_pixel(0, 255, 255));
        draw_solid_circle(img, shape.part(29), 3, dlib::rgb_pixel(0, 255, 255));
        draw_solid_circle(img, shape.part(30), 3, dlib::rgb_pixel(0, 255, 255));

        // detect forehead
        
        for (int k=27;k<31;k++)
        {
            x_noselist[k-27] = shape.part(k)(0);
            y_noselist[k-27] = shape.part(k)(1);
            // cout << "Nose point "<<k-27<< " is " << x_noselist[k-27] << "," << y_noselist[k-27] << endl;
            // cout << shape.part(k) << "\t";
        }
        
        
        
        // compute best fit line for nose
        if ((x_noselist[0] == x_noselist[1]) && (x_noselist[1] == x_noselist[2]) && (x_noselist[2] == x_noselist[3]))
        {
            flag_noselist_same = 1;
        }
        else
        {
            flag_noselist_same = 0;
        }
        
        // cout << "\nNose calculation ended. Flag is " <<flag_noselist_same<<endl;
        //flag_noselist_same = 1;  // delete
        
        if (flag_noselist_same == 0)
        {
            xmean = (double(x_noselist[0] + x_noselist[1] + x_noselist[2] + x_noselist[3]))/4;
            ymean = (double(y_noselist[0] + y_noselist[1] + y_noselist[2] + y_noselist[3]))/4;
            
            long double numer1 = 0, denom1 = 0;
            
            for (int k=0;k<4;k++)
            {
                numer1 += (double(x_noselist[k]) - xmean)*(double(y_noselist[k]) - ymean);
                denom1 += (double(x_noselist[k]) - xmean)*(double(x_noselist[k]) - xmean);
            }
            
            // cout << "xav,yav = " <<xmean << "," << ymean << endl;
            // cout << "num,den = " <<numer1 << "," << denom1 << endl;
            
            m_noselist = numer1/denom1; // delete
            c_noselist = ymean - m_noselist*xmean; // delete
            
            // cout << "M,C of nose = " << m_noselist << "," << c_noselist << endl;
            // m_noselist,c_noselist = np.polyfit(x_noselist, y_noselist, 1); //python
            // poly_eqn_noselist = np.poly1d([m_noselist,c_noselist]); //poly1d requires a single arguement //python
            // height, width, channel = imgcv.shape //python
        }
        
        // canthus line fit
        
        partno = 39; //l canthus
        x_canthus_eyeleft = shape.part(partno)(0);
        y_canthus_eyeleft = shape.part(partno)(1);
        
        partno = 42; //r canthus
        x_canthus_eyeright = shape.part(partno)(0);
        y_canthus_eyeright = shape.part(partno)(1);

        dlib::point p_canthus_l;
        p_canthus_l(0) = x_canthus_eyeleft;
        p_canthus_l(1) = y_canthus_eyeleft;
        // draw_solid_circle(img, p_canthus_l, 3, dlib::rgb_pixel(255, 0, 0));

        dlib::point p_canthus_r;
        p_canthus_r(0) = x_canthus_eyeright;
        p_canthus_r(1) = y_canthus_eyeright;
        // draw_solid_circle(img, p_canthus_r, 3, dlib::rgb_pixel(255, 0, 0));

        m_canthus = (double(y_canthus_eyeleft-y_canthus_eyeright)/double(x_canthus_eyeleft-x_canthus_eyeright));
        c_canthus = (y_canthus_eyeleft - (m_canthus*x_canthus_eyeleft));
        
        // cout << "M,C of canthus = " << m_canthus << "," << c_canthus << endl;
        
        if (flag_noselist_same == 1)
        {
            int xinter = x_noselist[0];
        }
        else
        {
            xinter = int((c_canthus-c_noselist)/(m_noselist-m_canthus));
            xinter_uncorr = (c_canthus-c_noselist)/(m_noselist-m_canthus);
            yinter = int((m_canthus*xinter)+c_canthus);
            yinter_uncorr = (m_canthus*xinter_uncorr)+c_canthus;
            
            dlib::point p_inter;
            p_inter(0) = xinter;
            p_inter(1) = yinter;
            draw_solid_circle(img, p_inter, 3, dlib::rgb_pixel(255, 255, 0));
        }
        
        // cout << "Intersection = " << xinter << "," << yinter << endl;
        // cout << "Intersection uncorrected= " << xinter_uncorr << "," << yinter_uncorr << endl;
        
        // calculate distance between canthuses
        distcan = sqrt( ((y_canthus_eyeright-y_canthus_eyeleft)*(y_canthus_eyeright-y_canthus_eyeleft)) + ((x_canthus_eyeright-x_canthus_eyeleft)*(x_canthus_eyeright-x_canthus_eyeleft)) );
        
        
        // for finding bottom of forehead, binomial coefficients
        if (flag_noselist_same == 1)
        {
            xfh1 = x_noselist[0];
            yfh1 = yinter - int(distcan);
        }
        else
        {
            afh1 = 1;
            bfh1 = -2 * xinter_uncorr;
            cfh1 = (xinter_uncorr * xinter_uncorr) - ( (distcan * distcan) / ((m_noselist * m_noselist)+1) );
            // print "afh1,bfh1,cfh1",afh1,bfh1,cfh1
            roots xfh1_roots = rootsabc(afh1,bfh1,cfh1);
            // cout << "ROOTS_xfh1 = " << xfh1_roots.root1 <<  "," << xfh1_roots.root2 << endl;
            //cout << "ROOTS RAW = " << rootval[1] << "," << rootval[0] << endl;
            xfh1_1 = xfh1_roots.root2; // bottom of forehead // python
            yfh1_1 = m_noselist*xfh1_1+c_noselist;
            xfh1_2 = xfh1_roots.root1; // bottom of forehead // python
            yfh1_2 = (m_noselist * xfh1_2) + c_noselist;
            long double temparray[] = {yfh1_1,yfh1_2};
            yfh1_uncorr = smallestdouble(temparray,2);
            yfh1 = int(yfh1_uncorr);
            xfh1 = int((yfh1_uncorr-c_noselist)/m_noselist);
        }

        // for finding top of forehead, binomial coefficients
        
        if (flag_noselist_same == 1)
        {
            xfh2 = x_noselist[0];
            yfh2 = yinter - int(1.5*distcan);
        }
        else
        {
            afh2 = 1;
            bfh2 = -2*xinter_uncorr;
            cfh2 = (xinter_uncorr*xinter_uncorr) - ( (distcan*distcan*2.25) / ((m_noselist*m_noselist)+1) );
            
            roots xfh2_roots = rootsabc(afh2,bfh2,cfh2);
            // cout << "ROOTS_xfh2 = " << xfh2_roots.root1 << "," << xfh2_roots.root2 << endl;
            xfh2_1 = xfh2_roots.root2; // bottom of forehead // python
            // xfh2_1 = min(np.roots([afh2,bfh2,cfh2])); //top of forehead // python
            yfh2_1 = m_noselist*xfh2_1+c_noselist;
            xfh2_2 = xfh2_roots.root1; // bottom of forehead // python
            // xfh2_2 = max(np.roots([afh2,bfh2,cfh2])); // top of forehead // python
            yfh2_2 = (m_noselist*xfh2_2)+c_noselist;
            long double temparray[] = {yfh2_1,yfh2_2};
            yfh2_uncorr = smallestdouble(temparray,2);
            yfh2 = int(yfh2_uncorr);
            xfh2 = int((yfh2_uncorr-c_noselist)/m_noselist) ;
        }

        dlib::point p_fh1_1;
        p_fh1_1(0) = xfh1_1;
        p_fh1_1(1) = yfh1_1;
        // draw_solid_circle(img, p_fh1_1, 3, dlib::rgb_pixel(255, 0, 0));

        dlib::point p_fh1_2;
        p_fh1_2(0) = xfh1_2;
        p_fh1_2(1) = yfh1_2;
        // draw_solid_circle(img, p_fh1_2, 3, dlib::rgb_pixel(255, 0, 0));

        
        dlib::point p_fh1;
        p_fh1(0) = xfh1;
        p_fh1(1) = yfh1;
        //draw_solid_circle(img, p_fh1, 3, dlib::rgb_pixel(255, 0, 0));
        
        dlib::point p_fh2;
        p_fh2(0) = xfh2;
        p_fh2(1) = yfh2;
        //draw_solid_circle(img, p_fh2, 3, dlib::rgb_pixel(255, 0, 0));

        
        // for finding line perpendicular to nose-to-forehead line
        
        if (flag_noselist_same == 1)
        {
            m_fh_bottom = 0;
        }
        else
        {
            m_fh_bottom = 1/m_noselist; //slope of forehead lower line // python
            m_fh_top = m_fh_bottom; // slope of forehead upper line
            c_fh_bottom = yfh1 - m_fh_bottom*xfh1; // y intercept of forehead lower line
            c_fh_top = yfh2 - m_fh_top*xfh2; // y intercept of forehead upper line
        }
        // points on forehead top left and bottom right
        if (flag_noselist_same == 1)
        {
            //top forehead
            yfh_top_left = int(yfh2);
            xfh_top_left = int(xfh2-(distcan));
            yfh_top_right = int(yfh2);
            xfh_top_right = int(xfh2+(distcan));
            
            //bottom forehead
            yfh_bottom_left = int(yfh1);
            xfh_bottom_left = int(xfh1-(distcan));
            yfh_bottom_right = int(yfh1);
            xfh_bottom_right = int(xfh1+(distcan));
        }
        else
        {
            //top forehead
            afh_top = 1;
            bfh_top = -2* xfh2;
            cfh_top = (xfh2*xfh2) - ( (distcan*distcan) / ((m_fh_top*m_fh_top)+1) );
            
            roots xfh_top_roots = rootsabc(afh_top,bfh_top,cfh_top);
            // cout << "ROOTS_fh_top = " << xfh_top_roots.root1 <<  "," << xfh_top_roots.root2 << endl;
            // xfh_top_left = int(min(np.roots([afh_top,bfh_top,cfh_top]))); // left of forehead top // python
            xfh_top_left = xfh_top_roots.root2;
            yfh_top_left = int(m_fh_top*xfh2+c_fh_top);
            // xfh_top_right = int(max(np.roots([afh_top,bfh_top,cfh_top]))); // right of forehead top // python
            xfh_top_right = xfh_top_roots.root1;
            yfh_top_right = int(m_fh_top*xfh2+c_fh_top);
            
            //bottom forehead
            afh_bottom = 1;
            bfh_bottom = -2* xfh1;
            cfh_bottom = (xfh1*xfh1) - ( (distcan*distcan) / ((m_fh_top*m_fh_top)+1) );
            
            roots xfh_bottom_roots = rootsabc(afh_bottom,bfh_bottom,cfh_bottom);
            // cout << "ROOTS_fh_bottom = " << xfh_bottom_roots.root1 << "," << xfh_bottom_roots.root2 << endl;
            // xfh_bottom_left = int(min(np.roots([afh_bottom,bfh_bottom,cfh_bottom]))); // left of forehead top // python
            xfh_bottom_left = xfh_bottom_roots.root2;
            yfh_bottom_left = int(m_fh_bottom*xfh2+c_fh_bottom);
            // xfh_bottom_right = int(max(np.roots([afh_bottom,bfh_bottom,cfh_bottom]))); // right of forehead top // python
            xfh_bottom_right = xfh_bottom_roots.root1;
            yfh_bottom_right = int(m_fh_bottom*xfh2+c_fh_bottom);
        }
        dlib::point fh_tr;
        fh_tr(0) = xfh_top_right;
        fh_tr(1) = yfh_top_right;
        draw_solid_circle(img, fh_tr, 3, dlib::rgb_pixel(0, 0, 255));

        dlib::point fh_br;
        fh_br(0) = xfh_bottom_right;
        fh_br(1) = yfh_bottom_right;
        draw_solid_circle(img, fh_br, 3, dlib::rgb_pixel(0, 0, 255));

        dlib::point fh_tl;
        fh_tl(0) = xfh_top_left;
        fh_tl(1) = yfh_top_left;
        draw_solid_circle(img, fh_tl, 3, dlib::rgb_pixel(0, 0, 255));
        
        dlib::point fh_bl;
        fh_bl(0) = xfh_bottom_left;
        fh_bl(1) = yfh_bottom_left;
        draw_solid_circle(img, fh_bl, 3, dlib::rgb_pixel(0, 0, 255));


        
        // detect hsv
    }
    
    // lets put everything back where it belongs
    CVPixelBufferLockBaseAddress(imageBuffer, 0);

    // copy dlib image data back into samplebuffer
    img.reset();
    position = 0;
    while (img.move_next()) {
        dlib::bgr_pixel& pixel = img.element();
        
        // assuming bgra format here
        long bufferLocation = position * 4; //(row * width + column) * 4;
        baseBuffer[bufferLocation] = pixel.blue;
        baseBuffer[bufferLocation + 1] = pixel.green;
        baseBuffer[bufferLocation + 2] = pixel.red;
        //        we do not need this
        //        char a = baseBuffer[bufferLocation + 3];
        
        position++;
    }
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
}

+ (std::vector<dlib::rectangle>)convertCGRectValueArray:(NSArray<NSValue *> *)rects {
    std::vector<dlib::rectangle> myConvertedRects;
    for (NSValue *rectValue in rects) {
        CGRect rect = [rectValue CGRectValue];
        long left = rect.origin.x;
        long top = rect.origin.y;
        long right = left + rect.size.width;
        long bottom = top + rect.size.height;
        dlib::rectangle dlibRect(left, top, right, bottom);

        myConvertedRects.push_back(dlibRect);
    }
    return myConvertedRects;
}

// Functions ------------------------------------------------------

int smallestint (int intarray [], int n)

{
    int i;
    int temp1 = 0;
    int temp2 = 0;
    
    for(i = 0; i < n; i ++)
    {
        
        if (intarray [i] < temp1)
        {
            intarray [i-1] = intarray [i];
            temp2 = intarray[i];
            intarray[i] = temp1;
            temp1 = temp2;
        }
        else
            temp1 = intarray[i];
    }
    
    return intarray[0];
}

long double smallestdouble (long double doublearray [], int n)

{
    int i;
    long double temp1 = 0;
    long double temp2 = 0;
    
    for(i = 0; i < n; i ++)
    {
        
        if (doublearray [i] < temp1)
        {
            doublearray [i-1] = doublearray [i];
            temp2 = doublearray[i];
            doublearray[i] = temp1;
            temp1 = temp2;
        }
        else
            temp1 = doublearray[i];
    }
    
    return doublearray[0];
}

roots rootsabc(long double a, long double b, long double c)
{
    //double rootval[2];
    roots rootval;
    
    // cout << "a,b,c = " << a << "," << b << "," << c << endl;
    
    rootval.root1 = (-b + sqrt(b * b - 4 * a * c))/(2*a);
    rootval.root2 = (-b - sqrt(b * b - 4 * a * c))/(2*a);
    
    // cout << "ROOTS RAW = " << rootval.root1 << "," << rootval.root2 << endl;
    
    return rootval;
}




@end
