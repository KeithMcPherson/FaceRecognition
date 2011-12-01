//This file contains all the image manipulations necessary for this program

#include <iostream>
#include <vector>
#include <string>
#include "opencv\cv.h"
#include "opencv\cvaux.h"
#include "opencv\highgui.h"
#include "Settings.h"

using namespace cv;


IplImage* cropImage(const IplImage *img, const CvRect region);
IplImage* equalizeImage(IplImage* imageSrc);
IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight);
IplImage* rotateImage(IplImage* image, double angle);

// Returns a new image that is a cropped version of the original image. 
IplImage* cropImage(const IplImage *img, const CvRect region)
{
	IplImage *imageTmp;
	IplImage *imageRGB;
	CvSize size;
	size.height = img->height;
	size.width = img->width;

	if (img->depth != IPL_DEPTH_8U) {
		printf("ERROR in cropImage: Unknown image depth of %d given in cropImage() instead of 8 bits per pixel.\n", img->depth);
		exit(1);
	}

	// First create a new (color or greyscale) IPL Image and copy contents of img into it.
	imageTmp = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
	cvCopy(img, imageTmp, NULL);

	// Create a new image of the detected region
	// Set region of interest to that surrounding the face
	cvSetImageROI(imageTmp, region);
	// Copy region of interest (i.e. face) into a new iplImage (imageRGB) and return it
	size.width = region.width;
	size.height = region.height;
	imageRGB = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
	cvCopy(imageTmp, imageRGB, NULL);	// Copy just the region.

    cvReleaseImage( &imageTmp );
	return imageRGB;		
}

IplImage* equalizeImage(IplImage* imageSrc) {
	// Either convert the image to greyscale, or use the existing greyscale image.
	IplImage *imageProcessed;

	// Make sure the image is the same dimensions as the training images.
	imageSrc = resizeImage(imageSrc, FACE_WIDTH, FACE_HEIGHT);

	if (imageSrc->nChannels == 3) {
		imageProcessed = cvCreateImage( cvGetSize(imageSrc), IPL_DEPTH_8U, 1 );
		// Convert from RGB (actually it is BGR) to Greyscale.
		cvCvtColor( imageSrc, imageProcessed, CV_BGR2GRAY );
	}
	else {
		// Just use the input image, since it is already Greyscale.
		imageProcessed = imageSrc;
	}

	// Resize the image to be a consistent size, even if the aspect ratio changes.
	//imageProcessed = cvCreateImage(cvSize(imageGrey->width, imageGrey->height), IPL_DEPTH_8U, 1);
	// Make the image a fixed size.
	// CV_INTER_CUBIC or CV_INTER_LINEAR is good for enlarging, and
	// CV_INTER_AREA is good for shrinking / decimation, but bad at enlarging.
	//cvResize(imageGrey, imageProcessed, CV_INTER_LINEAR);

	// Give the image a standard brightness and contrast.
	cvEqualizeHist(imageProcessed, imageProcessed);

	return imageProcessed;
}

// Creates a new image copy that is of a desired size.
// Remember to free the new image later.
IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight)
{
	IplImage *outImg = 0;
	int origWidth;
	int origHeight;
	if (origImg) {
		origWidth = origImg->width;
		origHeight = origImg->height;
	}
	if (newWidth <= 0 || newHeight <= 0 || origImg == 0 || origWidth <= 0 || origHeight <= 0) {
		printf("ERROR in resizeImage: Bad desired image size of %dx%d\n.", newWidth, newHeight);
		exit(1);
	}

	// Scale the image to the new dimensions, even if the aspect ratio will be changed.
	outImg = cvCreateImage(cvSize(newWidth, newHeight), origImg->depth, origImg->nChannels);
	if (newWidth > origImg->width && newHeight > origImg->height) {
		// Make the image larger
		cvResetImageROI((IplImage*)origImg);
		cvResize(origImg, outImg, CV_INTER_LINEAR);	// CV_INTER_CUBIC or CV_INTER_LINEAR is good for enlarging
	}
	else {
		// Make the image smaller
		cvResetImageROI((IplImage*)origImg);
		cvResize(origImg, outImg, CV_INTER_AREA);	// CV_INTER_AREA is good for shrinking / decimation, but bad at enlarging.
	}

	return outImg;
}

/*IplImage* rotateImage(IplImage* sourceImg, double angle)
{
	Mat source(sourceImg);
	Point2f src_center(source.cols/2.0, source.rows/2.0);
	Mat rot_mat = getRotationMatrix2D(src_center, 40.0, 1.0);
	Mat dst;
	warpAffine(source, dst, rot_mat, source.size());
	IplImage dst_img = dst;
    return &dst_img;
	return sourceImg;
}*/

IplImage* rotateImage(IplImage* image, double angle) {

	IplImage *rotatedImage = cvCreateImage(cvSize(480,320), IPL_DEPTH_8U,image->nChannels);

	CvPoint2D32f center;
	center.x = 160;center.y = 160;
	CvMat *mapMatrix = cvCreateMat( 2, 3, CV_32FC1 );

	cv2DRotationMatrix(center, angle, 1.0, mapMatrix);
	cvWarpAffine(image, rotatedImage, mapMatrix, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));

	cvReleaseImage(&image);
	cvReleaseMat(&mapMatrix);

	return rotatedImage;
}