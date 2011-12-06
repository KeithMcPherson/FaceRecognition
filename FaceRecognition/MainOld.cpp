#include <iostream>
#include <vector>
#include <string>
#include "opencv\cv.h"
#include "opencv\cvaux.h"
#include "opencv\highgui.h"
#include "ImgMods.h"
#include "Person.h"
#include <math.h>

#ifdef USE_OLD_MAIN



using namespace std;

// Global variables
vector<IplImage*> faceImgArr;
vector<Person> people; //Array of the different people objects
vector<Person*> faceToPerson; //Same size as faceImgArr, used to match up the faceImgArr index to a person
vector<IplImage*> eigenVectArr; //Array of the different eigenvalue images
IplImage * pAvgTrainImg       = 0; // the average image
CvMat * eigenValMat           = 0; // eigenvalues
CvMat * projectedTrainFaceMat = 0; // projected training faces

CvCapture* camera = 0;	// The camera device

//all the Haar cascades we'll search for
CvHaarClassifierCascade* faceCascade;
CvHaarClassifierCascade* noseCascade;
CvHaarClassifierCascade* mouthCascade;
CvHaarClassifierCascade* leftEyeCascade;
CvHaarClassifierCascade* rightEyeCascade;

//function prototypes
IplImage* getCameraFrame();
CvRect detectHaarClassifier(const IplImage *inputImg, const CvHaarClassifierCascade* cascade );
void loadHaarClassifiers();
void learn(char *szFileTrain);
void loadFaceImgArray(char * filename);
void doPCA();
int findNearestNeighbor(float * projectedTestFace, float *pConfidence, CvRect noseRect, CvRect mouthRect, CvRect leftEyeRect, CvRect rightEyeRect);
void loadFeatures();

int main(){
	CvMat * trainPersonNumMat = 0;  // the person numbers during training
	float * projectedTestFace = 0;

	//load all the necessary classifiers
	loadHaarClassifiers();

	//Learn everything from offline files
	learn("train.txt");

	// Project the test images onto the PCA subspace
	projectedTestFace = (float *)cvAlloc( eigenVectArr.size()*sizeof(float) );

	//Rectangles to draw around the features
	CvRect faceRect, noseRect, mouthRect, leftEyeRect, rightEyeRect;

	cvNamedWindow( "Camera Feed", CV_WINDOW_AUTOSIZE );

	//Images to look for objects in
	IplImage *frame = 0; //the camera frame
	IplImage *faceImage = 0; // the face area
	IplImage *faceTopLeftImage = 0; //top left face quadrant
	IplImage *faceTopRightImage = 0; //top right face quadrant
	IplImage *faceBottomImage = 0; //bottom half of the face
	IplImage *processedFaceImage = 0;
	while(true) {
		int iNearest, nearest, truth;
		float confidence;
		//grab the camera frame
		frame = getCameraFrame();
		if( !frame ) //break if it didn't get the frame
			break;

		faceRect = detectHaarClassifier(frame, faceCascade); //detect the face
		if(faceRect.width>0) { //make sure you found a face

			//break the face into multiple parts 
			faceImage = cropImage(frame, faceRect); //the face as a whole

			//draw a rectangle around the face
			cvRectangle(frame, cvPoint(faceRect.x, faceRect.y), cvPoint(faceRect.x + faceRect.width-1, faceRect.y + faceRect.height-1), CV_RGB(0,255,0), 1, 8, 0);

			faceTopLeftImage = cropImage(faceImage, cvRect(0, 0, faceRect.width/2, faceRect.height/2)); //top left quadrant
			faceTopRightImage = cropImage(faceImage, cvRect(faceRect.width/2, 0, faceRect.width/2, faceRect.height/2)); // top right quadrant
			faceBottomImage = cropImage(faceImage, cvRect(0, faceRect.height/2, faceRect.width, faceRect.height/2)); //bottom half
			
			//find the nose as part of the whole face
			noseRect = detectHaarClassifier(faceImage, noseCascade);
			if(noseRect.width>0)
				cvRectangle(frame, cvPoint(noseRect.x + faceRect.x, noseRect.y + faceRect.y), cvPoint(noseRect.x + noseRect.width-1 + faceRect.x, noseRect.y + noseRect.height-1 + faceRect.y), CV_RGB(0,255,255), 1, 8, 0);

			//find the mouth as part of the bottom half
			mouthRect = detectHaarClassifier(faceBottomImage, mouthCascade);
			if(mouthRect.width>0)
				cvRectangle(frame, cvPoint(mouthRect.x + faceRect.x, mouthRect.y + faceRect.y + faceRect.height/2), cvPoint(mouthRect.x + mouthRect.width-1 + faceRect.x, mouthRect.y + mouthRect.height-1 + faceRect.y + faceRect.height/2), CV_RGB(255,0,0), 1, 8, 0);

			//find the left eye as part of the top left quadrant
			leftEyeRect = detectHaarClassifier(faceTopLeftImage, leftEyeCascade);
			if(leftEyeRect.width>0)
				cvRectangle(frame, cvPoint(leftEyeRect.x + faceRect.x, leftEyeRect.y + faceRect.y), cvPoint(leftEyeRect.x + leftEyeRect.width-1 + faceRect.x, leftEyeRect.y + leftEyeRect.height-1 + faceRect.y), CV_RGB(0,0,255), 1, 8, 0);

			//find the right eye as part of the top right quadrant
			rightEyeRect = detectHaarClassifier(faceTopRightImage, rightEyeCascade);
			if(rightEyeRect.width>0)
				cvRectangle(frame, cvPoint(rightEyeRect.x + faceRect.x + faceRect.width/2, rightEyeRect.y + faceRect.y), cvPoint(rightEyeRect.x + rightEyeRect.width-1 + faceRect.x + faceRect.width/2, rightEyeRect.y + rightEyeRect.height-1 + faceRect.y), CV_RGB(255,255,0), 1, 8, 0);
				
		
		} 

		//equalize the face
		if(faceImage)
			processedFaceImage = equalizeImage(faceImage);
		
		// If the face rec database has been loaded, then try to recognize the person currently detected.
		if(processedFaceImage)
			if (eigenVectArr.size() > 0) {
				// project the test image onto the PCA subspace
				cvEigenDecomposite(
					processedFaceImage,
					eigenVectArr.size(),
					eigenVectArr.data(),
					0, 0,
					pAvgTrainImg,
					projectedTestFace);
				
				// Check which person it is most likely to be.
				iNearest = findNearestNeighbor(projectedTestFace, &confidence, noseRect, mouthRect, leftEyeRect, rightEyeRect);
				//printf("%d\n", trainPersonNumMat->data.i);
				//nearest  = trainPersonNumMat->data.i[iNearest];
				printf("Most likely person in camera: '%s' (confidence=%f.\n", faceToPerson.at(iNearest)->getName().c_str(), confidence);
				//printf("test2\n\n");

			}//endif nEigens

		//draw everything
		if(frame)
			cvShowImage("Camera Feed", frame);
		if(processedFaceImage)
			cvShowImage("Face", processedFaceImage);

		string *p;

		//Cleanup after yourself
		if (faceImage)
			cvReleaseImage(&faceImage);
		if (faceTopLeftImage)
			cvReleaseImage(&faceTopLeftImage);
		if (faceTopRightImage)
			cvReleaseImage(&faceTopRightImage);
		if (faceBottomImage)
			cvReleaseImage(&faceBottomImage);
		if (processedFaceImage)
			cvReleaseImage(&processedFaceImage);

		//ESC to exit
		char c = cvWaitKey(33);
		if( c == 27 ) 
			break;
	}
	cvReleaseCapture( &camera );
	cvDestroyWindow( "Camera Feed" );
	return 0;
}

// Grab the next camera frame. Waits until the next frame is ready,
// and provides direct access to it, so do NOT modify the returned image or free it!
// Will automatically initialize the camera on the first frame.
IplImage* getCameraFrame(void)
{
	IplImage *frame;

	// If the camera hasn't been initialized, then open it.
	if (!camera) {
		printf("Acessing the camera ...\n");
		camera = cvCaptureFromCAM( 0 );
		if (!camera) {
			printf("ERROR in getCameraFrame(): Couldn't access the camera.\n");
			exit(1);
		}
		// Try to set the camera resolution
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_WIDTH, 640 );
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_HEIGHT, 480 );
		// Wait a little, so that the camera can auto-adjust itself
		Sleep(1000);
		frame = cvQueryFrame( camera );	// get the first frame, to make sure the camera is initialized.
		if (frame) {
			printf("Got a camera using a resolution of %dx%d.\n", (int)cvGetCaptureProperty( camera, CV_CAP_PROP_FRAME_WIDTH), (int)cvGetCaptureProperty( camera, CV_CAP_PROP_FRAME_HEIGHT) );
		}
	}

	frame = cvQueryFrame( camera );
	if (!frame) {
		fprintf(stderr, "ERROR in recognizeFromCam(): Could not access the camera or video file.\n");
		exit(1);
		//return NULL;
	}
	return frame;
}

//Detects all the objects we're looking for.  Specify the image and the haar cascade to search for, then get a rectangle for that area
CvRect detectHaarClassifier(const IplImage *inputImg, const CvHaarClassifierCascade* cascade )
{
	const CvSize minFeatureSize = cvSize(20, 20);
	const int flags = CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_DO_CANNY_PRUNING;	// Only search for 1 face.
	const float search_scale_factor = 1.1f;
	IplImage *detectImg;
	IplImage *greyImg = 0;
	CvMemStorage* storage;
	CvRect mainObjectRectangle;
	double time;
	CvSeq* detectedObjectRectangles;
	int i;

	storage = cvCreateMemStorage(0);
	cvClearMemStorage( storage );

	// If the image is color, use a greyscale copy of the image.
	detectImg = (IplImage*)inputImg;	// Assume the input image is to be used.
	if (inputImg->nChannels > 1) 
	{
		greyImg = cvCreateImage(cvSize(inputImg->width, inputImg->height), IPL_DEPTH_8U, 1 );
		cvCvtColor( inputImg, greyImg, CV_BGR2GRAY );
		detectImg = greyImg;	// Use the greyscale version as the input.
	}

	// Detect all the faces.
	time = (double)cvGetTickCount();
	detectedObjectRectangles = cvHaarDetectObjects( detectImg, (CvHaarClassifierCascade*)cascade, storage,
				search_scale_factor, 3, flags, minFeatureSize );
	time = (double)cvGetTickCount() - time;
	//printf("[Object Detection took %d ms and found %d objects]\n", cvRound( time/((double)cvGetTickFrequency()*1000.0) ), detectedObjectRectangles->total );

	// Get the first detected face (the biggest).
	if (detectedObjectRectangles->total > 0) {
        mainObjectRectangle = *(CvRect*)cvGetSeqElem( detectedObjectRectangles, 0 );
    }
	else
		mainObjectRectangle = cvRect(-1,-1,-1,-1);	// Couldn't find the face.

	//cvReleaseHaarClassifierCascade( &cascade );
	//cvReleaseImage( &detectImg );
	if (greyImg)
		cvReleaseImage( &greyImg );
	cvReleaseMemStorage( &storage );

	return mainObjectRectangle;	// Return the biggest face found, or (-1,-1,-1,-1).
}

//Loads all the necessary classifier files for use elsewhere
void loadHaarClassifiers(){
	// Haar Cascade file, used for Face Detection.
	const char *faceCascadeFilename = "cascades\\haarcascade_frontalface_alt.xml";
	const char *noseCascadeFilename = "cascades\\haarcascade_mcs_nose.xml";
	const char *mouthCascadeFilename = "cascades\\haarcascade_mcs_mouth.xml";
	const char *leftEyeCascadeFilename = "cascades\\haarcascade_mcs_lefteye.xml";
	const char *rightEyeCascadeFilename = "cascades\\haarcascade_mcs_righteye.xml";

	//Load the HaarCascade classifier for faces
	faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0 );
	if( !faceCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", faceCascadeFilename);
		exit(1);
	}

	//Load the HaarCascade classifier for noses
	noseCascade = (CvHaarClassifierCascade*)cvLoad(noseCascadeFilename, 0, 0, 0 );
	if( !noseCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", noseCascadeFilename);
		exit(1);
	}

	//Load the HaarCascade classifier for mouths
	mouthCascade = (CvHaarClassifierCascade*)cvLoad(mouthCascadeFilename, 0, 0, 0 );
	if( !mouthCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", mouthCascadeFilename);
		exit(1);
	}

	//Load the HaarCascade classifier for left eyes
	leftEyeCascade = (CvHaarClassifierCascade*)cvLoad(leftEyeCascadeFilename, 0, 0, 0 );
	if( !leftEyeCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", leftEyeCascadeFilename);
		exit(1);
	}

	//Load the HaarCascade classifier for right eyes
	rightEyeCascade = (CvHaarClassifierCascade*)cvLoad(rightEyeCascadeFilename, 0, 0, 0 );
	if( !rightEyeCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", rightEyeCascadeFilename);
		exit(1);
	}
}

// Train from the data in the given text file, and store the trained data into the file 'facedata.xml'.
void learn(char *szFileTrain)
{
	int i, offset;

	// load training data
	printf("Loading the training images in '%s'\n", szFileTrain);
	/*nTrainFaces = */loadFaceImgArray(szFileTrain);
	printf("Got %d training images.\n", faceImgArr.size());
	if( faceImgArr.size() < 2 )
	{
		fprintf(stderr,
		        "Need 2 or more training faces\n"
		        "Input file contains only %d\n", faceImgArr.size());
		return;
	}
	
	// do PCA on the training faces
	doPCA();
	loadFeatures();

	// project the training images onto the PCA subspace
	projectedTrainFaceMat = cvCreateMat( faceImgArr.size(), eigenVectArr.size(), CV_32FC1 );
	offset = projectedTrainFaceMat->step / sizeof(float);
	for(i=0; i<faceImgArr.size(); i++)
	{
		//int offset = i * nEigens;
		cvEigenDecomposite(
			faceImgArr[i],
			eigenVectArr.size(),
			eigenVectArr.data(),
			0, 0,
			pAvgTrainImg,
			//projectedTrainFaceMat->data.fl + i*nEigens);
			projectedTrainFaceMat->data.fl + i*offset);
	}

}

// Read the names & image filenames of people from a text file, and load all those images listed.
void loadFaceImgArray(char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int nFiles=0;

	// open the input file
	if( !(imgListFile = fopen(filename, "r")) )
	{
		fprintf(stderr, "Can\'t open file %s\n", filename);
		return;
	}

	// count the number of faces
	while( fgets(imgFilename, 512, imgListFile) ) ++nFiles;
	rewind(imgListFile);

	// store the face images in an array
	for(int iFace=0; iFace<nFiles; iFace++)
	{
		char personName[256];
		string sPersonName;

		// read person number (beginning with 1), their name and the image filename.
		fscanf(imgListFile, "%s %s", personName, imgFilename);
		sPersonName = personName;

		IplImage *originalImage = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);
		CvRect faceRect = cvRect(-1,-1,-1,-1);

		//Check if this person already exists
		int personInset;
		bool nameExists = false;
		for (unsigned int i=0; i<people.size();i++){
			if(people.at(i).getName().compare(sPersonName) == 0) {
				nameExists = true;
				personInset = i;
			}
		}
		//if not, add them
		if (!(nameExists)){
			Person *newPerson = new Person(sPersonName);
			people.push_back(*newPerson);
			personInset=people.size()-1;
		}

		if(ROTATE_INPUT_IMAGES) {
			originalImage = rotateImage(originalImage, -ROTATE_MAX);
			for (int i=0;i<=20;i++) {
				if(originalImage)
					faceRect = detectHaarClassifier(originalImage, faceCascade);
				else
					faceRect = cvRect(-1,-1,-1,-1);
				
				if(faceRect.width>0) { //make sure you found a face
					IplImage* tempImage = &IplImage(*originalImage); //crop just the face as a whole and equalize it
					tempImage = cropImage(tempImage, faceRect);
					people.at(personInset).addFace(tempImage);
					//cvReleaseImage(&tempImage);
				} else {
					//fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
					//nFiles--;
					continue;
				}
				
				originalImage = rotateImage(originalImage, ROTATE_STEP);
				faceRect = cvRect(-1,-1,-1,-1);
			}
			cvFlip(originalImage, NULL, 1);
			originalImage = rotateImage(originalImage, -ROTATE_MAX*2);
			for (int i=0;i<=20;i++) {
				if(originalImage)
					faceRect = detectHaarClassifier(originalImage, faceCascade);
				else
					faceRect = cvRect(-1,-1,-1,-1);
				
				if(faceRect.width>0) { //make sure you found a face
					IplImage* tempImage = &IplImage(*originalImage); //crop just the face as a whole and equalize it
					tempImage = cropImage(tempImage, faceRect);
					people.at(personInset).addFace(tempImage);
					//cvReleaseImage(&tempImage);
				} else {
					//fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
					//nFiles--;
					continue;
				}
				originalImage = rotateImage(originalImage, ROTATE_STEP);
				faceRect = cvRect(-1,-1,-1,-1);
			}
			
			//cvReleaseImage(&originalImage);
		} else {
			if(originalImage)
				faceRect = detectHaarClassifier(originalImage, faceCascade);
			else
				faceRect = cvRect(-1,-1,-1,-1);
			if(faceRect.width>0) { //make sure you found a face
				//people.at(personInset).addFace(equalizeImage(cropImage(originalImage, faceRect)));
				people.at(personInset).addFace(cropImage(originalImage, faceRect));
				cvFlip(originalImage, NULL, 1);
				//people.at(personInset).addFace(equalizeImage(cropImage(originalImage, faceRect)));
				people.at(personInset).addFace(cropImage(originalImage, faceRect));
				//cvReleaseImage(&originalImage);
			} else {
				//fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
				//nFiles--;
			continue;
			}
	}
	}
	fclose(imgListFile);

	//runs through all the different people and adds their images to the global array 
	//and their references so they can be found later
	for (int iPeople=0; iPeople<people.size(); iPeople++) {
		for (int iFace=0; iFace<people.at(iPeople).getFaceImgs().size(); iFace++) {
			faceToPerson.push_back(&people.at(iPeople));
			faceImgArr.push_back(equalizeImage(people.at(iPeople).getFaceImgs().at(iFace)));
		}
	}


	printf("Data loaded from '%s': (%d images of %d people).\n", filename, faceImgArr.size(), people.size());
	printf("People: ");
	if (people.size() > 0)
		printf("%s(%d images)", people.at(0).getName().c_str(), people.at(0).getFaceImgs().size());
	if (people.size() > 1)
		for (unsigned int i=1; i<people.size(); i++) {
			printf(", %s(%d images)", people.at(i).getName().c_str(), people.at(i).getFaceImgs().size());
		}
	printf(".\n");
}

// Do the Principal Component Analysis, finding the average image
// and the eigenfaces that represent any image in the given dataset.
void doPCA()
{
	int i;
	CvTermCriteria calcLimit;
	CvSize faceImgSize;

	// allocate the eigenvector images
	faceImgSize.width  = FACE_WIDTH;
	faceImgSize.height = FACE_HEIGHT;
	
	for(i=0; i<faceImgArr.size()-1; i++) {
		eigenVectArr.push_back(cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1));
	}
	// allocate the eigenvalue array
	eigenValMat = cvCreateMat( 1, eigenVectArr.size(), CV_32FC1 );
	
	// allocate the averaged image
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);
	
	// set the PCA termination criterion
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, eigenVectArr.size(), 1);
	// compute average image, eigenvalues, and eigenvectors
	cvCalcEigenObjects(
		faceImgArr.size(),
		(void*)faceImgArr.data(),
		(void*)eigenVectArr.data(),
		CV_EIGOBJ_NO_CALLBACK,
		0,
		0,
		&calcLimit,
		pAvgTrainImg,
		eigenValMat->data.fl);
	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}

// Find the most likely person based on a detection. Returns the index, and stores the confidence value into pConfidence.
int findNearestNeighbor(float * projectedTestFace, float *pConfidence, CvRect noseRect, CvRect mouthRect, CvRect leftEyeRect, CvRect rightEyeRect)
{
	int noseToMouth = 0;
	int eyeToEye = 0;
	if (noseRect.width > 0 && mouthRect.width >0) {
		
		int noseCenterX = noseRect.x + noseRect.width/2;
		int noseCenterY = noseRect.y + noseRect.height/2;
		int mouthCenterX = mouthRect.x + mouthRect.width/2;
		int mouthCenterY = mouthRect.y + mouthRect.height/2;
		
		noseToMouth = (noseCenterX - mouthCenterX)*(noseCenterX - mouthCenterX) + (noseCenterY - mouthCenterY)*(noseCenterY - mouthCenterY);
	}

	if (leftEyeRect.width > 0 && rightEyeRect.width >0) {
		int leftEyeCenterX = leftEyeRect.x + leftEyeRect.width/2.0;
		int leftEyeCenterY = leftEyeRect.y + leftEyeRect.height/2.0;
		int rightEyeCenterX = rightEyeRect.x + rightEyeRect.width/2.0;
		int rightEyeCenterY = rightEyeRect.y + rightEyeRect.height/2.0;
		
		eyeToEye = (leftEyeCenterX - rightEyeCenterX)*(leftEyeCenterX - rightEyeCenterX) + (leftEyeCenterY - rightEyeCenterY)*(leftEyeCenterY - rightEyeCenterY) ;
				
	}
	int maxEyeToEyeDiff = 99999;
	int maxNoseToMouthDiff = 99999;
	int eyeToEyeNearest = 0;
	int noseToMouthNearest = 0;
	for (int i=0; i<people.size(); i++) {
		int eyeToEyeDiff = abs(eyeToEye - people.at(i).eyeToEye);
		if (eyeToEyeDiff < maxEyeToEyeDiff) {
			maxEyeToEyeDiff = eyeToEyeDiff;
			eyeToEyeNearest = i;
		}

		int noseToMouthDiff = abs(noseToMouth - people.at(i).noseToMouth);
		if (noseToMouthDiff < maxEyeToEyeDiff) {
		maxEyeToEyeDiff = noseToMouthDiff;
		noseToMouthNearest = i;
		}
	}

	printf("Nearest eye to eye is: %s\n", people.at(eyeToEyeNearest).getName().c_str());
	printf("Nearest nose to mouth is: %s\n", people.at(noseToMouthNearest).getName().c_str());

	//double leastDistSq = 1e12;
	double leastDistSq = DBL_MAX;
	int i, iTrain, iNearest = 0;

	for(iTrain=0; iTrain<faceImgArr.size(); iTrain++)
	{
		double distSq=0;

		for(i=0; i<eigenVectArr.size(); i++)
		{
			float d_i = projectedTestFace[i] - projectedTrainFaceMat->data.fl[iTrain*eigenVectArr.size() + i];

			distSq += d_i*d_i / eigenValMat->data.fl[i];  // Mahalanobis distance (might give better results than Eucalidean distance)
			//distSq += d_i*d_i; // Euclidean distance.
		}

		if(distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}

	// Return the confidence level based on the Euclidean distance,
	// so that similar images should give a confidence between 0.5 to 1.0,
	// and very different images should give a confidence between 0.0 to 0.5.
	//*pConfidence = 1.0f - sqrt( leastDistSq / (float)(faceImgArr.size() * eigenVectArr.size()) ) / 255.0f;
	
	//Instead, just provide the leastDistSq, since the above doesn't actually work
	*pConfidence = sqrt(leastDistSq);

	// Return the found index.
	return iNearest;
}

void loadFeatures() {
	
	printf("Loading Features...\n");

	for (int i=0; i<people.size(); i++) {
		int eyeToEye = 0;
		int numEyeToEye = 0;
		float noseToMouth = 0;
		int numNoseToMouth = 0;
		for (int j=0; j<people.at(i).getFaceImgs().size(); j++) {

			IplImage* faceImage = people.at(i).getFaceImgs().at(j);

			CvRect noseRect, mouthRect, leftEyeRect, rightEyeRect;

			IplImage *faceTopLeftImage = 0; //top left face quadrant
			IplImage *faceTopRightImage = 0; //top right face quadrant
			IplImage *faceBottomImage = 0; //bottom half of the face

			//break the face into multiple parts 
			faceTopLeftImage = cropImage(faceImage, cvRect(0, 0, faceImage->width/2, faceImage->height/2)); //top left quadrant
			faceTopRightImage = cropImage(faceImage, cvRect(faceImage->width/2, 0, faceImage->width/2, faceImage->height/2)); // top right quadrant
			faceBottomImage = cropImage(faceImage, cvRect(0, faceImage->height/2, faceImage->width, faceImage->height/2)); //bottom half
			
			//find the nose as part of the whole face
			noseRect = detectHaarClassifier(faceImage, noseCascade);
			//find the mouth as part of the bottom half
			mouthRect = detectHaarClassifier(faceBottomImage, mouthCascade);
			//find the left eye as part of the top left quadrant
			leftEyeRect = detectHaarClassifier(faceTopLeftImage, leftEyeCascade);
			//find the right eye as part of the top right quadrant
			rightEyeRect = detectHaarClassifier(faceTopRightImage, rightEyeCascade);
			
			if (noseRect.width > 0 && mouthRect.width >0) {
				
				int noseCenterX = noseRect.x + noseRect.width/2;
				int noseCenterY = noseRect.y + noseRect.height/2;
				int mouthCenterX = mouthRect.x + mouthRect.width/2;
				int mouthCenterY = mouthRect.y + mouthRect.height/2;
				
				numNoseToMouth++;
				noseToMouth += (noseCenterX - mouthCenterX)*(noseCenterX - mouthCenterX) + (noseCenterY - mouthCenterY)*(noseCenterY - mouthCenterY);
			}

			if (leftEyeRect.width > 0 && rightEyeRect.width >0) {
				int leftEyeCenterX = leftEyeRect.x + leftEyeRect.width/2.0;
				int leftEyeCenterY = leftEyeRect.y + leftEyeRect.height/2.0;
				int rightEyeCenterX = rightEyeRect.x + rightEyeRect.width/2.0;
				int rightEyeCenterY = rightEyeRect.y + rightEyeRect.height/2.0;
				
				numEyeToEye++;
				eyeToEye += (leftEyeCenterX - rightEyeCenterX)*(leftEyeCenterX - rightEyeCenterX) + (leftEyeCenterY - rightEyeCenterY)*(leftEyeCenterY - rightEyeCenterY) ;
				
			}

			//if (faceImage)
				//cvReleaseImage(&faceImage);
			if (faceTopLeftImage)
				cvReleaseImage(&faceTopLeftImage);
			if (faceTopRightImage)
				cvReleaseImage(&faceTopRightImage);
			if (faceBottomImage)
				cvReleaseImage(&faceBottomImage);

		}
		int eyeToEyeAvg = 0;
		int noseToMouthAvg = 0;
		if (numEyeToEye != 0)
			eyeToEyeAvg = eyeToEye / numEyeToEye;
		if (numNoseToMouth != 0)
			int noseToMouthAvg = noseToMouth / numNoseToMouth;
		people.at(i).eyeToEye = eyeToEyeAvg;
		people.at(i).noseToMouth = noseToMouthAvg;
		printf("%s eyeToEyeAvg: %d\n", people.at(i).getName().c_str(), eyeToEyeAvg);
	}
}



#endif