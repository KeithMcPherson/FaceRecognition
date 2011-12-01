/*#include "Settings.h"

#ifndef USE_OLD_MAIN

#include "ImgMods.h"
#include "Person.h"


using namespace std;

CvCapture* camera = 0;	// The camera device
vector<Person> people;

//all the Haar cascades we'll search for
CvHaarClassifierCascade* faceCascade;
CvHaarClassifierCascade* noseCascade;
CvHaarClassifierCascade* mouthCascade;
CvHaarClassifierCascade* leftEyeCascade;
CvHaarClassifierCascade* rightEyeCascade;

//function prototypes
IplImage* getCameraFrame(); //loads the camera object if it hasn't been loaded yet, grabs the current frame
CvRect detectHaarClassifier(const IplImage *inputImg, const CvHaarClassifierCascade* cascade );
void loadHaarClassifiers(); //loads in all the Haar cascade files for use
int loadFaceImgs(char * filename);
Person* findNearestNeighbor(float * projectedTestFace, float *pConfidence);

int main(){
	IplImage *frame = 0; //the camera frame
	IplImage *faceImg = 0;
	IplImage *processedFaceImg = 0;
	CvRect faceRect;
	cvNamedWindow( "Camera Feed", CV_WINDOW_AUTOSIZE );

	//load all the necessary classifiers
	loadHaarClassifiers();

	//loads all the different people and their images based on the information in train.txt
	loadFaceImgs("train.txt");

	//Do PCA on all of the people we have
	printf("Doing PCA on all persons...\n");
	for (int i=0; i<people.size(); i++)
		people.at(i).doPCA();

	while(true) {
		
		//grab the camera frame
		frame = getCameraFrame();

		if(!frame){
			break;
		}

		faceRect = detectHaarClassifier(frame, faceCascade); //detect the face
		if(faceRect.width>0) { //make sure you found a face

			//break the face into multiple parts 
			faceImg = cropImage(frame, faceRect); //the face as a whole

		}

		//equalize the face
		if(faceImg)
			processedFaceImg = equalizeImage(faceImg);
		
		// If the face rec database has been loaded, then try to recognize the person currently detected.
		if(processedFaceImg)
			if (nEigens > 0) {
				// project the test image onto the PCA subspace
				cvEigenDecomposite(
					processedFaceImg,
					nEigens,
					eigenVectArr,
					0, 0,
					NULL,
					projectedTestFace);
	
				// Check which person it is most likely to be.
				float confidence = 0;
				Person nearestPerson = findNearestNeighbor(projectedTestFace, &confidence);
				nearest  = trainPersonNumMat->data.i[iNearest];

				printf("Most likely person in camera: '%s' (confidence=%f.\n", personNames[nearest-1].c_str(), confidence);

			}//endif nEigens

		cvShowImage("Camera Feed", frame);

		//ESC to exit
		char c = cvWaitKey(33);
		if( c == 27 ) 
			break;
	}
	
	cvReleaseCapture( &camera );
	cvDestroyWindow( "Camera Feed" );

	return 0;
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
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_WIDTH, 320 );
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_HEIGHT, 240 );
		// Wait a little, so that the camera can auto-adjust itself
		Sleep(1000);	// (in milliseconds)
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
#endif

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
	printf("[Object Detection took %d ms and found %d objects]\n", cvRound( time/((double)cvGetTickFrequency()*1000.0) ), detectedObjectRectangles->total );

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

// Read the names & image filenames of people from a text file, and load all those images listed.
int loadFaceImgs(char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iFace, nFaces=0;

	// open the input file
	if( !(imgListFile = fopen(filename, "r")) )
	{
		fprintf(stderr, "Can\'t open file %s\n", filename);
		return 0;
	}

	// count the number of faces
	while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
	rewind(imgListFile);

	//Create the Person objects and store the images and names
	for(iFace=0; iFace<nFaces; iFace++)
	{
		char personName[256];
		string sPersonName;
		int personInset = 0;

		// read their name and the image filename.
		fscanf(imgListFile, "%s %s", personName, imgFilename);
		sPersonName = personName;
		//printf("Got %d: <%s>, <%s>.\n", iFace, personName, imgFilename);

		//Check if this person already exists
		bool nameExists = false;
		for (unsigned int i=0; i<people.size();i++){
			if(people.at(i).getName().compare(sPersonName) == 0) {
				nameExists = true;
				personInset = i;
			}
		}
		//if not, add them
		if (!(nameExists)){
			personInset = 0;
			Person person(sPersonName);
			people.push_back(person);
		}
		
		IplImage *originalImage = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);
		CvRect faceRect;

		//find the face in the training image
		if(originalImage)
			faceRect = detectHaarClassifier(originalImage, faceCascade);
		else
			faceRect = cvRect(-1,-1,-1,-1);
		
		if(faceRect.width>0) { //make sure you found a face
			originalImage = cropImage(originalImage, faceRect); //the face as a whole
			// load the face image and add it to the person
			people.at(personInset).addFace(equalizeImage(originalImage));
			
		} else {
			fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
			nFaces--;
			continue;
		}

		cvReleaseImage(&originalImage);
	}

	fclose(imgListFile);

	printf("Data loaded from '%s': (%d images of %d people).\n", filename, nFaces, people.size());
	printf("People: ");
	if (people.size() > 0)
		printf("<%s>", people.at(0).getName().c_str());
	if (people.size() > 1)
		for (unsigned int i=1; i<people.size(); i++) {
			printf(", <%s>", people.at(i).getName().c_str());
		}
	printf(".\n");

	return nFaces;
}

// Find the most likely person based on a detection. Returns the index, and stores the confidence value into pConfidence.
Person* findNearestNeighbor(float * projectedTestFace, float *pConfidence)
{
	//double leastDistSq = 1e12;
	double leastDistSq = DBL_MAX;
	Person* nearestPerson;

	//TODO make the faceImgs public so you can access them faster
	for(int iPerson=0; iPerson<people.size(); iPerson++){
		for(int iTrain=0; iTrain<people.at(iPerson).getFaceImgs().size(); iTrain++)
		{
			double distSq=0;

			for(int i=0; i<people.at(iPerson).getEigens().size(); i++)
			{
				float d_i = projectedTestFace[i] - people.at(iPerson).getProjectedTrainFaceMat()->data.fl[iTrain*people.at(iPerson).getEigens().size() + i];

				distSq += d_i*d_i / people.at(iPerson).getEigenValMat()->data.fl[i];  // Mahalanobis distance (might give better results than Eucalidean distance)
				//distSq += d_i*d_i; // Euclidean distance.
			}

			if(distSq < leastDistSq)
			{
				leastDistSq = distSq;
				nearestPerson = &people.at(iPerson);
			}
		}
	}

	// Return the confidence level based on the Euclidean distance,
	// so that similar images should give a confidence between 0.5 to 1.0,
	// and very different images should give a confidence between 0.0 to 0.5.
	//TODO make the imgs and eigens the total that exist, but I'd rather it be something more  meaningful
	*pConfidence = 1.0f - sqrt( leastDistSq / (float)(nearestPerson->getFaceImgs().size() * nearestPerson->getEigens().size()) ) / 255.0f;

	// Return the found index.
	return nearestPerson;
}*/