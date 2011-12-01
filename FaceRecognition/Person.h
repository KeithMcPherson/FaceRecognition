//Person declaration, contains their name, images, eigenvectors, and whatever other details about a person I feel like adding

#include <string>
#include <vector>
#include "opencv\cv.h"
#include "opencv\cvaux.h"
#include "opencv\highgui.h"

using namespace std;

class Person
{
public:
	Person(string name);
	string getName();
	vector<IplImage*> getFaceImgs();
	void addFace(IplImage * faceImg);

private:
	string name;
	vector<IplImage *> faceImgs;

};