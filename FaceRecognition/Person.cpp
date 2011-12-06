//Person definitions

#include "Person.h"

Person::Person(string name){
	this->name = name;
}

//returns the name of this person
string Person::getName(){
	return this->name;
}

vector<IplImage*> Person::getFaceImgs(){
	return this->faceImgs;
}

//Adds a face image for this person
void Person::addFace(IplImage * faceImg){
	this->faceImgs.push_back(faceImg);
}
