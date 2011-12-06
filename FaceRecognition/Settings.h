//Just some settings, should probably be put into some kind of .ini file

#define USE_OLD_MAIN //New main doesn't work exactly, just plan to reimplement it in the old main


const int	FACE_WIDTH = 100;
const int	FACE_HEIGHT = 100;
//const bool	SAVE_EIGENFACE_IMAGES = 1;
const bool	ROTATE_INPUT_IMAGES = false;
const int	ROTATE_MAX = 10; //maximum rotation in each direction, in degrees
const int	ROTATE_STEP = 1; //the amount to increment between each rotation, in degrees