#include <iostream>

using namespace std;

#define INDEX_FLOATMAT(f, i, j, n) (f[(i*n)+(j)])

#define ARG_W 3
#define ARG_H 4


float * transpose(const float * x, const int width, const int height){
  float * ret = (float *) malloc(sizeof(float) * width * height);
  int new_width = height;
  int new_height = width;
  for(int i = 0; i < new_width; i++){
    for(int j = 0; j < new_height; j++){
      INDEX_FLOATMAT(ret, i, j, new_width) = INDEX_FLOATMAT(x, j, i, width);
    }
  }
  /*
  for(int i = 0; i < width; i++){
    for(int j = 0; j < height; j++){
      INDEX_FLOATMAT(ret, j, i, height) = INDEX_FLOATMAT(x, i, j, width);
    }
  }
  */
  return ret;
}

void print_floatmat(const float * fp, int height, int width){
	for(int j = 0; j < height; j++){
		for(int i = 0; i < width; i++){
			cout << INDEX_FLOATMAT(fp, i, j, width) << ' ';
		}
		cout << endl;
	}
	cout << endl;
}

void print_flat(const float * data, int size){
  for(int i = 0; i < size; i++){
    cout << data[i] << ' ';
  }
  cout << endl;
}

int main(int argc, char ** argv){
	float * arg = (float *) malloc(sizeof(float) * ARG_W*ARG_H);
	for(int i = 0; i < ARG_W*ARG_H; i++){
		arg[i] = 1.0f + (1.0f*i);
	}
	float * transposed = transpose(arg, ARG_W, ARG_H);
	
	
	print_floatmat(arg, ARG_H, ARG_W);
	print_flat(arg, ARG_H*ARG_W);
	cout <<  "\n\n";
	
	print_floatmat(transposed, ARG_W, ARG_H);
	print_flat(transposed, ARG_H*ARG_W);
	cout << "\n\n";
	
	free(arg);
	free(transposed);

	return 0;
}


