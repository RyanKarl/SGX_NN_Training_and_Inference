//Testing file for hashing

#include <iostream>
using std::cout;

#include "hash.h"

int main(int argc, char ** argv){
  const static unsigned int ARRSIZE = 101;
  float dat[ARRSIZE];
  for(unsigned int i = 0; i < ARRSIZE; i++){
    dat[i] = 1.0*i;
  }
  cout << float_hash(dat, ARRSIZE) << '\n';
}
