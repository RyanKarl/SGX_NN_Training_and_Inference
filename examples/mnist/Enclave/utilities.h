//Jonathan S. Takeshita (jtakeshi)
#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>

//read_file and write_file are utility functions to read/write the contents of a file to/from a buffer of float (NOT null-terminated).
//The functions return 0 for normal operation, -1 for a memory allocation error, and 1 for all other errors.

int read_file(float ** buffer, long * len, char * filename){
  if(!len){
    return 1;
  }
  FILE * f = fopen(filename, "rb");
  if(!f){
    return 1;
  }
  if(fseek(f, 0, SEEK_END)){
    return 1;
  }
  (*len) = ftell(f);
  *buffer = (float *) malloc((*len)*sizeof(float));
  if(!*buffer){
    return -1;
  }
  if(fseek(f, 0, SEEK_SET)){
    return 1;
  }
  long bytes_read = fread(*buffer, sizeof(float), *len, f);
  if(bytes_read != *len){
    return 1;
  }
  if(fclose(f)){
    return 1;
  }
  return 0;
}

int write_file(const float * buffer, long len, char * filename){
  FILE * f = fopen(filename, "wb");
  if(!f){
    return 1;
  }
  long bytes_written = fwrite(buffer, sizeof(float), len, f);
  if(bytes_written != len){
    return 1;
  }
  if(fclose(f)){
    return 1;
  }
  return 0;
}

#endif
