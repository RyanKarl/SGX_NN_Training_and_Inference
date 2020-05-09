#ifndef OCALLS_H
#define OCALLS_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <fcntl.h>
#include <string.h>
#include <limits.h>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::cout;
using std::cerr;
using std::endl;

#define STRUCTURE_BUFLEN 1024


static FILE * instream;
static FILE * outstream;

int init_streams(char * inpipe_fname, char * outpipe_fname){

  int input_pipe = 0;
  if(inpipe_fname){
    input_pipe = open(inpipe_fname, O_RDONLY);
    if(input_pipe == -1){
      fprintf(stderr, "ERROR: could not open input pipe %s\n", inpipe_fname);
      return 1;
    }
    instream = fdopen(input_pipe, "r");
  } 
  else{
    instream = stdin;
  }
  
  int output_pipe = 0;
  if(outpipe_fname){
#define OUTFILE_PERMS 0644    
    output_pipe = open(outpipe_fname, O_WRONLY | O_CREAT, OUTFILE_PERMS);
    if(output_pipe == -1){
      fprintf(stderr, "ERROR: could not open output pipe %s\n", outpipe_fname);
      return 1;
    }
    outstream = fdopen(output_pipe, "w");
  }  
  else{
    outstream = stdout;
  }
  return 0;
}

//Requires a buffer to be allocated
int read_stream(void * buf, size_t total_bytes){
  int res = fread((void *) buf, 1, total_bytes, instream);
  return (res != 0)? 0 : 1;
}


int write_stream(void * buf, size_t total_bytes){
  if(buf == NULL){
    return 1;
  }
  int res = fwrite((void *) buf, total_bytes, 1, outstream);
  fflush(outstream);
  return (res != 0)? 0 : 1;
}

int close_streams(){
  if(instream != stdin){
    fclose(instream);
    free(instream);
    instream = NULL;
  }
  if(outstream != stdout){
    fclose(outstream);
    free(outstream);
    outstream = NULL;
  }
  return 0;
}


int csv_getline(char * csv_filename, float * vals, char * label, size_t num_vals){
  static std::ifstream ifs(csv_filename);
  if(!ifs.good()){
    return 1;
  }
  
  char comma_holder;
  for(unsigned int i = 0; i < num_vals; i++){
    ifs >> vals[i] >> comma_holder;
    //Normalize float value
    vals[i] /= (1 << CHAR_BIT);
  }
  ifs >> *label;
  return 0;
}

void print_out(char * msg, int error){
  if(error){
    cerr << msg << endl;
  }
  else{
    cout << msg << endl;
  }
}


int file_to_string(char * fname, char * out){
  ifstream network_ifs(fname);
  std::ostringstream oss;
  assert(network_ifs.good());
  
  oss << network_ifs.rdbuf();

  unsigned int len = oss.str().size() + 1;
  if(len >= STRUCTURE_BUFLEN){
    return 1;
  }
  strncpy(out, oss.str().c_str(), len);
  network_ifs.close();
  return 0;
}

//Assumes a buffer is allocated
//This function should be an OCALL
int read_weight_file(char * filename, size_t num_elements, float * buf){
  if(!num_elements){
    return 1;
  }
  FILE * f = fopen(filename, "rb");
  if(!f){
    return 1;
  }
  long bytes_read = fread(buf, sizeof(float), num_elements, f);
  if(bytes_read*sizeof(float) != (unsigned long) num_elements){
    return 1;
  }
  if(fclose(f)){
    return 1;
  }
  return 0;
}

#endif