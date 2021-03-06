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
#include <chrono>

#include "../Enclave/Enclave_Defines.h"

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::cout;
using std::cerr;
using std::endl;
using namespace std::chrono;


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
  return (res == (int)total_bytes)? 0 : res;
}


int write_stream(void * buf, size_t total_bytes){
  if(buf == NULL){
    return 1;
  }
  int res = fwrite((void *) buf, 1, total_bytes, outstream);
  /*
  if(res != (int) total_bytes){
    fprintf(stderr, "%s\n", explain_fwrite((void *) buf, 1, total_bytes, outstream));
  }
  */
  fflush(outstream);
  return (res == (int)total_bytes)? 0 : res;
}

//fclose frees memory
int close_streams(){
  if(instream != stdin){
    fclose(instream);
    instream = NULL;
  }
  if(outstream != stdout){
    fclose(outstream);
    outstream = NULL;
  }
  return 0;
}




int csv_getline(char * csv_filename, FP_TYPE * vals, unsigned int * label, size_t vals_buffer_size, int reset = 0){
  static std::ifstream ifs(csv_filename);

  assert(vals_buffer_size % sizeof(FP_TYPE) == 0);

  if(vals == NULL){
    vals = (FP_TYPE *) malloc(vals_buffer_size);
  }
  if(label == NULL){
    label = (unsigned int *) malloc(sizeof(unsigned int));
  }

  if(reset){
    ifs.close();
    ifs.open(csv_filename);
  }

  if(!ifs.good()){
    return 1;
  }
  char comma_holder;
  for(unsigned int i = 0; i < vals_buffer_size/sizeof(FP_TYPE); i++){
    ifs >> vals[i] >> comma_holder;
    //Normalize
    vals[i] /= (1 << CHAR_BIT);
  }
  //cout << endl;
  int label_i;
  ifs >> label_i;
  *label = label_i;
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

int floats_to_csv(char * fname, size_t data_buf_size, FP_TYPE * data){
  ofstream ofs(fname);
  assert(data_buf_size % sizeof(FP_TYPE) == 0);
  for(size_t i = 0; i < data_buf_size/sizeof(FP_TYPE); i++){
    ofs << data[i];
    if(i != (data_buf_size/sizeof(FP_TYPE))-1){
      ofs << ',';
    }
  }
  return ofs.good() ? 0 : 1;
}

//Does not count null terminator 
size_t file_size(char * fname){
  ifstream network_ifs(fname);
  std::ostringstream oss;
  assert(network_ifs.good());
  
  oss << network_ifs.rdbuf();
  network_ifs.close();
  return oss.str().size();
}


int file_to_string(char * fname, char * out, size_t str_buf_len){
  ifstream network_ifs(fname);
  std::ostringstream oss;
  assert(network_ifs.good());

  if(out == NULL){
    out = (char *) malloc(sizeof(char) * str_buf_len);
  }
  
  oss << network_ifs.rdbuf();

  unsigned int len = oss.str().size() + 1;
  if(len != str_buf_len){
    return 1;
  }
  strncpy(out, oss.str().c_str(), len);
  network_ifs.close();
  return 0;
}

//Assumes a buffer is allocated
int read_weight_file(char * filename, size_t buf_size, FP_TYPE * buf){
  if(!buf_size){
    return 1;
  }
  FILE * f = fopen(filename, "rb");
  if(!f){
    return 1;
  }
  long bytes_read = fread(buf, sizeof(FP_TYPE), buf_size/sizeof(FP_TYPE), f);
  if((unsigned long) bytes_read*sizeof(FP_TYPE) != (unsigned long) buf_size){
    fclose(f);
    return 1;
  }
  if(fclose(f)){
    return 1;
  }
  return 0;
}

//Assumes comma-delimited
int read_weight_file_plain(char * filename, size_t bufsize, FP_TYPE * buf){
  ifstream fs(filename);
  assert(bufsize % sizeof(FP_TYPE) == 0);
  if(buf == NULL){
    buf = (FP_TYPE *) malloc(bufsize);
  }
  char comma_holder;
  for(size_t i = 0; i < bufsize/sizeof(FP_TYPE); i++){
    if(!fs.good()){
      return 1;
    }
    fs >> buf[i] >> comma_holder;   
  }
  return 0;
}

static high_resolution_clock::time_point overall_start;
static high_resolution_clock::time_point forward_start;
static high_resolution_clock::time_point backprop_start;

static double overall_duration;
static std::vector<double> forward_times;
static std::vector<double> backprop_times;

//Bad programming, but more efficient
int start_timing(int task){
  switch(task){
    case TASK_ALL:{
      overall_start = high_resolution_clock::now();
      break;
    }
    case TASK_FORWARD:{
      forward_start = high_resolution_clock::now();
      break;
    }
    case TASK_BACKPROP:{
      backprop_start = high_resolution_clock::now();
      break;
    }
    default:{
      return 1;
    }
  }
  return 0;
}

int finish_timing(int task){
  high_resolution_clock::time_point end = high_resolution_clock::now();
  switch(task){
    case TASK_ALL:{
      overall_duration = duration_cast<std::chrono::nanoseconds>(end - overall_start).count();
      break;
    }
    case TASK_FORWARD:{
      forward_times.push_back(duration_cast<std::chrono::nanoseconds>(end - forward_start).count());
      break;
    }
    case TASK_BACKPROP:{
      backprop_times.push_back(duration_cast<std::chrono::nanoseconds>(end - backprop_start).count());
      break;
    }
    default:{
      return 1;
    }
  }
  return 0;
}
//Not technically an OCALL, but should be in this file to access stored timings
void print_timings(std::ostream & os){
  os << "Total_duration: " << overall_duration << endl;
  for(size_t i = 0; i < forward_times.size(); i++){
    os << "Forward_pass: " << forward_times[i] << endl;
  }
  for(size_t j = 0; j < backprop_times.size(); j++){
    os << "Backprop: " << backprop_times[j] << endl;
  }
  
}


#endif
