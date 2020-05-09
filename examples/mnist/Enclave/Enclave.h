//Enclave.h
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

#ifndef ENCLAVE_FUNCTIONS_H
#define ENCLAVE_FUNCTIONS_H

#include "Enclave_Defines.h"

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef NENCLAVE
# include <sgx_trts.h>
# define rand_bytes(r, n_bytes) (sgx_read_rand((unsigned char *) r, n_bytes) )
#else
# include <stdlib.h> //Need this for rand
# include <assert.h>
inline void rand_bytes(unsigned char * r, size_t n_bytes){
  assert(r);
  for(size_t i = 0; i < n_bytes; i++){
  	r[i] = (unsigned char) rand();
  }
  /*
  for(int i = 0; i < n; i++){
    r[i] = 0;
    for(unsigned int j = 0; j < sizeof(int)*CHAR_BIT; j++){
      r[i] |= (int)((1 & rand()) << j);
    }
  }
  */
  return;
}
#endif

//a, b, c are flattened 2d arrays
//Can consider moving validation outside
int frievald(float * a, float * b, float * c, 
  int a_height, int a_width, int b_height, int b_width, int c_height, int c_width);

//Return 1 if verification fails, 0 if successful
int verify_frievald(float * data, int a_height, int a_width, int b_height, int b_width, int c_height, int c_width);

//Return 1 if activation fails, 0 if successful
int activate(float * data, int height, int width); //Reference or value?

//Trust that data_in and data_out have the correct size
//Buffers must be allocated outside the enclave!
int verify_and_activate(float * data_in, int a_height, int a_width, int b_height, int b_width, int c_height, int c_width,
<<<<<<< HEAD
<<<<<<< HEAD
 float * data_out, int out_height, int out_width){
  //Copy data to enclave space
  //Validate data here
  if(a_height < 0 || a_width < 0 ||
     b_height < 0 || b_width < 0 ||
     c_height < 0 || c_width < 0 ||
     out_height < 0 || out_width < 0){
    return 1;
  }
  int mult_ptr_offset = (a_height*a_width) + (b_height*b_width);
  int total_input_elts = mult_ptr_offset + (c_height*c_width);
  float * enclave_data = (float *) malloc(total_input_elts*sizeof(float));
  for(int i = 0; i < total_input_elts; i++){
    enclave_data[i] = data_in[i];
  }
  
  if(verify_frievald(enclave_data, a_height, a_width, b_height, b_width, c_height, c_width)){
    free(enclave_data);
    enclave_data = NULL;
    return 1;
  }
  
  float * activation_buffer_enclave = enclave_data + mult_ptr_offset;
  if(activate(activation_buffer_enclave, c_height, c_width)){
    free(enclave_data);
    enclave_data = NULL;
    return 1;
  }
  
  free(enclave_data);
  enclave_data = NULL;
  return 0;

}

/*
Suggested format for network structure files:
First line contains one positive integer stating the number of inputs, and two for the height and width of the input size, plus the input filename
Lines after are in the format:
height width filename type
*/
//TODO make this an OCALL


int file_to_string(const char * fname, char * out){
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

=======
 float * data_out, int out_height, int out_width);
>>>>>>> parent of 96e73a5... Moved to header-only for enclave stuff
=======
 float * data_out, int out_height, int out_width);
>>>>>>> parent of 96e73a5... Moved to header-only for enclave stuff


#ifndef ENCLAVE_MAIN_F
#define ENCLAVE_MAIN_F
int enclave_main(char * network_structure_fname, char * input_csv_filename, char * inpipe_fname, char * outpipe_fname, int verbose);
#endif

void mask(float * data, int len, float * mask_data);

void unmask(float * data, int width, int height, float * mask_data, float * input_layer);

int init_streams(char * inpipe_fname, char * outpipe_fname);
int read_stream(void * buf, size_t total_bytes);
int write_stream(void * buf, size_t total_bytes);
int close_streams();
int csv_getline(char * input_csv_name, float * vals, unsigned char * label, size_t num_vals);
void print_out(char * msg, int error);
int file_to_string(char * fname, char * out);

#if defined(__cplusplus)
}
#endif

#endif
