//Enclave.h
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

#ifndef ENCLAVE_H
#define ENCLAVE_H

#include "Enclave_Defines.h"

#if defined(__cplusplus)
extern "C" {
#endif

//a, b, c are flattened 2d arrays
//Can consider moving validation outside
int frievald(FP_TYPE * a, FP_TYPE * b, FP_TYPE * c, 
  int a_height, int a_width, int b_height, int b_width, int c_height, int c_width);

//Return 1 if verification fails, 0 if successful
int verify_frievald(FP_TYPE * data, int a_height, int a_width, int b_height, int b_width, int c_height, int c_width);

//Return 1 if activation fails, 0 if successful
//int activate(FP_TYPE * data, int height, int width); //Reference or value?

//Trust that data_in and data_out have the correct size
//Buffers must be allocated outside the enclave!
int verify_and_activate(FP_TYPE * data_in, int a_height, int a_width, int b_height, int b_width, int c_height, int c_width, FP_TYPE * data_out, int out_height, int out_width);

/*
Suggested format for network structure files:
First line contains one positive integer stating the number of inputs, and two for the height and width of the input size, plus the input filename
Lines after are in the format:
height width filename type
*/
//TODO make this an OCALL



int enclave_main(char * network_structure_fname, char * input_csv_filename, char * inpipe_fname, char * outpipe_fname, char * weights_outfile, int backprop, int verbose);

void mask(FP_TYPE * data, int len, FP_TYPE * mask_data);

void unmask(FP_TYPE * data, int width, int height, FP_TYPE * mask_data, FP_TYPE * input_layer);

int init_streams(char * inpipe_fname, char * outpipe_fname);
int read_stream(void * buf, size_t total_bytes);
int write_stream(void * buf, size_t total_bytes);
int close_streams();
int csv_getline(char * input_csv_name, FP_TYPE * vals, unsigned int * label, size_t num_vals);
void print_out(char * msg, int error);
int file_to_string(char * fname, char * out, size_t str_buf_len);
int read_weight_file(char * filename, size_t num_elements, FP_TYPE * buf);
int FP_TYPEs_to_csv(char * fname, size_t num_elts, FP_TYPE * data);
size_t file_size(char * fname);

#if defined(__cplusplus)
}
#endif

#endif
