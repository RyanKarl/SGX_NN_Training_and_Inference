//enclave_functions.c
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

//#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>

#include "Enclave.h"
#include "utilities.h"

//0 is height, 1 is width
int frievald(float * a, float * b, float * c, 
  int a_height, int a_width, int b_height, int b_width, int c_height, int c_width)
{
  //Mult. is defined
  assert(a_width == b_height);
  //Output dims are correct
  assert(c_height == a_height);
  assert(c_width == b_width);
  //Create a random vector r
  size_t num_bytes_randarr = (b_width/CHAR_BIT) + (b_width%CHAR_BIT? 1 : 0);
  unsigned char * r = (unsigned char *) calloc(num_bytes_randarr, sizeof(unsigned char));
  if(!r){
    assert(r && "calloc failed");
  }
  rand_bytes(r, num_bytes_randarr);

  //Hope that calloc properly sets bits to 0
  //Calculate br, cr in the same loop
  float * br = (float *) calloc(b_width, sizeof(float));
  float * cr = (float *) calloc(c_width, sizeof(float));
  if(!br){
    assert(br && "malloc failed");
  }
  if(!cr){
    assert(cr && "malloc failed");
  }
  for (int i = 0; i < b_height; i++){
      for (int j = 0; j < b_width; j++){
          br[i] += INDEX_FLOATMAT(b, i, j, b_height) * ((unsigned char)INDEX_BITARR(r, j));
      }
  }

  for (int i = 0; i < c_height; i++){
      for (int j = 0; j < c_width; j++){
          cr[i] += INDEX_FLOATMAT(c, i, j, c_height) * ((unsigned char)INDEX_BITARR(r, j));
      }
  }

  free(r);
  r = NULL;

  float * axbr = (float *) calloc(b_height, sizeof(float));
  if(!axbr){
    assert(axbr && "malloc failed");
  }
  assert(axbr && "Allocating axbr failed!");
  for (int i = 0; i < b_height; i++){
      for (int j = 0; j < b_width; j++){
          axbr[i] += INDEX_FLOATMAT(a, i, j, b_height) * br[j];
      }
  }

  free(br);
  br = NULL;

  for (int i = 0; i < c_width; i++){
    if (FLOAT_CMP(axbr[i], cr[i])){
        free(axbr);
        free(cr);
        axbr = cr = NULL;
        return 1;
    }
  }

  free(axbr);
  axbr = NULL;
  free(cr);
  cr = NULL;

  return 0;
}

int verify_frievald(float * data, int a_height, int a_width, int b_height, int b_width, int c_height, int c_width){
  float * matrix_offsets[NUM_MATRICES];
  matrix_offsets[0] = data;
  matrix_offsets[1] = matrix_offsets[0] + (a_height*a_width);
  matrix_offsets[2] = matrix_offsets[1] + (b_height*b_width);
  for(unsigned int j = 0; j < K_PROBABILITY; j++){
  	if(frievald(matrix_offsets[0], matrix_offsets[1], matrix_offsets[2], a_height, a_width, b_height, b_width, c_height, c_width)){
  		return 1;
  	}
  }
  return 0;
}

int activate(float * data_in, int in_height, int in_width, 
  float * data_out, int out_height, int out_width){
  //Use the below if things are done in-place
  //data_outshape must have DATA_DIMENSIONS elements
  /*
  for(unsigned int i = 0; i < MAT_DIM; i++){
    matrix_n_out[i] = matrix_n[i];
  }
  */

  //Using tanh as the activation function
  assert(out_height == in_height && out_width == in_width);
  for(int j = 0; j < out_height*out_width; j++){
    data_out[j] = tanh(data_in[j]);
  }  

  return 0;
}

int verify_and_activate(float * data_in, int a_height, int a_width, int b_height, int b_width, int c_height, int c_width,
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
  if(activate(activation_buffer_enclave, c_height, c_width, data_out, out_height, out_width)){
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

int parse_structure(char * network_structure_fname, vector<layer_file_t> & layer_files, unsigned int & num_layers,
unsigned int & num_inputs, int & input_height, int & input_width, string & input_filename){
  ifstream network_ifs(network_structure_fname);
  layer_files.clear();
  network_ifs >> num_inputs >> input_height >> input_width >> input_filename;
  for(unsigned int i = 0; i < num_inputs; i++){
    layer_file_t lft;
    network_ifs >> lft.height >> lft.width >> lft.filename;
    layer_files.push_back(lft);
  }
  ifs.close();
  return layer_files.size() == num_inputs? 0 : 1;
}

#define MNIST_VALS 784

//Assumes the existence of a large enough buffer
int csv_getline(char * input_csv_name, float * vals, 
  unsigned char * label, unsigned int num_vals){

  static std::ifstream ifs(input_csv_name);

  ifs >> *label;

  //*vals = (unsigned char *) malloc(num_vals*sizeof(unsigned char));
  for(unsigned int i = 0; i < num_vals; i++){
    ifs >> (vals)[i];
  }
  return 0;
}

//Assume all buffers are allocated
//Read in NORMALIZED values
int csv_getbatch(char * input_csv_name, float ** vals, unsigned int num_inputs, unsigned int num_vals, unsigned char * labels){
  std::ifstream ifs(input_csv_name);
  for(unsigned int i = 0; i < num_inputs; i++){
    ifs >> labels[i];
    for(unsigned int j = 0; j < num_vals; j++){
      ifs >> vals[i][j];
      vals[i][j] /= (float)(1 << CHAR_BIT);
    }
  }
  ifs.close();
  return 0;
}

int mask(float * input, float * masks, unsigned int input_size){
  for(unsigned int i = 0; i < input_size; i++){
    input[i] += masks[i];
  }
  return 0;
}

//Assumes a buffer is allocated
//This function should be an OCALL
int read_weight_file(char * filename, int num_elements, float * buf){
  if(!num_elements){
    return 1;
  }
  FILE * f = fopen(filename, "rb");
  if(!f){
    return 1;
  }
  long len;
  long bytes_read = fread(buf, sizeof(float), num_elements, f);
  if(bytes_read*sizeof(float) != num_elements){
    return 1;
  }
  if(fclose(f)){
    return 1;
  }
  return 0;
}

//Assumes a buffer is allocated
int read_all_weights(std::string * filenames, int num_layers, int * elts_per_layer, float ** bufs){
  for(int i = 0; i < num_layers; i++){
    bufs[i] = (float *) malloc(elts_per_layer[i]*sizeof(float));
    //Should check return val
    read_weight_file(filenames[i].c_str(), elts_per_layer[i], bufs[i]);
  }
  return 0;
}

//For inputs: read in entire batches to enclave memory from ordinary file through OCALLs, and free() data when done.

static FILE * instream;
static FILE * outstream;

int init_streams(char * inpipe_fname, char * outpipe_fname){

  int input_pipe = 0;
  if(inpipe_fname){
    input_pipe = open(inpipe_fname, O_RDONLY);
  } 
  if(input_pipe == -1){
    fprintf(stderr, "ERROR: could not open input pipe %s\n", inpipe_fname);
    return 1;
  }
  int output_pipe = 0;
  if(outpipe_fname){
    output_pipe = open(outpipe_fname, O_WRONLY);
  }  
  if(output_pipe == -1){
    fprintf(stderr, "ERROR: could not open output pipe %s\n", outpipe_fname);
    return 1;
  }

  if(!inpipe_fname){
    instream = stdin;
  }
  else{
    instream = fdopen(input_pipe, "r");
  }
  if(!outpipe_fname){
    outstream = stdout;
  }
  else{
    outstream = fdopen(output_pipe, "w");
  }

  return 0;

}




//Requires a buffer to be allocated
int read_stream(void * buf, size_t total_bytes){
  fread((void *) buf, 1, total_bytes, instream);
}



int write_stream(void * buf, size_t total_bytes){
  fwrite((void *) buf, total_bytes, 1, outstream);
  //Optional for performance
  //fflush(oustream);
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
  /*
  if(use_std_io){
    close(input_pipe);
    close(output_pipe);
  }
  */
  return 0;
}

//Need OCALLS for pipe I/O, setup, teardown
int enclave_main(char * network_structure_fname, char * input_csv_filename, char * inpipe_fname, char * outpipe_fname){

  unsigned int num_layers;
  vector<layer_file_t> layer_files;
  unsigned int num_layers;
  unsigned int num_inputs;
  int input_height; 
  int input_width;
  string input_filename;

  if(parse_structure(network_structure_fname, layer_files, num_layers,
    num_inputs, input_height, input_width, input_filename)){
    //Error!
  }

  unsigned int num_inputs;
  
  for(unsigned int input_idx = 0; input_idx < num_inputs; input_idx++){
    //Read in input batch to an array

    //Check that the array's size is equal to the expected height*width
    for(unsigned int layer_idx = 0; layer_idx < num_layers; layer_idx++){
      //Mask the current input

      //Send it and a layer of weights to the GPU
      //Send first dimensions, then data (twice)

      //Get back a result C ?= A*B
      //Read in dimensions, then data

      //Validate C through Frievald's algorithm
      //If it fails, send {-1, -1} back to the GPU and exit

      //Unmask

      //Activation function on the data

    }
    //Clean up current batch
  }

  return 0;

}