//enclave_functions.c
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

//#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <fcntl.h>

#include <vector>
#include <string>
#include <fstream>


#include "Enclave.h"
#include "utilities.h"

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;

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

/*
int activate(float * data_in, int in_height, int in_width, 
  float * data_out, int out_height, int out_width){
  //Use the below if things are done in-place
  //data_outshape must have DATA_DIMENSIONS elements

  //Using tanh as the activation function
  assert(out_height == in_height && out_width == in_width);
  for(int j = 0; j < out_height*out_width; j++){
    data_out[j] = tanh(data_in[j]);
  }  

  return 0;
}
*/

int activate(float * data, int height, int width){
  if(height <= 0 || width <= 0){
    return 1;
  }
  for(int i = 0; i < height * width; i++){
    data[i] = tanh(data[i]);
  }
  return 0;
}

//TODO complete this
void unmask(float * data, int width, int height, float * mask_data, float * input_layer){
  return;
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
int parse_structure(char * network_structure_fname, vector<layer_file_t> & layer_files, unsigned int & num_inputs, int & input_height, int & input_width){
  
  ifstream network_ifs(network_structure_fname);
  layer_files.clear();
  //num_inputs is batch size
  network_ifs >> num_inputs >> input_height >> input_width;
  for(unsigned int i = 0; i < num_inputs; i++){
    layer_file_t lft;
    network_ifs >> lft.height >> lft.width >> lft.filename;
    //TODO change for different networks
    lft.type = FULLY_CONNECTED;
    layer_files.push_back(lft);
  }
  network_ifs.close();
  return layer_files.size() == num_inputs? 0 : 1;
}

#define MNIST_VALS 784

//Assumes the existence of a large enough buffer
int csv_getline(const char * input_csv_name, float * vals, 
  char * label, size_t num_vals){

  static std::ifstream ifs(input_csv_name);

  ifs >> *label;

  //*vals = (unsigned char *) malloc(num_vals*sizeof(unsigned char));
  for(unsigned int i = 0; i < num_vals; i++){
    ifs >> (vals)[i];
    //Normalize float value
    vals[i] /= (1 << CHAR_BIT);
  }
  return 0;
}

//Assume all buffers are allocated
//Read in NORMALIZED values
int csv_getbatch(char * input_csv_name, float ** vals, unsigned int num_inputs, unsigned int num_vals, char * labels){
  std::ifstream ifs(input_csv_name);
  for(unsigned int i = 0; i < num_inputs; i++){
    ifs >> labels[i];
    for(unsigned int j = 0; j < num_vals; j++){
      ifs >> vals[i][j];
      //Normalize float value
      vals[i][j] /= (1 << CHAR_BIT);
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
int read_weight_file(const char * filename, int num_elements, float * buf){
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

//Assumes a buffer is allocated
int read_all_weights(const vector<layer_file_t> & layers, float ** bufs){
  for(size_t i = 0; i < layers.size(); i++){
    bufs[i] = (float *) malloc(layers[i].height * layers[i].width * sizeof(float));
    //Should check return val
    read_weight_file(layers[i].filename.c_str(), layers[i].height * layers[i].width, bufs[i]);
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
  int res = fread((void *) buf, 1, total_bytes, instream);
  return (res != 0);
}



int write_stream(void * buf, size_t total_bytes){
  if(buf == NULL){
    return 1;
  }
  int res = fwrite((void *) buf, total_bytes, 1, outstream);
  //Optional for performance
  //fflush(oustream);
  return (res != 0);
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

void mask(float * data, int len, float * mask_data){
  for(int i = 0; i < len; i++){
    data[i] += mask_data[i];
  }
  return;
}

//Need OCALLS for pipe I/O, setup, teardown
int enclave_main(char * network_structure_fname, char * input_csv_filename, char * inpipe_fname, char * outpipe_fname){

  unsigned int num_layers;
  vector<layer_file_t> layer_files;
  unsigned int num_inputs;
  int input_height; 
  int input_width;
  

  if(parse_structure(network_structure_fname, layer_files, 
    num_inputs, input_height, input_width)){
    //Error!
  }
  
  num_layers = layer_files.size();


  float ** layer_data;
  layer_data = (float **) malloc(sizeof(float *) * num_layers);
  //Read in all layer data
  if(read_all_weights(layer_files, layer_data)){
    //Error
  }
  
  
  for(unsigned int input_idx = 0; input_idx < num_inputs; input_idx++){
  
    float * input_data;
    
    //Check that the array's size is equal to the expected height*width
    for(unsigned int layer_idx = 0; layer_idx < num_layers; layer_idx++){
    
      int data_height, data_width;
    
      if(layer_idx == 0){
        data_height = input_height;
        data_width = input_width;
        //Read in input to an array
        //Allocate buffer
        input_data = (float *) malloc(data_height*data_width*sizeof(float));
        char data_label;
        if(input_data == NULL){
          //Error
        }    
        //Read to the buffer
        if(csv_getline(input_csv_filename, input_data, &data_label, data_height*data_width)){
          //Error
        }
      }
      else{
        data_height = layer_files[layer_idx].height;
        data_width = layer_files[layer_idx].width;
      }
      //If not first layer (i.e. raw input), then the input data is already initialized
      
      //Mask the current input
      //First, get the random mask
      float * mask_data = (float *) malloc(sizeof(float) * data_height*data_width);
      //Cast should be explicit, for the non-SGX version
      rand_bytes((unsigned char *) mask_data, sizeof(float) * data_height*data_width);
      //Next, mask the data
      mask(input_data, data_height*data_width, mask_data);

      
      //Send it and a layer of weights to the GPU
      //Send first dimensions, then data (twice)
      int out_dims[2] = {data_height, data_width};
      if(write_stream((void *) out_dims, sizeof(out_dims))){
        //Error
      }
      if(write_stream((void *) input_data, sizeof(float)*data_height*data_width)){
        //Error
      }
      if(write_stream((void *) out_dims, sizeof(out_dims))){
        //Error
      }
      if(write_stream((void *) layer_data[layer_idx], sizeof(float)*data_height*data_width)){
        //Error
      }
      
      //Get back a result C ?= A*B
      //Read in dimensions, then data
      int in_dims[2] = {-1, -1};
      int next_height, next_width;
      if(read_stream((void *) in_dims, sizeof(in_dims))){
        //Error
      }
      next_height = in_dims[0];
      next_width = in_dims[1];
      float * gpu_result = (float *) malloc(sizeof(float)*next_height*next_width);
      if(read_stream((void *) gpu_result, sizeof(float)*next_height*next_width)){
        //Error
      }

      //Validate C through Frievald's algorithm
      //If it fails, send {-1, -1} back to the GPU and exit
      if(frievald(input_data, layer_data[layer_idx], gpu_result, 
  data_height, data_width, data_height, data_width, next_height, next_width)){
        //Verification failed!
        int failed_resp[2] = {-1, -1};
        if(write_stream((void *) failed_resp, sizeof(failed_resp))){
          //Error
        }
        return 1;
      }

      //Unmask
      unmask(gpu_result, next_height, next_width, mask_data, layer_data[layer_idx]);
      
      //Cleanup random mask
      free(mask_data);
      mask_data = NULL;
      //Cleanup original input
      free(input_data);
      input_data = NULL;
      //Activation function on the data
      if(activate(gpu_result, next_height, next_width)){
        //Error
      }
      
      //TODO backpropagation
      
      //Setup things for the next iteration
      if(layer_idx){
        input_data = gpu_result;
      }
    }
  }
  
  //Cleanup layers
  for(size_t i = 0; i < layer_files.size(); i++){
    free(layer_data[i]);
  }
  free(layer_data);
  

  return 0;

}
