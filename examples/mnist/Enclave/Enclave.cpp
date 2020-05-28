//enclave_functions.c
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

//#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include <vector>
#include <string>

#include "matrix.h"

#ifdef NENCLAVE
# include "Enclave.h"
# include "../App/ocalls.h"
#else
# include "Enclave_Defines.h"
# include "Enclave_t.h"
#endif

using std::vector;
using std::string;

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

//TODO change index macro to take width
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
  //TODO get less randomness
  float * r = (float *) malloc(sizeof(float) * b_width);
  for(int i = 0; i < b_width; i++){
    unsigned char rand_byte;
    rand_bytes(&rand_byte, 1);
    r[i] = (float) (rand_byte & 1);
  }

  //Hope that calloc properly sets bits to 0
  //Calculate br, cr in the same loop
  float * br;
  int br_w, br_h;
  float * cr;
  int cr_w, cr_h;
  
  matrix_multiply(b, b_width, b_height, 
    r, 1, b_width,
    &br, &br_w, &br_h, 0);
  assert(br_h == a_width);
  assert(br_w == 1);

  matrix_multiply(c, c_width, c_height,
    r, 1, b_width,
    &cr, &cr_w, &cr_h, 0);
  assert(cr_h == b_width);
  assert(cr_w == 1);

  free(r);
  r = NULL;

  float * axbr;
  int axbr_w, axbr_h;
  assert(axbr && "Allocating axbr failed!");
  matrix_multiply(a, a_width, a_height, 
    br, br_w, br_h,
    &axbr, &axbr_w, &axbr_h, 0);

  free(br);
  br = NULL;

  for (int i = 0; i < c_width; i++){
    //cout << "axbr[" << i << "] " << axbr[i] << " cr[" << i << "] " << cr[i] << endl;
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
  
  char str_in[STRUCTURE_BUFLEN] = {'\0'};


#ifdef NENCLAVE
  if(file_to_string(network_structure_fname, str_in, STRUCTURE_BUFLEN+1)){
    return 1;
  }
#else
  sgx_status_t ocall_status;
  int ocall_ret;
  ocall_status = file_to_string(&ocall_ret, network_structure_fname, str_in, STRUCTURE_BUFLEN+1);
  if(ocall_ret){
    return 1;
  }
#endif  
  //std::stringstream network_ifs(str_in);


  num_inputs = atoi(strtok(str_in, " \n"));
  input_height = atoi(strtok(NULL, " \n"));
  input_width = atoi(strtok(NULL, " \n"));
  unsigned int num_layers = atoi(strtok(NULL, " \n"));
  layer_files.reserve(num_layers);

  for(unsigned int i = 0; i < num_layers; i++){
    layer_file_t lft;

    lft.height = atoi(strtok(NULL, " \n"));
    lft.width = atoi(strtok(NULL, " \n"));
    lft.filename = strtok(NULL, " \n");
    lft.type = atoi(strtok(NULL, " \n"));

    layer_files.push_back(lft);
  }

  //num_inputs is batch size
  /*
  network_ifs >> num_inputs >> input_height >> input_width;
  for(unsigned int i = 0; i < num_inputs; i++){
    layer_file_t lft;
    network_ifs >> lft.height >> lft.width >> lft.filename >> lft.type;
    layer_files.push_back(lft);
  }
  */
  //network_ifs.close();
  return layer_files.size() == num_layers? 0 : 1;
}

#define MNIST_VALS 784

/*
static std::ifstream g_ifs;

//TODO make these OCALLs
std::istream & get_input_stream(const char * streamname){
  if(streamname == NULL){
    return std::cin;
  }
  else{
    g_ifs = std::ifstream(streamname);
    return g_ifs;
  }
}

//Assumes the existence of a large enough buffer
int csv_getline(std::istream & ifs, float * vals, 
  char * label, size_t num_vals){
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
*/


//Assume all buffers are allocated
//Read in NORMALIZED values
//WARNING: before using, update to properly read in csv
/*
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
*/

int mask(float * input, float * masks, unsigned int input_size){
  for(unsigned int i = 0; i < input_size; i++){
    input[i] += masks[i];
  }
  return 0;
}

//TODO complete this
void unmask(float * data, int width, int height, float * mask_data, float * input_layer){
  return;
}






//Assumes a buffer is allocated
int read_all_weights(const vector<layer_file_t> & layers, float ** bufs){
  for(size_t i = 0; i < layers.size(); i++){
    bufs[i] = (float *) malloc(layers[i].height * layers[i].width * sizeof(float));
    //Should check return val
    size_t len = layers[i].filename.size();
    char * fname_buf = (char *) calloc(len+1, sizeof(char));
    strncat(fname_buf, layers[i].filename.c_str(), len);
#ifdef NENCLAVE
    if(read_weight_file_plain(fname_buf, layers[i].height * layers[i].width * sizeof(float), bufs[i])){
      return 1;
    }
#else
    int ocall_ret;
    sgx_status_t ocall_status;
    ocall_status = read_weight_file_plain(&ocall_ret, fname_buf, layers[i].height * layers[i].width * sizeof(float), bufs[i]);
    if(ocall_status || ocall_ret){
      return 1;
    }
    //Need data in non-const container for ocall
    //DO NOT REMOVE - used for byte-level file reading
    /*
    size_t len = layers[i].filename.size();
    char * fname_buf = (char *) malloc(len+1);
    strncat(fname_buf, layers[i].filename.c_str(), len);
    sgx_status_t ocall_status;
    int ocall_ret;
    ocall_status = read_weight_file(&ocall_ret, fname_buf, layers[i].height * layers[i].width, bufs[i]);
    free(fname_buf);
    if(ocall_ret){
      return 1;
    }
    */
#endif    
  free(fname_buf);  
  }
  return 0;
}

#ifdef NENCLAVE
#include <iostream>
using std::cout;
using std::endl;
#endif

//Debugging function
void print_layer_info(const vector<layer_file_t> & layers){
#ifdef NENCLAVE
  cout << "Layers info: " << endl;
  for(const auto & l : layers){
    cout << l.height << ' ' << l.width << ' ' << l.filename << ' ' << l.type << endl;
  }
#endif  
  return;
}

//For inputs: read in entire batches to enclave memory from ordinary file through OCALLs, and free() data when done.



void mask(float * data, int len, float * mask_data, bool do_mask=true){
  if(!do_mask){
    return;
  }
  for(int i = 0; i < len; i++){
    data[i] += mask_data[i];
  }
  return;
}





void backwards_demask(const float * input, const int input_width, const int input_height,
    const float * input_mask, const int input_mask_width, const int input_mask_height,
    const float * outputs, const int outputs_width, const int outputs_height,
    const float * weights, const int weights_width, const int weights_height,
    const float * weights_mask, const int weights_mask_width, const int weights_mask_height,
    const float * grad_output, const int grad_output_width, const int grad_output_height,
    const float * grad_mask, const int grad_mask_width, const int grad_mask_height,
    float ** d_ret, float ** e_ret){
  //Calculate weight_rand_mask - b
  float * diff3_diff = (float *) malloc(sizeof(float) * input_mask_width * weights_height);
  float * weights_transpose = transpose(weights, weights_width, weights_height);
  float * weights_mask_transpose = transpose(weights_mask, weights_mask_width, weights_mask_height);
  matrix_sub(weights_mask_transpose, weights_transpose, weights_width*weights_height, diff3_diff);
  int diff_w, diff_h;
  float * diff_tmp;
  matrix_multiply(input_mask, weights_width, weights_height,
   diff3_diff, input_mask_height, weights_width, //Swap width and height
   &diff_tmp, &diff_w, &diff_h, 0);

  float * diff2;
  matrix_multiply(input, input_width, input_height, 
    weights_mask_transpose, weights_mask_height, weights_mask_width, //Swap height and weights here
    &diff2, &diff_w, &diff_h, 0);
  matrix_sub(diff_tmp, diff2, diff_w * diff_h, diff_tmp);

  free(weights_mask_transpose);
  weights_mask_transpose = NULL;
  free(weights_transpose);
  weights_transpose = NULL;
  free(diff2);
  diff2 = NULL;
  free(diff3_diff);
  diff3_diff = NULL;

  //diff_tmp now holds diff3-diff-diff2
  float * grad_rand_mask_transformed = transform(grad_mask, diff_tmp, diff_w, diff_h);
  float * weight_mask_weights = (float *) malloc(sizeof(float) * grad_output_width * weights_height);
  matrix_sub(weights, weights_mask, grad_output_width * weights_height, weight_mask_weights);
  int d_diffb_w, d_diffb_h;
  float * d_diffb;
  matrix_multiply(grad_output, grad_output_width, grad_output_height,
    weight_mask_weights, grad_output_width, weights_height, 
    &d_diffb, &d_diffb_w, &d_diffb_h, 0);
  int diffc_diffa_w, diffc_diffa_h;
  float * diffc_diffa;
  matrix_multiply(grad_rand_mask_transformed, diff_w, diff_h,
    weight_mask_weights, grad_output_width, weights_height,
    &diffc_diffa, &diffc_diffa_w, &diffc_diffa_h, 1);
  matrix_add(diffc_diffa, d_diffb, d_diffb_w * d_diffb_h, diffc_diffa);
  free(d_diffb); //Don't free diffc_diffa

  float * transformed_transpose = transpose(grad_rand_mask_transformed, diff_w, diff_h);
  float * a_randmask = (float *) malloc(sizeof(float)*weights_height*input_width);
  //float * difff_diffg = (float *) malloc(sizeof(float)*diff_h*input_width);
  //a-rand_mask
  matrix_sub(input, input_mask, input_width*input_height, a_randmask);
  
  float * e_diffe;
  float * diffg_difff;
  //transpose of c times (a-rand_mask)
  int e_w, e_h;
  float * grad_output_transpose = transpose(grad_output, grad_output_width, grad_output_height);
  matrix_multiply(grad_output_transpose, grad_output_height, grad_output_width, //Switch width and height
    a_randmask, input_width, input_height,
    &e_diffe, &e_w, &e_h, 0);
  matrix_multiply(transformed_transpose, diff_h, diff_w, //Switch width and height
    a_randmask, input_width, input_height,
    &diffg_difff, &e_w, &e_h, 1);
  matrix_add(e_diffe, diffg_difff, e_w*e_h, e_diffe);

  //TODO set to null
  free(diff_tmp);
  free(grad_output_transpose);
  free(a_randmask);
  free(weight_mask_weights);
  free(diffg_difff);

  free(transformed_transpose);
  transformed_transpose = NULL;
  free(grad_rand_mask_transformed);
  grad_rand_mask_transformed = NULL;

  *d_ret = diffc_diffa;
  *e_ret = e_diffe;

  return;

}

void forward_demask(const float * input, const float * input_masks, 
  const float * weights, const float * weights_masks, 
  const int height, const int width, float ** result){

  int total_elts = height*width;
  float * tmp = (float *) malloc(sizeof(float)*height*width);
  matrix_sub(weights, weights_masks, total_elts, tmp);

  float * c_d2;
  int w_dummy, h_dummy;
  matrix_multiply(input, width, height, tmp, width, height, &c_d2, &w_dummy, &h_dummy, 0);

  float * d3_d;
  matrix_multiply(input_masks, width, height, tmp, width, height, &d3_d, &w_dummy, &h_dummy, 1);
  
  matrix_sub(c_d2, d3_d, total_elts, c_d2);
  activate(c_d2, height, width);
  matrix_add(c_d2, input_masks, total_elts, c_d2);

  *result = c_d2;

  free(tmp);
  free(d3_d);
}

void update_weights(float * weights, const float * weights_gradient, int total_elts, float learning_rate){
  for(int i = 0; i < total_elts; i++){
    weights[i] -= learning_rate*weights_gradient[i];
  }
  return;
}



//Need OCALLS for pipe I/O, setup, teardown
int enclave_main(char * network_structure_fname, char * input_csv_filename, 
  char * inpipe_fname, char * outpipe_fname, int verbose){

  unsigned int num_layers;
  vector<layer_file_t> layer_files;
  unsigned int num_inputs;
  int input_height; 
  int input_width; 

#ifndef NENCLAVE
  sgx_status_t ocall_status;
  int ocall_ret;
#endif  


#ifdef NENCLAVE
  int ocall_ret = init_streams(inpipe_fname, outpipe_fname);
#else
  ocall_status = init_streams(&ocall_ret, inpipe_fname, outpipe_fname);
  //TODO check result
#endif  
  if(ocall_ret){
    print_out("ERROR: could not initialize I/O streams", true);
    return -1;
  }
  if(verbose){
    print_out("Initialized I/O streams", false);
  }
  

  if(parse_structure(network_structure_fname, layer_files, 
    num_inputs, input_height, input_width)){
    print_out("Network parsing failed!", true);
    return 1;
  }
  if(verbose){
    print_out("Finished parsing network", false);
  }  
#ifdef NENCLAVE
  if(verbose >= 3){
    for(const layer_file_t & lft : layer_files){
      cout << lft.height << ' ' << lft.width << ' ' << lft.filename << endl;
    }
    cout << endl;
  }
#endif  
  
  num_layers = layer_files.size();

  float ** layer_data;
  layer_data = (float **) malloc(sizeof(float *) * num_layers);
  //Read in all layer data
  if(read_all_weights(layer_files, layer_data)){
    print_out("Failed to read weights", true);
    return 1;
  }
  if(verbose){
    print_out("Read in weights", false);
  }

  for(unsigned int input_idx = 0; input_idx < num_inputs; input_idx++){
#ifdef NENCLAVE
      if(verbose >= 2){
        cout << "Starting input " << input_idx << endl;
      }
#endif   

    float * input_data;
    
    //Check that the array's size is equal to the expected height*width
    for(unsigned int layer_idx = 0; layer_idx < num_layers; layer_idx++){

  #ifdef NENCLAVE
      if(verbose >= 2){
        cout << "Starting layer " << layer_idx << endl;
      }
#endif   
    
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
        if(verbose){
          print_out("Reading input from file...", false);
        }
#ifdef NENCLAVE        
        if(csv_getline(input_csv_filename, input_data, &data_label, data_height*data_width)){
          print_out("Failed to read input .csv", true);
          return 1;
        }
#else
        ocall_status = csv_getline(&ocall_ret, input_csv_filename, input_data, &data_label, data_height*data_width);
        if(ocall_ret){
          print_out("Failed to read input .csv", true);
          return 1;
        }
#endif        
        if(verbose){
          print_out("Read input from file", false);
        }
#ifdef NENCLAVE
        if(verbose >= 3){
          cout << "Input from file:\n";
          for(int i = 0; i < data_height*data_width; i++){
            cout << input_data[i] << ' ';
          }
          cout << endl;
        }
#endif        
      }
      else{
        data_height = layer_files[layer_idx].height;
        data_width = layer_files[layer_idx].width;
        if(verbose){
          print_out("Using previous layer's result as input", false);
        }
      }
      //If not first layer (i.e. raw input), then the input data is already initialized
      
      //Mask the current input
      //First, get the random mask
      float * mask_data = (float *) malloc(sizeof(float)*data_height*data_width);
      //Cast should be explicit, for the non-SGX version
      rand_bytes((unsigned char *) mask_data, sizeof(float)*data_height*data_width);
      //Next, mask the data
      mask(input_data, data_height*data_width, mask_data, false);
      if(verbose){
        print_out("Finished masking", false);
      }

      
      //Send it and a layer of weights to the GPU
      //Send first dimensions, then data (twice)
      int out_dims[2] = {data_height, data_width};
     
#ifdef NENCLAVE
      int written = write_stream((void *) out_dims, sizeof(out_dims));
      if(written){
        print_out("Failed writing input dimensions", true);
        cout << written << " bytes sent" << endl;        
        return 1;
      }
#else
      ocall_status = write_stream(&ocall_ret, (void *) out_dims, sizeof(out_dims));
      if(ocall_ret){
        print_out("Failed writing input dimensions", true);
        return 1;
      }
#endif      
      if(verbose){
        print_out("Sent input dimensions", false);   
      }    
     
#ifdef NENCLAVE  
      if(write_stream((void *) input_data, sizeof(float)*data_height*data_width)){
        print_out("Failed writing input", true);
        assert(input_data != NULL);
        if(verbose >= 3){
          for(int i = 0; i < data_height*data_width; i++){
            cout << input_data[i] << ' ';
          }
          cout << endl;
        }
        return 1;
      }
#else
      ocall_status = write_stream(&ocall_ret, (void *) input_data, sizeof(float)*data_height*data_width);
      if(ocall_ret){
        print_out("Failed writing input", true);
        return 1;
      }
#endif

      if(verbose){
        print_out("Sent input", false);
      }
#ifdef NENCLAVE
      if(verbose >= 3){
        cout << "Input:\n";
        for(int i = 0; i < data_height*data_width; i++){
          cout << input_data[i] << ' ';
        }
        cout << endl;
      }
#endif      
      


      int weight_dims[2] = {layer_files[layer_idx].height, layer_files[layer_idx].width};
#ifdef NENCLAVE      
      if(write_stream((void *) weight_dims, sizeof(weight_dims))){
        print_out("Failed writing weights dimensions", true);
        return 1;
      }
#else      
      ocall_status = write_stream(&ocall_ret, (void *) weight_dims, sizeof(weight_dims));
      if(ocall_ret){
        print_out("Failed writing weights dimensions", true);
        return 1;
      }
#endif      
      if(verbose){
        print_out("Sent weights dimensions", false);
      }
      
#ifdef NENCLAVE      
      if(write_stream((void *) layer_data[layer_idx], sizeof(float)*weight_dims[0]*weight_dims[1])){
        print_out("Failed writing weights", true);
        return 1;
      }
#else
      ocall_status = write_stream(&ocall_ret, (void *) layer_data[layer_idx], sizeof(float)*weight_dims[0]*weight_dims[1]);
      if(ocall_ret){
        print_out("Failed writing weights", true);
        return 1;
      }
#endif      
      if(verbose){
        print_out("Sent weights", false);
      }
#ifdef NENCLAVE
      if(verbose >= 3){
        for(int i = 0; i < weight_dims[0]*weight_dims[1]; i++){
          cout << layer_data[layer_idx][i] << ' ';
        }
        cout << endl;
      }
#endif   
      
      //Get back a result C ?= A*B
      //Read in dimensions, then data
      int in_dims[2] = {-1, -1};
      int next_height, next_width;
#ifdef NENCLAVE      
      if(read_stream((void *) in_dims, sizeof(in_dims))){
        print_out("Failed reading in result dimensions", true);
        return 1;
      }
#else
      ocall_status = read_stream(&ocall_ret, (void *) in_dims, sizeof(in_dims));
      if(ocall_ret){
        print_out("Failed reading in result dimensions", true);
        return 1;
      }
#endif      
      if(verbose){
        print_out("Read in result dimensions", false);
      }

      //TODO check that these are valid
      next_height = in_dims[0];
      next_width = in_dims[1];
      float * gpu_result = (float *) malloc(sizeof(float)*next_height*next_width);
#ifdef NENCLAVE
      if(read_stream((void *) gpu_result, sizeof(float)*next_height*next_width)){
        print_out("Failed reading in result", true);
        return 1;
      }
#else
      ocall_status = read_stream(&ocall_ret, (void *) gpu_result, sizeof(float)*next_height*next_width);
      if(ocall_ret){
        print_out("Failed reading in result", true);
        return 1;
      }
#endif      
      if(verbose){
        print_out("Read in result", false);
      }

      //Validate C through Frievalds' algorithm
      //If it fails, send {-1, -1} back to the GPU and exit
      if(frievald(input_data, layer_data[layer_idx], gpu_result, 
  data_height, data_width, data_height, data_width, next_height, next_width)){
        //Verification failed!
        int failed_resp[2] = {-1, -1};
#ifdef NENCLAVE      
        if(write_stream((void *) failed_resp, sizeof(failed_resp))){
          //Error
          print_out("Failed writing failure", true);
          return 1;
        }
#else
        ocall_status = write_stream(&ocall_ret, (void *) failed_resp, sizeof(failed_resp));
        if(ocall_ret){
          print_out("Failed writing failure", true);
          return 1;
        }
#endif        
        print_out("Frievalds' algorithm failed!", true);
        return 1;
      }
      else{
#ifdef NENCLAVE
        if(verbose >= 1){
          cout << "Frievalds' algorithm succeeded!" << endl;
        }
#endif        
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
      //TODO write layers back out
      
      //Setup things for the next iteration
      if(layer_idx != num_layers - 1){
        input_data = gpu_result;
      }
#ifdef NENCLAVE
      if(verbose >= 2){
        cout << "Finished layer " << layer_idx << endl;
      }
#endif      
    } //layer_idx
  } //input_idx

  
  //Cleanup layers
  for(size_t i = 0; i < layer_files.size(); i++){
    free(layer_data[i]);
  }
  free(layer_data);
  

  return 0;

}
