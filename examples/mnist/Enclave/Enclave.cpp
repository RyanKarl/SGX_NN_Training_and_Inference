//Enclave.cpp
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
  return;
}
#endif

#ifdef NENCLAVE
void print_floatarr(float * data, int size){
  for(int i = 0; i < size; i++){
    cout << data[i] << ' ';
  }
  cout << endl;
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
  matrix_multiply(a, a_width, a_height, 
    br, br_w, br_h,
    &axbr, &axbr_w, &axbr_h, 0);

  free(br);
  br = NULL;

  for (int i = 0; i < c_width; i++){
    //cout << "axbr[" << i << "] " << axbr[i] << " cr[" << i << "] " << cr[i] << endl;
    if (FLOAT_CMP(axbr[i], cr[i])){
    //DEBUG      
#ifdef NENCLAVE
      cout << "i: " << i << " axbr: " << axbr[i] << " cr: " << cr[i] << endl;
#endif      
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


int parse_structure(char * network_structure_fname, vector<layer_file_t> & layer_files, unsigned int & num_inputs, 
  unsigned int & num_pixels, unsigned int & batchsize){
  
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

  num_inputs = atoi(strtok(str_in, " \n"));
  num_pixels = atoi(strtok(NULL, " \n"));
  unsigned int num_layers = atoi(strtok(NULL, " \n"));
  batchsize = atoi(strtok(NULL, " \n"));
  layer_files.reserve(num_layers);

  for(unsigned int i = 0; i < num_layers; i++){
    layer_file_t lft;

    lft.neurons = atoi(strtok(NULL, " \n"));
    lft.filename = strtok(NULL, " \n");
    lft.type = atoi(strtok(NULL, " \n"));

    layer_files.push_back(lft);
  }
  return layer_files.size() == num_layers? 0 : 1;
}

int mask(float * input, const float * masks, int input_size, bool negate){
  if(negate){
    for(int i = 0; i < input_size; i++){
      input[i] -= masks[i];
    }

  }
  else{
    for(int i = 0; i < input_size; i++){
      input[i] += masks[i];
    }
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
    bufs[i] = (float *) malloc(layers[i].neurons * sizeof(float));
    //Should check return val
    size_t len = layers[i].filename.size();
    char * fname_buf = (char *) calloc(len+1, sizeof(char));
    strncat(fname_buf, layers[i].filename.c_str(), len);
#ifdef NENCLAVE
    if(read_weight_file_plain(fname_buf, layers[i].neurons * sizeof(float), bufs[i])){
      return 1;
    }
#else
    int ocall_ret;
    sgx_status_t ocall_status;
    ocall_status = read_weight_file_plain(&ocall_ret, fname_buf, layers[i].neurons * sizeof(float), bufs[i]);
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
    cout << l.neurons << ' ' << l.filename << ' ' << l.type << endl;
  }
#endif  
  return;
}

//For inputs: read in entire batches to enclave memory from ordinary file through OCALLs, and free() data when done.




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

  float * c_d2 = NULL;
  int w_dummy, h_dummy;
  //Transpose argument
  float * tmp_transposed = NULL;
  tmp_transposed = transpose(tmp, width, height);

  //DEBUG
  print_floatarr(tmp, height*width);
  print_floatarr(tmp_transposed, height*width);

  free(tmp);
  tmp = NULL;

  //Swap width and height
  matrix_multiply(input, width, height, tmp_transposed, height, width, &c_d2, &w_dummy, &h_dummy, 0);

  float * d3_d = NULL;
  matrix_multiply(input_masks, width, height, tmp_transposed, height, width, &d3_d, &w_dummy, &h_dummy, 1);
  
  matrix_sub(c_d2, d3_d, total_elts, c_d2);
  activate(c_d2, height, width);
  matrix_add(c_d2, input_masks, total_elts, c_d2);

  *result = c_d2;

  free(tmp_transposed);
  tmp_transposed = NULL;
  free(d3_d);
  d3_d = NULL;
}

void update_weights(float * weights, const float * weights_gradient, int total_elts, float learning_rate){
  for(int i = 0; i < total_elts; i++){
    weights[i] -= learning_rate*weights_gradient[i];
  }
  return;
}

void normalize(float * x, int total_elts){
  for(int i = 0; i < total_elts; i++){
    x[i] = atan(x[i]) * (2.0f/M_PI);
  }
  return;
}

int send_to_gpu(const float * data, const int batchsize, const int num_neurons, const int verbose){
  //Send it and a layer of weights to the GPU
  //Send first dimensions, then data (twice)
  int out_dims[2] = {batchsize, num_neurons};
 
#ifdef NENCLAVE
  int written = write_stream((void *) out_dims, sizeof(out_dims));
  if(written){
    print_out((char *) &("Failed writing input dimensions"[0]), true);
    cout << written << " bytes sent" << endl;        
    return 1;
  }
#else
  ocall_status = write_stream(&ocall_ret, (void *) out_dims, sizeof(out_dims));
  if(ocall_ret){
    print_out((char *) &("Failed writing input dimensions"[0]), true);
    return 1;
  }
#endif      
  if(verbose){
    print_out((char *) &("Sent input dimensions"[0]), false);   
  }    
 
#ifdef NENCLAVE  
  if(write_stream((void *) data, sizeof(float)*batchsize*num_neurons)){
    print_out((char *) &("Failed writing input"[0]), true);
    return 1;
  }
#else
  ocall_status = write_stream(&ocall_ret, (void *) data, sizeof(float)*batchsize*num_neurons);
  if(ocall_ret){
    print_out((char *) &("Failed writing input"[0]), true);
    return 1;
  }
#endif
  return 0;
}

int receive_from_gpu(float ** result, int * num_neurons, int * batchsize, const int verbose){
  //Get back a result C ?= A*B
  //Read in dimensions, then data
  int in_dims[2] = {0, 0};
#ifdef NENCLAVE      
  if(read_stream((void *) in_dims, sizeof(in_dims))){
    print_out((char *) &("Failed reading in result dimensions"[0]), true);
    return 1;
  }
#else
  ocall_status = read_stream(&ocall_ret, (void *) in_dims, sizeof(in_dims));
  if(ocall_ret){
    print_out((char *) &("Failed reading in result dimensions"[0]), true);
    return 1;
  }
#endif      
  if(verbose){
    print_out((char *) &("Read in result dimensions"[0]), false);
  }

  *batchsize = in_dims[0];
  *num_neurons = in_dims[1];
  

  *result = (float *) malloc(sizeof(float)*(*num_neurons)*(*batchsize));
#ifdef NENCLAVE
  if(read_stream((void *) *result, sizeof(float)*(*num_neurons)*(*batchsize))){
    print_out((char *) &("Failed reading result result"[0]), true);
    return 1;
  }
#else
  ocall_status = read_stream(&ocall_ret, (void *) *result, sizeof(float)*(*num_neurons)*(*batchsize));
  if(ocall_ret){
    print_out((char *) &("Failed reading in result"[0]), true);
    return 1;
  }
#endif      
  if(verbose){
    print_out((char *) &("Read in result"[0]), false);
  }
  return 0;
}


//Need OCALLS for pipe I/O, setup, teardown
int enclave_main(char * network_structure_fname, char * input_csv_filename, 
  char * inpipe_fname, char * outpipe_fname, char * weights_outfile, int verbose){

  unsigned int num_layers;
  vector<layer_file_t> layer_files;
  unsigned int num_inputs = 0;
  unsigned int batchsize = 0; //TODO initialize
  unsigned int num_pixels = 0;
  string weights_out_str = "";
  if(weights_outfile != NULL){
    weights_out_str = weights_outfile;
  }

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
    print_out((char *) &("ERROR: could not initialize I/O streams"[0]), true);
    return -1;
  }
  if(verbose){
    print_out((char *) &("Initialized I/O streams"[0]), false);
  }
  

  if(parse_structure(network_structure_fname, layer_files, 
    num_inputs, num_pixels, batchsize)){
    print_out((char *) &("Network parsing failed!"[0]), true);
    return 1;
  }
  if(verbose){
    print_out((char *) &("Finished parsing network"[0]), false);
  }  
#ifdef NENCLAVE
  if(verbose >= 3){
    for(const layer_file_t & lft : layer_files){
      cout << lft.neurons << ' ' << lft.filename << endl;
    }
    cout << endl;
  }
#endif  
  
  num_layers = layer_files.size();

  float ** layer_data;
  layer_data = (float **) malloc(sizeof(float *) * num_layers);
  //Read in all layer data
  if(read_all_weights(layer_files, layer_data)){
    print_out((char *) &("Failed to read weights"[0]), true);
    return 1;
  }
  if(verbose){
    print_out((char *) &("Read in weights"[0]), false);
  }

  unsigned num_batches = (num_inputs / batchsize) + ((num_inputs % batchsize) ? 1 : 0);

  float * input_data;

  for(unsigned int batch_idx = 0; batch_idx < num_batches; batch_idx++){
    //Get images into a matrix
    unsigned num_images_this_batch = (batch_idx == num_batches-1) ? (num_inputs / num_batches) : (num_inputs % num_batches);
    input_data = (float *) malloc(sizeof(float) * num_images_this_batch * num_pixels);
    float * image_data_csv_ptr = input_data;
    char * data_labels = (char *) malloc(sizeof(unsigned char) * num_inputs);
    for(unsigned int image_idx = 0; image_idx < num_images_this_batch; image_idx++){
#ifdef NENCLAVE        
      if(csv_getline(input_csv_filename, image_data_csv_ptr, data_labels, num_pixels)){
        print_out((char *) &("Failed to read input .csv"[0]), true);
        return 1;
      }
#else
      ocall_status = csv_getline(&ocall_ret, input_csv_filename, image_data_csv_ptr, &data_labels, num_pixels);
      if(ocall_ret){
        print_out((char *) &("Failed to read input .csv"[0]), true);
        return 1;
      }
#endif        
      if(verbose){
        print_out((char *) &("Read input from file"[0]), false);
      }
      image_data_csv_ptr += num_pixels; //Increment pointer
    }

    
    //Now we have the whole batch in a single array
    for(unsigned int layer_idx = 0; layer_idx < num_layers; layer_idx++){
      int num_neurons;
      num_neurons = layer_idx ? layer_files[layer_idx].neurons : num_pixels;

      //Mask the current input
      //First, get the random mask
      float * mask_data = (float *) malloc(sizeof(float)*num_neurons*num_images_this_batch);
      //Cast should be explicit, for the non-SGX version
      rand_bytes((unsigned char *) mask_data, sizeof(float)*num_neurons*num_images_this_batch);
      //Normalize mask
      normalize(mask_data, num_neurons*num_images_this_batch);
      //Next, mask the data
      mask(input_data, mask_data, num_neurons*num_images_this_batch, false);
      if(verbose){
        print_out((char *) &("Finished masking input"[0]), false);
      }
      //Send masked input to the GPU
      if(send_to_gpu(input_data, batchsize, num_neurons, verbose)){
        print_out((char *) &("Failed to send input data"[0]), true);
        return 1;
      }

      //Mask weights
      float * mask_weights = (float *) malloc(sizeof(float) * num_neurons);
      //Cast should be explicit, for the non-SGX version
      rand_bytes((unsigned char *) mask_weights, sizeof(float) * num_neurons);
      normalize(mask_weights, num_neurons);
      mask(layer_data[layer_idx], mask_weights, num_neurons, false);
      if(verbose){
        print_out((char *) &("Finished masking weights"[0]), false);
      }
      //Send weights to GPU
      if(send_to_gpu(layer_data[layer_idx], 1, num_neurons, verbose)){
        print_out((char *) &("Failed to send input data"[0]), true);
        return 1;
      }


      //Receive result back
      float * gpu_result;
      int num_result_neurons;
      int result_batchsize;
      if(receive_from_gpu(&gpu_result, &num_result_neurons, &result_batchsize, verbose)){
        print_out((char *) &("Failed to receive mult. result from GPU"[0]), true);
        return 1;
      }

#ifdef NENCLAVE
      if(verbose >= 2){
        cout << "Input: " << num_images_this_batch << ' ' << num_neurons << endl;
        cout << "Weights: " << num_neurons << ' ' << 1 << endl;
        cout << "GPU result: " << num_result_neurons <<  ' ' << result_batchsize << endl;
      }
#endif      

      //Validate result with Frievalds' algorithm
      //If it fails, send {-1, -1} back to the GPU and exit
      if(frievald(input_data, layer_data[layer_idx], gpu_result, 
  num_images_this_batch, num_neurons, num_neurons, 1, num_result_neurons, result_batchsize)){
        //Verification failed!
        int failed_resp[2] = {-1, -1};
#ifdef NENCLAVE      
        if(write_stream((void *) failed_resp, sizeof(failed_resp))){
          //Error
          print_out((char *) &("Failed writing failure"[0]), true);
          return 1;
        }
#else
        ocall_status = write_stream(&ocall_ret, (void *) failed_resp, sizeof(failed_resp));
        if(ocall_ret){
          print_out((char *) &("Failed writing failure"[0]), true);
          return 1;
        }
#endif        
        print_out((char *) &("Frievalds' algorithm failed!"[0]), true);
        return 1;
      }
      else{
#ifdef NENCLAVE
        if(verbose >= 1){
          cout << "Frievalds' algorithm succeeded!" << endl;
        }
#endif        
      }

      //Result has been verified
      //Unmask (forward) the GPU result
      //Recall that input and weights are currently masked
      float * gpu_unmasked_result = NULL;
      forward_demask(input_data, mask_data, 
        layer_data[layer_idx], mask_weights, 
        num_images_this_batch, num_neurons, &gpu_unmasked_result);

      //Undo masking
      //May later just have a seperate buffer for masked weights
      mask(layer_data[layer_idx], mask_weights, num_neurons, true);

      //Activate unmasked result
      activate(gpu_unmasked_result, num_images_this_batch, num_neurons);

      //Assign next iteration's input to be the unmasked GPU result
      free(input_data);
      if(layer_idx != num_layers-1){
        input_data = gpu_unmasked_result;
      }
      else{
        input_data = NULL;
      }
      free(mask_data);
      mask_data = NULL;
      free(mask_weights);
      mask_weights = NULL;

    } //layer_idx


    //TODO Compute loss

    //TODO Traverse layers backwards and do backprop


  } //batch_idx


  //Write weights back to file
  if(weights_out_str != ""){
    for(size_t i = 0; i < layer_files.size(); i++){
      string idx_str = std::to_string(i);
      string full_name = weights_out_str + idx_str;
#ifdef NENCLAVE    
      if(floats_to_csv((char *) &(full_name[0]), layer_files[i].neurons, layer_data[i])){
        cerr << "ERROR: could not write to " << full_name << endl;
        return 1;
      }
#else
      ocall_status = floats_to_csv(&ocall_ret, full_name, layer_files[i].neurons, layer_data[i]);
      if(ocall_ret){
        print_out((char *) &("Failed writing .csv out"[0]), true);
        return 1;
      }
#endif    
    }
  }
  

  //Cleanup layers
  for(size_t i = 0; i < layer_files.size(); i++){
    free(layer_data[i]);
  }
  free(layer_data);

  return 0;

}
