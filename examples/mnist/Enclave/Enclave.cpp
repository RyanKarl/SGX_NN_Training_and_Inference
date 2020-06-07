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
inline void rand_bytes(unsigned char * r, const size_t n_bytes){
  assert(r);
  for(size_t i = 0; i < n_bytes; i++){
    r[i] = (unsigned char) rand();
  }
  return;
}
#endif

void rand_buf_to_floats(float * buf, size_t num_floats){
  for(size_t i = 0; i < num_floats; i++){
    unsigned char * char_addr = (unsigned char *) &buf[i];
    buf[i] = ((float) (*char_addr) / (1 << CHAR_BIT));
    assert(!isnan(buf[i]));
    assert(buf[i] >= 0.0f);
    assert(buf[i] < 1.0f);
    //assert(!isnan(((float) (*char_addr) /(sizeof(unsigned char)*CHAR_BIT))));
  }
}

#ifdef NENCLAVE
void print_floatarr(const float * data, int size){
  for(int i = 0; i < size; i++){
    cout << data[i] << ' ';
  }
  cout << endl;
}
#endif

//0 is height, 1 is width
int frievald(const float * a, const float * b, const float * c, 
  const int a_width, const int a_height, 
  const int b_width, const int b_height,  
   const int c_width, const int c_height)
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

  /*
  cout << "r: \n";
  print_floatmat(r, b_width, 1);
  */

  float * br = NULL;
  int br_w, br_h;
  float * cr = NULL;
  int cr_w, cr_h;
  
  matrix_multiply(b, b_width, b_height, 
    r, 1, b_width,
    &br, &br_w, &br_h, 0, 1);
  assert(br_h == a_width);
  assert(br_w == 1);

  matrix_multiply(c, c_width, c_height,
    r, 1, c_width,
    &cr, &cr_w, &cr_h, 0, 1);
  assert(cr_h == c_height);
  assert(cr_w == 1);
  
  /*
  cout << "br: \n";
  print_floatmat(br, br_h, br_w);
  
  cout << "cr: \n";
  print_floatmat(cr, cr_h, cr_w);
  */

  free(r);
  r = NULL;

  float * axbr;
  int axbr_w, axbr_h;
  matrix_multiply(a, a_width, a_height, 
    br, br_w, br_h,
    &axbr, &axbr_w, &axbr_h, 0);

  free(br);
  br = NULL;
  /*
  cout << "axbr: \n";
  print_floatmat(axbr, axbr_h, axbr_w);
  */
  for (int i = 0; i < cr_h; i++){
    //cout << "axbr[" << i << "] " << axbr[i] << " cr[" << i << "] " << cr[i] << endl;
    if (FLOAT_CMP(axbr[i], cr[i])){    
        cout << "axbr " << axbr[i] << " cr " << cr[i] << " i " << i << endl;
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

int verify_frievald(const float * a, const float * b, const float * c,
    const int a_width, const int a_height, 
    const int b_width, const int b_height,
    const int c_width, const int c_height){
  for(unsigned int j = 0; j < K_PROBABILITY; j++){
  	if(frievald(a, b, c, 
        a_width, a_height, 
        b_width, b_height,
        c_width, c_height)){
      cout << "Frievalds' failed on pass " << j << endl;
  		return 1;
  	}
  }
  return 0;
}


int parse_structure(char * network_structure_fname, vector<layer_file_t> & layer_files, unsigned int & num_inputs, 
  unsigned int & num_pixels, unsigned int & batchsize, unsigned int & num_labels){
  
  size_t file_len;


#ifdef NENCLAVE
  file_len = file_size(network_structure_fname);
#else
  sgx_status_t ocall_status;
  ocall_status = file_size(&file_len, network_structure_fname);
  if(ocall_status){
    return 1;
  }
#endif  

  char * str_in = (char *) calloc(file_len+1, sizeof(char));


#ifdef NENCLAVE
  if(file_to_string(network_structure_fname, str_in, file_len)){
    return 1;
  }
#else
  int ocall_ret;
  ocall_status = file_to_string(&ocall_ret, network_structure_fname, str_in, file_len);
  if(ocall_ret){
    return 1;
  }
#endif  

  num_inputs = atoi(strtok(str_in, " \n"));
  batchsize = atoi(strtok(NULL, " \n"));
  num_pixels = atoi(strtok(NULL, " \n"));
  unsigned int num_layers = atoi(strtok(NULL, " \n"));
  num_labels = atoi(strtok(NULL, " \n"));
  
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
      assert(!isnan(input[i]));
      assert(!isnan(masks[i]));
      input[i] -= masks[i];
    }

  }
  else{
    for(int i = 0; i < input_size; i++){
      assert(!isnan(input[i]));
      assert(!isnan(masks[i]));
      input[i] += masks[i];
    }
  }
  return 0;
}

//Assumes a buffer is allocated
int read_all_weights(const vector<layer_file_t> & layers, float ** bufs, unsigned int num_pixels){
  for(size_t i = 0; i < layers.size(); i++){
    int num_floats = layers[i].neurons * (i? layers[i-1].neurons : num_pixels);

    bufs[i] = (float *) malloc(num_floats * sizeof(float));
    //Should check return val
    size_t len = layers[i].filename.size();
    char * fname_buf = (char *) calloc(len+1, sizeof(char));
    strncat(fname_buf, layers[i].filename.c_str(), len);
#ifdef NENCLAVE
    if(read_weight_file_plain(fname_buf, num_floats * sizeof(float), bufs[i])){
      return 1;
    }
#else
    int ocall_ret;
    sgx_status_t ocall_status;
    ocall_status = read_weight_file_plain(&ocall_ret, fname_buf, num_floats * sizeof(float), bufs[i]);
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

void backwards_demask_lastlayer(const float * input, const int input_width, const int input_height,
    //const float * input_masks,
    const float * final_data, const int final_data_width, const int final_data_height,
    const float * weights, const int weights_width, const int weights_height,
    //const float * weights_masks,
    const float * grad_output, const int grad_output_width, const int grad_output_height,
    float ** d_ret
    //int * re_w, int * ret_w
    ){
  //final_data is softmax(a*b.t)
  //take the derivative of that term
  //TODO only need one small buffer for soft_der at a time
  float * grad_output_transposed = transpose(grad_output, grad_output_width, grad_output_height);
  float * soft_der = (float *) malloc(sizeof(float) * input_height * input_width * input_width);
  float * c_prod = (float *) malloc(sizeof(float) * input_width * input_height);
  float * c_prod_ptr = c_prod;
  int prod_w, prod_h;
  for(int i = 0; i < input_height; i++){
    //TODO softmax_derivative allocates memory
    softmax_derivative(input + (i*input_width), input_width,
     soft_der + (i*input_width*input_width)); 
    //Get the right part of the buffer to write to
    c_prod_ptr = c_prod + (i*input_width);
    //Now multiply
    matrix_multiply(grad_output_transposed + (i*grad_output_width), 1, grad_output_height,
    soft_der, input_width, input_width, //Same args for w and h intentional
    (float **) &c_prod_ptr, &prod_w, &prod_h, 0, 0);
    assert(prod_w == input_width);
    assert(prod_h == input_height);
  }

  free(grad_output_transposed);
  grad_output_transposed = NULL;
  free(soft_der);
  soft_der = NULL;

  //*d_ret = (float *) malloc(sizeof(float *) * input_height * weights_height);
  
  //Transpose b
  float * b_transpose = transpose(weights, weights_width, weights_height);

  int d_w, d_h;
  //Allocates new memory pointed to by *d_ret
  matrix_multiply(c_prod, input_width, input_height,
    b_transpose, weights_height, weights_width,
    d_ret, &d_w, &d_h, 0);

  free(b_transpose);
  b_transpose = NULL;
  free(c_prod);
  c_prod = NULL;
}


void backwards_demask_ordinary(const float * input, const int input_width, const int inp     ut_height,const float * input_mask, const float * outputs, const int outputs_width, const int outputs_height, const float * weights, const int weights_width, const int weights_height, const float * weights_mask, const float * grad_output, const int grad_output_width, const int grad_output_height, const float * grad_mask, float ** d_ret, float ** e_ret){

    float * a_mul_wmask = (float *) malloc(sizeof(float) * input_height * weights_width); 
    int a_mul_wmask_height;
    int a_mul_wmask_width;
    
    float * weight_rand_mask_transpose = (float *) malloc(sizeof(float) * weight_height * weights_width)     ; 
    float * weight_transpose = (float *) malloc(sizeof(float) * weight_height * weights_width);
    float * weight_mask_transpose_weights_transpose = (float *) malloc(sizeof(float) * weight_height * weights_     width);
    float * weight_mask_transpose_weights_transpose_a_prod = (float *) malloc(sizeof(float) * w     eight_height * weights_     width);

    float ** diff_term

    matrix_multiply(input, input_width, input_height, weights_mask, weights_width, weights_height, &a_mul_wmask, a_mul_wmask_width, a_mul_wmask_height, 0, 1);
    
      weight_rand_mask_transpose = transpose(weight_ran_mask, weights_width, weights_height);
      weight_transpose = transpose(weights, weights_width, weights_height); 

      matrix_sub(weight_rand_mask_transpose, weights_transpose, weights_height * weights_width, weight_mask_transpose_weights_transpose)

      matrix_sub(weight_rand_mask_transpose, a_mul_wmask, weights_height * weights_width, weight_mask_transpose_weights_transpose_a_prod)

    matrix_multiply(weight_mask_transpose_weights_transpose_a_prod, weights_width, weights_height, input_mask, input_width, input_height, diff_term, iput_width, weights_height, 0, 1)

    transform_and_mult(grad_output, output, diff_term, grad_output_height * output_width);


    float **grm_tramsformed;
    grm_transformed = transform(grad_rand_mask, diff_term, grad_output_height * output_width);


    float * weightmask_b = (float *) malloc(sizeof(float) * weight_height * weights_width);
    
    matrix_sub(weight_rand_mask, weights, weight_height*weight_width, weightmask_b);


    float ** diffc_diffa;

    matrix_multiply(grm_transformed, grad_output_height, output_width, weightmask_b, weight_width, weight_height, diffc_diffa, grad_output_height, weight_width, 0, 1);
      
    float ** diffb;

    diffb = matrix_multiply(grad_output, grad_output_height, grad_output_width, weight_rand_mask, weight_width, weight_height, diffb, grad_output_height, weight_width, 0, 1);
      
    
    float * difftemp = (float *) malloc(sizeof(float) * grad_output_height*weight_width);
    float * d = (float *) malloc(sizeof(float) * grad_output_height*weight_width)     ;
    matrix_sub(diffc_diffa, diffb, grad_output_height*weight_width, difftemp);
    
    matrix_add(d, difftemp, grad_output_height*weight_width, d);
 
    float ** c_transpose;
//float * c_transpose = (float *) malloc(sizeof(float) * output_height*output_width)     ; 

    float **diffx;
  
    c_transpose = transpose(output, output_width, output_height);

    matrix_multiply(c_transpose, output_height, output_width, input_mask, input_mask_width, input_mask_height, diffx, input_mask_width, output_height, 0, 1)

    grm_transformed_transposed = transpose(grm_transformed_transposed, grad_output_height, output_width);

    float ** rand_mask_a = (float *) malloc(sizeof(float) * input_mask_height*a_width);
    matrix_add(rand_mask, a, input_mask_height*a, rand_mask_a);


    float ** diffz_diffy;
    matrix_multiply(grm_transformed_transposed, grad_output_height, output_width, rand_mask_a, input_height, input_width, diffz_diffy, grad_output_height, input_width, 0, 1);

    float ** e = (float *) malloc(sizeof(float) * grad_output_height*input_width);
    matrix_sub(diffz_diffy, diffx, grad_output_height*input_width, diffz_diffy);
    matrix_add(e, diffz_diffy, input_mask_height*a, e);

    //e += diffz_diffy - diffx 



}

/*
void backwards_demask_ordinary(const float * input, const int input_width, const int input_height,
    const float * input_mask, 
    const float * outputs, const int outputs_width, const int outputs_height,
    const float * weights, const int weights_width, const int weights_height,
    const float * weights_mask, 
    const float * grad_output, const int grad_output_width, const int grad_output_height,
    const float * grad_mask, 
    float ** d_ret, float ** e_ret){
  //Calculate weight_rand_mask - b
  float * diff3_diff = (float *) malloc(sizeof(float) * input_width * weights_height);
  float * weights_transpose = transpose(weights, weights_width, weights_height);
  float * weights_mask_transpose = transpose(weights_mask, weights_width, weights_height);
  matrix_sub(weights_mask_transpose, weights_transpose, weights_width*weights_height, diff3_diff);
  int diff_w, diff_h;
  float * diff_tmp;
  matrix_multiply(input_mask, weights_width, weights_height,
   diff3_diff, input_height, weights_width, //Swap width and height
   &diff_tmp, &diff_w, &diff_h, 0);

  float * diff2;
  matrix_multiply(input, input_width, input_height, 
    weights_mask_transpose, weights_height, weights_width, //Swap height and weights here
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
  float * grad_rand_mask_transformed = transform(grad_mask, diff_tmp, diff_w, diff_h, 0); //Not the last layer, so don't use softmax
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

  free(diff_tmp);
  diff_tmp = NULL;
  free(grad_output_transpose);
  grad_output_transpose = NULL;
  free(a_randmask);
  a_randmask = NULL;
  free(weight_mask_weights);
  weight_mask_weights = NULL;
  free(diffg_difff);
  diffg_difff = NULL;
  free(transformed_transpose);
  transformed_transpose = NULL;
  free(grad_rand_mask_transformed);
  grad_rand_mask_transformed = NULL;

  *d_ret = diffc_diffa;
  *e_ret = e_diffe;

  return;

}
*/
void forward_demask(const float * input, const int input_width, const int input_height,
  const float * input_masks, 
  const float * weights, const int weights_width, const int weights_height,
  const float * weights_masks,
  float ** result, int * result_width, int * result_height){

  assert(input_width == weights_height);

  int weights_elts = weights_height*weights_width;
  float * tmp = (float *) malloc(sizeof(float)*weights_elts);
  matrix_sub(weights, weights_masks, weights_elts, tmp);

  float * c_d2 = NULL;
  int w_dummy, h_dummy;
  //Transpose argument
  /*
  float * tmp_transposed = NULL;
  tmp_transposed = transpose(tmp, weights_width, weights_height);
#ifdef NENCLAVE
  //DEBUG  
  print_floatarr(tmp_transposed, weights_elts);
#endif
*/
  //DEBUG
  /*
  print_floatarr(tmp, height*width);
  print_floatarr(tmp_transposed, height*width);
  */

  

  //Swap width and height for tmp, as it's been transposed
  matrix_multiply(input, input_width, input_height, 
    tmp, weights_width, weights_height,
    &c_d2, &w_dummy, &h_dummy, 0);

  float * d3_d = NULL;
  matrix_multiply(input_masks, input_width, input_height,
   tmp, weights_width, weights_height,
   &d3_d, &w_dummy, &h_dummy, 1);
  
  matrix_sub(c_d2, d3_d, w_dummy*h_dummy, c_d2);
  activate(c_d2, h_dummy, w_dummy);
  matrix_add(c_d2, input_masks, w_dummy*h_dummy, c_d2);

  *result = c_d2;
  //TODO optimize later
  *result_width = w_dummy;
  *result_height = h_dummy;

  free(tmp);
  tmp = NULL;

  free(d3_d);
  d3_d = NULL;

/*
  free(tmp_transposed);
  tmp_transposed = NULL;
  */
  free(d3_d);
  d3_d = NULL;
}

//Actual is ground truth - 0 or 1 for each label
float crossentropy_loss(const unsigned int * actual, const float * predicted,
 const int num_possible_labels, const int batchsize){
  float sum = 0.0f;
  for(int i = 0; i < batchsize; i++){
    assert((int)actual[i] < num_possible_labels);
    sum -= log(predicted[(i*num_possible_labels)+actual[i]]);
  }
  return sum/batchsize;
}

//Allocates memory
float * crossentropy_derivative(const unsigned int * actual, const float * predicted,
 const int num_possible_labels, const int batchsize){
  float * ret = (float *) calloc(batchsize*num_possible_labels, sizeof(float));
  for(int i = 0; i < batchsize; i++){
    int idx = (i*num_possible_labels)+actual[i];
    ret[idx] = (-1.0f/predicted[idx])/batchsize;
  }
  return ret;
}





void update_weights(float * weights, const float * weights_gradient, const int total_elts, const float learning_rate){
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
    print_out((char *) &("Failed writing matrix dimensions"[0]), true);
    cout << written << " bytes sent" << endl;        
    return 1;
  }
#else
  sgx_status_t ocall_status;
  int ocall_ret;

  ocall_status = write_stream(&ocall_ret, (void *) out_dims, sizeof(out_dims));
  if(ocall_ret){
    print_out((char *) &("Failed writing matrix dimensions"[0]), true);
    return 1;
  }
#endif      
  if(verbose){
    print_out((char *) &("Sent matrix dimensions"[0]), false);   
  }    
 
#ifdef NENCLAVE  
  if(write_stream((void *) data, sizeof(float)*batchsize*num_neurons)){
    print_out((char *) &("Failed writing matrix"[0]), true);
    return 1;
  }
#else
  ocall_status = write_stream(&ocall_ret, (void *) data, sizeof(float)*batchsize*num_neurons);
  if(ocall_ret){
    print_out((char *) &("Failed writing matrix"[0]), true);
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
  sgx_status_t ocall_status;
  int ocall_ret;

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

//NB weights are a COLUMN vector


//Need OCALLS for pipe I/O, setup, teardown
int enclave_main(char * network_structure_fname, char * input_csv_filename, 
  char * inpipe_fname, char * outpipe_fname, char * weights_outfile, int backprop, int verbose){

  unsigned int num_layers;
  vector<layer_file_t> layer_files;
  unsigned int num_inputs = 0;
  unsigned int batchsize = 0; 
  unsigned int num_pixels = 0;
  unsigned int num_possible_labels = 0;
  bool skip_masking = false;
  string weights_out_str = "";
  if(weights_outfile != NULL){
    weights_out_str = weights_outfile;
  }

#ifndef NENCLAVE
  sgx_status_t ocall_status;
#endif  
  int ocall_ret;


#ifdef NENCLAVE
  ocall_ret = init_streams(inpipe_fname, outpipe_fname);
#else
  ocall_status = init_streams(&ocall_ret, inpipe_fname, outpipe_fname);
#endif  
  if(ocall_ret){
    print_out((char *) &("ERROR: could not initialize I/O streams"[0]), true);
    return -1;
  }
  if(verbose){
    print_out((char *) &("Initialized I/O streams"[0]), false);
  }
  

  if(parse_structure(network_structure_fname, layer_files, 
    num_inputs, num_pixels, batchsize, num_possible_labels)){
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
  if(read_all_weights(layer_files, layer_data, num_pixels)){
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
    unsigned int * data_labels = (unsigned int *) malloc(sizeof(unsigned int) * num_inputs);
    unsigned int * data_labels_ptr = data_labels;

    float * final_data = NULL;
    

    for(unsigned int image_idx = 0; image_idx < num_images_this_batch; image_idx++){
#ifdef NENCLAVE        
      if(csv_getline(input_csv_filename, image_data_csv_ptr, data_labels_ptr, num_pixels)){
        print_out((char *) &("Failed to read input .csv"[0]), true);
        return 1;
      }
#else
      ocall_status = csv_getline(&ocall_ret, input_csv_filename, image_data_csv_ptr, data_labels_ptr, num_pixels);
      if(ocall_ret){
        print_out((char *) &("Failed to read input .csv"[0]), true);
        return 1;
      }
#endif        
      if(verbose){
        print_out((char *) &("Read input from file"[0]), false);
      }
      image_data_csv_ptr += num_pixels; //Increment pointer
      data_labels_ptr++;
    }

    
    //Now we have the whole batch in a single array
    for(unsigned int layer_idx = 0; layer_idx < num_layers; layer_idx++){
      int num_neurons;
      num_neurons = layer_idx ? layer_files[layer_idx-1].neurons : num_pixels;

      //Mask the current input
      //First, get the random mask
      float * mask_data = (float *) malloc(sizeof(float)*num_neurons*num_images_this_batch);
      //Cast should be explicit, for the non-SGX version
      rand_bytes((unsigned char *) mask_data, sizeof(float)*num_neurons*num_images_this_batch);
      //Normalize mask
      //normalize(mask_data, num_neurons*num_images_this_batch);
      rand_buf_to_floats(mask_data, num_neurons*num_images_this_batch);
      //Next, mask the data
      if(!skip_masking){
        mask(input_data, mask_data, num_neurons*num_images_this_batch, false);
      }
      if(verbose){
        print_out((char *) &("Finished masking input"[0]), false);
      }    
#ifdef NENCLAVE
      if(verbose >= 2){
        int n_idx = nan_idx(input_data, num_images_this_batch*num_pixels);
        if(n_idx != -1){
          cout << "NaN found at " << n_idx << " of input_data, value is " << input_data[n_idx] << endl;
        }
      }
#endif      
      //Send masked input to the GPU
      if(send_to_gpu(input_data, num_images_this_batch, num_neurons, verbose)){
        print_out((char *) &("Failed to send input data"[0]), true);
        return 1;
      }

      //Mask weights
      int num_weights = num_neurons * layer_files[layer_idx].neurons;
      float * mask_weights = (float *) malloc(sizeof(float) * num_weights);
      //Cast should be explicit, for the non-SGX version
      rand_bytes((unsigned char *) mask_weights, sizeof(float) * num_weights);
      rand_buf_to_floats(mask_weights, num_weights);
      //normalize(mask_weights, num_neurons);
      if(!skip_masking){
        mask(layer_data[layer_idx], mask_weights, num_weights, false);
        
      }
      if(verbose){
        print_out((char *) &("Finished masking weights"[0]), false);
      }

      //Send weights to GPU
      if(send_to_gpu(layer_data[layer_idx], num_neurons, layer_files[layer_idx].neurons, verbose)){
        print_out((char *) &("Failed to send weights data"[0]), true);
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
        cout << "Input: " << num_neurons << ' ' << num_images_this_batch << endl;
        cout << "Weights: " << ' ' << layer_files[layer_idx].neurons << ' ' << num_neurons << endl;
        cout << "GPU result: " << num_result_neurons << ' ' << result_batchsize << endl;
      }
#endif      

      //DEBUG FRIEVALD
/*      
#ifdef NENCLAVE
      //Test multiplication
      float * enc_mul_test = NULL;
      int test_w, test_h;
      matrix_multiply(input_data, num_neurons, num_images_this_batch,
        layer_data[layer_idx], layer_files[layer_idx].neurons, num_neurons,
        &enc_mul_test, &test_w, &test_h, 0, 1);
      assert(test_w == num_result_neurons);
      assert(test_h == result_batchsize);

      for(int i = 0; i < test_w * test_h; i++){
        assert(!FLOAT_CMP(enc_mul_test[i], gpu_result[i]));
      }

      //Now test Frievalds'
      if(verify_frievald(input_data, layer_data[layer_idx], enc_mul_test, 
          num_neurons, num_images_this_batch,
          layer_files[layer_idx].neurons, num_neurons,
          num_result_neurons, result_batchsize)){
        assert("Frievalds' algorithm failed on internally-calculated product" && 0);
      }  
      free(enc_mul_test);
      enc_mul_test = NULL;
#endif      
*/

      //Validate result with Frievalds' algorithm
      //If it fails, send {-1, -1} back to the GPU and exit
      if(verify_frievald(input_data, layer_data[layer_idx], gpu_result, 
          num_neurons, num_images_this_batch,
          layer_files[layer_idx].neurons, num_neurons,
          num_result_neurons, result_batchsize)){
        //Verification failed!
/*
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
*/
        print_out((char *) &("Frievalds' algorithm failed!"[0]), true);
        //return 1;
      }
      else{

        if(verbose >= 2){
#ifdef NENCLAVE          
          cout << "Frievalds' algorithm succeeded!" << endl;
#else
          print_out("Frievalds' algorithm succeeded!", false);          
#endif              
        }
    
      }

      //Result has been verified
      //Unmask (forward) the GPU result
      //Recall that input and weights are currently masked
      float * gpu_unmasked_result = NULL;
      int gpu_unmasked_w, gpu_unmasked_h;
      if(!skip_masking){
        forward_demask(input_data, num_neurons, num_images_this_batch,
        mask_data, 
        layer_data[layer_idx], layer_files[layer_idx].neurons, num_neurons,
        mask_weights, 
        &gpu_unmasked_result, &gpu_unmasked_w, &gpu_unmasked_h);
        //Undo masking
        //May later just have a seperate buffer for masked weights
        mask(layer_data[layer_idx], mask_weights, num_neurons, true);
      }
      else{
        gpu_unmasked_result = gpu_result;
         gpu_unmasked_h = result_batchsize;
         gpu_unmasked_w = layer_files[layer_idx].neurons;
      }
      



      //Activate unmasked result
      if(layer_idx != num_layers-1){
        activate(gpu_unmasked_result, gpu_unmasked_h, gpu_unmasked_w);
      }
      else{
        //Softmax
        softmax(gpu_unmasked_result, gpu_unmasked_h*gpu_unmasked_w);
      }

      //Assign next iteration's input to be the unmasked GPU result
      if(layer_idx != num_layers-1){
        if(input_data != NULL){
          free(input_data);
          input_data = NULL;
        }
        input_data = gpu_unmasked_result;
      }
      else{
        final_data = gpu_unmasked_result;
        //Check sizes
        assert(gpu_unmasked_h == (int) num_images_this_batch);
        assert(gpu_unmasked_w == (int) num_possible_labels);
        if(input_data != NULL){
          free(input_data);
          input_data = NULL;
        }
      }
      free(mask_data);
      mask_data = NULL;
      free(mask_weights);
      mask_weights = NULL;

    } //layer_idx


    if(backprop){

      if(verbose >= 2){
#ifdef NENCLAVE
        cout << "Starting backpropagation...\n";
#endif        
      }

        //Loss for the whole batch
      float batch_loss = crossentropy_loss(data_labels, final_data, num_possible_labels, num_images_this_batch);
      float * derivative = crossentropy_derivative(data_labels, final_data, num_possible_labels, num_images_this_batch);

      //Print output
      if(verbose >= 1){
        std::string loss_str = "Loss this batch: " + std::to_string(batch_loss);
        print_out((char *) &(loss_str.c_str()[0]), false);
      }

      for(int rev_layer_idx = num_layers-1; rev_layer_idx >= 0; rev_layer_idx--){

        if(rev_layer_idx == (int)num_layers-1){
          float * d_ret = NULL;
          backwards_demask_lastlayer(input_data, num_images_this_batch, layer_files[rev_layer_idx-1].neurons,
            final_data, num_images_this_batch, layer_files[rev_layer_idx].neurons,
            layer_data[rev_layer_idx], layer_files[rev_layer_idx-1].neurons, layer_files[rev_layer_idx].neurons,
            derivative, num_possible_labels, num_images_this_batch,
            &d_ret);


          free(input_data);
          input_data = NULL;

          free(derivative);
          derivative = NULL;
          derivative = d_ret;

          continue;
        }

        //Mask matrices to send to GPU
        float * rev_mask_input = (float *) malloc(sizeof(float)*num_images_this_batch*layer_files[rev_layer_idx].neurons);
        float * rev_mask_weights = (float *) malloc(sizeof(float)*layer_files[rev_layer_idx].neurons);
        float * deriv_mask = (float *) malloc(sizeof(float)*num_images_this_batch*layer_files[rev_layer_idx].neurons);
        rand_bytes((unsigned char *) rev_mask_input, sizeof(float)*num_images_this_batch*layer_files[rev_layer_idx].neurons);
        rand_bytes((unsigned char *) rev_mask_weights, sizeof(float)*layer_files[rev_layer_idx].neurons);
        rand_bytes((unsigned char *) deriv_mask, sizeof(float)*num_images_this_batch*layer_files[rev_layer_idx].neurons);
        //final_data now used for the next level
        mask(final_data, rev_mask_input, num_images_this_batch*layer_files[rev_layer_idx].neurons, 0);
        mask(layer_data[rev_layer_idx], rev_mask_weights, layer_files[rev_layer_idx-1].neurons*layer_files[rev_layer_idx].neurons, 0);
        mask(derivative, deriv_mask, num_images_this_batch*layer_files[rev_layer_idx].neurons, 0);
        //Send out to GPU
        if(send_to_gpu(final_data, num_images_this_batch, layer_files[rev_layer_idx].neurons, verbose)){
          print_out((char *) &("Failed to send input data (backprop)"[0]), true);
          return 1;
        }
        if(send_to_gpu(layer_data[rev_layer_idx], 1, layer_files[rev_layer_idx].neurons, verbose)){
          print_out((char *) &("Failed to send weights (backprop)"[0]), true);
          return 1;
        }
        if(send_to_gpu(derivative, num_images_this_batch, layer_files[rev_layer_idx].neurons, verbose)){
          print_out((char *) &("Failed to send derivative (backprop)"[0]), true);
          return 1;
        }

        //Receive back
        float * gpu_derivative = NULL;
        int deriv_neurons, deriv_batchsize;
        if(receive_from_gpu(&gpu_derivative, &deriv_neurons, &deriv_batchsize, verbose)){
          print_out((char *) &("Failed to receive backprop. result from GPU"[0]), true);
          return 1;
        }

        float * gpu_weights_update = NULL;
        int update_neurons, update_batchsize;
        if(receive_from_gpu(&gpu_weights_update, &update_neurons, &update_batchsize, verbose)){
          print_out((char *) &("Failed to receive weights update from GPU"[0]), true);
          return 1;
        }
        assert(update_neurons == (int) layer_files[rev_layer_idx].neurons);
        assert(update_batchsize == 1);
        assert(deriv_neurons == (int) layer_files[rev_layer_idx].neurons);
        assert(deriv_batchsize == (int) num_images_this_batch);

        //Also read back this layer's feed-forward output from the GPU for convenience
        //This is a tradeoff of communication vs. memory
        float * layer_forward_data;
        int fwd_w, fwd_h;
        if(receive_from_gpu(&layer_forward_data, &fwd_w, &fwd_h, verbose)){
          print_out((char *) &("Failed to receive feed-forward data from GPU"[0]), true);
          return 1;
        }

        float * output_data;
        int output_w, output_h;
        if(receive_from_gpu(&output_data, &output_w, &output_h, verbose)){
          print_out((char *) &("Failed to receive output data from GPU"[0]), true);
          return 1;
        }
/*
        float * weights_masked;
        int weights_masked_w, weights_masked_h;
        if(receive_from_gpu(&weights_masked, &weights_masked_w, &weights_masked_h, verbose)){
          print_out((char *) &("Failed to receive weights mask data from GPU"[0]), true);
          return 1;
        }        
        assert(weights_masked_w == (rev_layer_idx? layer_files[rev_layer_idx-1].neurons : num_pixels));
        assert(weights_masked_h == (int) layer_files[rev_layer_idx].neurons);
*/
        float * grad_mask;
        int grad_mask_w, grad_mask_h;
        if(receive_from_gpu(&grad_mask, &grad_mask_w, &grad_mask_h, verbose)){
          print_out((char *) &("Failed to receive gradient mask data from GPU"[0]), true);
          return 1;
        }

        


        //Verify with Frievalds' algorithm
        //Need to verify 2 multiplications
        if(verify_frievald(final_data, layer_data[rev_layer_idx], gpu_derivative, 
          num_images_this_batch, layer_files[rev_layer_idx].neurons, layer_files[rev_layer_idx].neurons, 1, num_images_this_batch, 1)){
          print_out((char *) &("Frievalds' algorithm failed on prod. of final_data and layer_data"), true);
          //return 1;
        }
        if(verify_frievald(final_data, derivative, gpu_weights_update, 
          num_images_this_batch, layer_files[rev_layer_idx].neurons, layer_files[rev_layer_idx].neurons, 1, num_images_this_batch, num_images_this_batch)){
          print_out((char *) &("Frievalds' algorithm failed on prod. of final_data and derivative"), true);
          //return 1;
        }

        float * e_weights = NULL;
        float * d_derivative = NULL;

        //Unmask weights before demasking
        mask(layer_data[rev_layer_idx], rev_mask_weights, 
          layer_files[rev_layer_idx+1].neurons*layer_files[rev_layer_idx].neurons, 1);
        //Figure out args for backwards demasking 
        backwards_demask_ordinary(layer_forward_data, fwd_w, fwd_h,
          rev_mask_input, 
          output_data, output_w, output_h,
          layer_data[rev_layer_idx], layer_files[rev_layer_idx+1].neurons, layer_files[rev_layer_idx].neurons,
          rev_mask_weights, 
          derivative, output_w, output_h,
          grad_mask, 
          &d_derivative, &e_weights);

        update_weights(layer_data[rev_layer_idx], e_weights, 
          layer_files[rev_layer_idx+1].neurons*layer_files[rev_layer_idx].neurons, LEARNING_RATE);

        //Get derivative ready for next round
        free(derivative);
        derivative = NULL;
        derivative = d_derivative;
        d_derivative = NULL;


        free(layer_forward_data);
        layer_forward_data = NULL;
        free(output_data);
        output_data = NULL;
        /*
        free(weights_masked);
        weights_masked = NULL;
        */
        free(grad_mask);
        grad_mask = NULL;


        free(e_weights);
        e_weights = NULL;        


        free(rev_mask_input);
        rev_mask_input = NULL;
        free(rev_mask_weights);
        rev_mask_weights = NULL;
        free(deriv_mask);
        deriv_mask = NULL;
        free(gpu_derivative);
        gpu_derivative = NULL;
        free(gpu_weights_update);
        gpu_weights_update = NULL;
      }

      //Free up labels
      free(data_labels);
      data_labels = NULL;

      free(derivative);
      derivative = NULL;


    } //rev_layer_idx


    
  } //batch_idx


  //Write weights back to file
  if((weights_out_str != "") && backprop){
    for(size_t i = 0; i < layer_files.size(); i++){
      string idx_str = std::to_string(i);
      string full_name = weights_out_str + idx_str;
#ifdef NENCLAVE    
      if(floats_to_csv((char *) &(full_name[0]), layer_files[i].neurons, layer_data[i])){
        cerr << "ERROR: could not write to " << full_name << endl;
        return 1;
      }
#else
      ocall_status = floats_to_csv(&ocall_ret, (char *) &(full_name.c_str()[0]), layer_files[i].neurons, layer_data[i]);
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

//TODO consistent verbosity levels
//0 no output
//1 timing data
//2 logging
//3 data
