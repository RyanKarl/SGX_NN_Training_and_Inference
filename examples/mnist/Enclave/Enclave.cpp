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

//TODO fix this to be between 0 and 1 quantized
#define RAND_BUF_T unsigned int
void rand_floats(FP_TYPE * buf, size_t num_floats, unsigned int second_scale=1){
  RAND_BUF_T * tmp_buf = (RAND_BUF_T *) malloc(sizeof(RAND_BUF_T) * num_floats);
  rand_bytes((unsigned char *) tmp_buf, sizeof(RAND_BUF_T) * num_floats);
  for(size_t i = 0; i < num_floats; i++){
    /*
    buf[i] = (FP_TYPE) tmp_buf[i];
    buf[i] /= (1 << (CHAR_BIT*sizeof(RAND_BUF_T)));
    buf[i] /= second_scale;
    */
    //TODO inefficient - change later
    /*
    FLOAT_RAW_TYPE mynum = ((static_cast<float> (tmp_buf[i] % RAND_MAX) / static_cast<float> (RAND_MAX))/second_scale);
    buf[i] = float_to_fixed(mynum);
    buf[i] = round_float(buf[i]);
    assert(buf[i] >= 0);
    assert(buf[i] < (1.0f));
    assert(!isnan(buf[i]));
    */
    buf[i] = 0.0f;
  }
  free(tmp_buf);
  tmp_buf = NULL;
}

//TODO rewrite this to use less space if needed
/*
void rand_buf_to_floats(FP_TYPE * buf, size_t num_FP_TYPEs){
  for(size_t i = 0; i < num_FP_TYPEs; i++){
    unsigned char * char_addr = (unsigned char *) &buf[i];
    buf[i] = ((FP_TYPE) (*char_addr) / (1 << CHAR_BIT));
    assert(!isnan(buf[i]));
    assert(buf[i] >= 0.0f);
    assert(buf[i] < 1.0f);
    //assert(!isnan(((FP_TYPE) (*char_addr) /(sizeof(unsigned char)*CHAR_BIT))));
  }
}
*/

#ifdef NENCLAVE
void print_floatarr(const FP_TYPE * data, int size){
  for(int i = 0; i < size; i++){
    cout << fixed_to_float(data[i]) << ' ';
  }
  cout << endl;
}
#endif

//0 is height, 1 is width
int frievald(const FP_TYPE * a, const FP_TYPE * b, const FP_TYPE * c, 
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
  FP_TYPE * r = (FP_TYPE *) malloc(sizeof(FP_TYPE) * b_width);
  for(int i = 0; i < b_width; i++){
    unsigned char rand_byte;
    rand_bytes(&rand_byte, 1);
    r[i] = (FP_TYPE) (rand_byte & 1);
  }

  /*
  cout << "r: \n";
  print_FP_TYPEmat(r, b_width, 1);
  */

  FP_TYPE * br = NULL;
  int br_w, br_h;
  FP_TYPE * cr = NULL;
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
  print_FP_TYPEmat(br, br_h, br_w);
  
  cout << "cr: \n";
  print_FP_TYPEmat(cr, cr_h, cr_w);
  */

  free(r);
  r = NULL;

  FP_TYPE * axbr;
  int axbr_w, axbr_h;
  matrix_multiply(a, a_width, a_height, 
    br, br_w, br_h,
    &axbr, &axbr_w, &axbr_h, 0);

  free(br);
  br = NULL;
  /*
  cout << "axbr: \n";
  print_FP_TYPEmat(axbr, axbr_h, axbr_w);
  */
  for (int i = 0; i < cr_h; i++){
    //cout << "axbr[" << i << "] " << axbr[i] << " cr[" << i << "] " << cr[i] << endl;
    if (FLOAT_CMP(axbr[i], cr[i])){
#ifdef NENCLAVE    
        cerr << "axbr " << axbr[i] << " cr " << cr[i] << " i " << i << endl;
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

int verify_frievald(const FP_TYPE * a, const FP_TYPE * b, const FP_TYPE * c,
    const int a_width, const int a_height, 
    const int b_width, const int b_height,
    const int c_width, const int c_height){
  for(unsigned int j = 0; j < K_PROBABILITY; j++){
  	if(frievald(a, b, c, 
        a_width, a_height, 
        b_width, b_height,
        c_width, c_height)){
#ifdef NENCLAVE
      cerr << "Frievalds' failed on pass " << j << endl;
#endif
  		return 1;
  	}
  }
  return 0;
}


int parse_structure(char * network_structure_fname, vector<layer_file_t> & layer_files, unsigned int & num_inputs, 
  unsigned int & num_pixels, unsigned int & batchsize, unsigned int & num_labels, unsigned int & epochs){
  
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
  if(file_to_string(network_structure_fname, str_in, file_len+1)){
    return 1;
  }
#else
  int ocall_ret;
  ocall_status = file_to_string(&ocall_ret, network_structure_fname, str_in, file_len+1);
  if(ocall_ret){
    return 1;
  }
#endif  

  num_inputs = atoi(strtok(str_in, " \n"));
  batchsize = atoi(strtok(NULL, " \n"));
  num_pixels = atoi(strtok(NULL, " \n"));
  unsigned int num_layers = atoi(strtok(NULL, " \n"));
  num_labels = atoi(strtok(NULL, " \n"));
  epochs = atoi(strtok(NULL, " \n"));
  
  layer_files.reserve(num_layers);

  for(unsigned int i = 0; i < num_layers; i++){
    layer_file_t lft;

    lft.neurons = atoi(strtok(NULL, " \n"));
    lft.filename = strtok(NULL, " \n");
    lft.type = atoi(strtok(NULL, " \n"));

    layer_files.push_back(lft);
  }

  free(str_in);
  str_in = NULL;

  return layer_files.size() == num_layers? 0 : 1;
}

int mask(const FP_TYPE * input, const FP_TYPE * masks, int input_size, FP_TYPE * output, bool negate){
  if(negate){
    for(int i = 0; i < input_size; i++){
      assert(!isnan(input[i]));
      assert(!isnan(masks[i]));
      output[i] = round_float(input[i]) - round_float(masks[i]);
    }

  }
  else{
    for(int i = 0; i < input_size; i++){
      assert(!isnan(input[i]));
      assert(!isnan(masks[i]));
      output[i] = round_float(input[i]) + round_float(masks[i]);
    }
  }
  return 0;
}

//Assumes a buffer is allocated
int read_all_weights(const vector<layer_file_t> & layers, FP_TYPE ** bufs, unsigned int num_pixels){
  for(size_t i = 0; i < layers.size(); i++){
    int num_floats = layers[i].neurons * (i? layers[i-1].neurons : num_pixels);

    bufs[i] = (FP_TYPE *) malloc(num_floats * sizeof(FP_TYPE));
    //Should check return val
    size_t len = layers[i].filename.size();
    char * fname_buf = (char *) calloc(len+1, sizeof(char));
    strncat(fname_buf, layers[i].filename.c_str(), len);
#ifdef NENCLAVE
    if(read_weight_file_plain(fname_buf, num_floats * sizeof(FP_TYPE), bufs[i])){
      return 1;
    }
#else
    int ocall_ret;
    sgx_status_t ocall_status;
    ocall_status = read_weight_file_plain(&ocall_ret, fname_buf, num_floats * sizeof(FP_TYPE), bufs[i]);
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



int send_to_gpu(const FP_TYPE * data, const int batchsize, const int num_neurons, const int verbose){
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
  if(verbose >= 2){
    print_out((char *) &("Sent matrix dimensions"[0]), false);   
  }    
 
#ifdef NENCLAVE  
  if(write_stream((void *) data, sizeof(FP_TYPE)*batchsize*num_neurons)){
    print_out((char *) &("Failed writing matrix"[0]), true);
    return 1;
  }
#else
  ocall_status = write_stream(&ocall_ret, (void *) data, sizeof(FP_TYPE)*batchsize*num_neurons);
  if(ocall_ret){
    print_out((char *) &("Failed writing matrix"[0]), true);
    return 1;
  }
#endif
  return 0;
}

int receive_from_gpu(FP_TYPE ** result, int * num_neurons, int * batchsize, const int verbose){
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
  if(verbose >= 2){
    print_out((char *) &("Read in result dimensions"[0]), false);
  }

  *batchsize = in_dims[0];
  *num_neurons = in_dims[1];
  

  *result = (FP_TYPE *) malloc(sizeof(FP_TYPE)*(*num_neurons)*(*batchsize));
#ifdef NENCLAVE
  if(read_stream((void *) *result, sizeof(FP_TYPE)*(*num_neurons)*(*batchsize))){
    print_out((char *) &("Failed reading result result"[0]), true);
    return 1;
  }
#else
  ocall_status = read_stream(&ocall_ret, (void *) *result, sizeof(FP_TYPE)*(*num_neurons)*(*batchsize));
  if(ocall_ret){
    print_out((char *) &("Failed reading in result"[0]), true);
    return 1;
  }
#endif      
  if(verbose >= 2){
    print_out((char *) &("Read in result"[0]), false);
  }
  return 0;
}

//For inputs: read in entire batches to enclave memory from ordinary file through OCALLs, and free() data when done.

void backwards_demask_lastlayer(const FP_TYPE * input, const int input_width, const int input_height,
    //const FP_TYPE * input_masks,
    const FP_TYPE * final_data, const int final_data_width, const int final_data_height,
    const FP_TYPE * weights, const int weights_width, const int weights_height,
    //const FP_TYPE * weights_masks,
    const FP_TYPE * grad_output, const int grad_output_width, const int grad_output_height,
    FP_TYPE ** d_ret, FP_TYPE ** e_ret
    //int * re_w, int * ret_w
    ){
  assert(input_height == final_data_height);
  //final_data is softmax(a*b.t)
  //take the derivative of that term
  //TODO only need one small buffer for soft_der at a time
  FP_TYPE * grad_output_transposed = transpose(grad_output, grad_output_width, grad_output_height);
  FP_TYPE * soft_der = (FP_TYPE *) malloc(sizeof(FP_TYPE) * final_data_width * final_data_width);
  FP_TYPE * c_prod = (FP_TYPE *) malloc(sizeof(FP_TYPE) * final_data_width * final_data_height);
  FP_TYPE * c_prod_ptr = c_prod;
  int prod_w, prod_h;

  for(int i = 0; i < final_data_height; i++){
    //TODO softmax_derivative allocates memory
    softmax_derivative(final_data + (i*final_data_width), final_data_width,
      soft_der); 
    //Get the right part of the buffer to write to
    c_prod_ptr = c_prod + (i*final_data_width);
    //Now multiply
    matrix_multiply(grad_output + (i*grad_output_width), grad_output_width, 1,
      soft_der, final_data_width, final_data_width, //Same args for w and h intentional
      (FP_TYPE **) &c_prod_ptr, &prod_w, &prod_h, 0, 0); 

    assert(prod_w == final_data_width);
    assert(prod_h == 1);
  }

  c_prod_ptr = NULL;



  free(grad_output_transposed);
  grad_output_transposed = NULL;
  free(soft_der);
  soft_der = NULL;

  
  //Transpose b
  FP_TYPE * b_transpose = transpose(weights, weights_width, weights_height);

  int d_w, d_h;
  //Allocates new memory pointed to by *d_ret
  matrix_multiply(c_prod, final_data_width, final_data_height,
    b_transpose, weights_height, weights_width,
    d_ret, &d_w, &d_h, 0);


  free(b_transpose);
  b_transpose = NULL;
  
  //Calculate e_ret as c.t() @ input
  //First transpose c
  FP_TYPE * c_t = transpose(c_prod, final_data_width, final_data_height);
  free(c_prod);
  c_prod = NULL;
  FP_TYPE * e = NULL;
  int e_w, e_h;
  matrix_multiply(c_t, final_data_height, final_data_width, //Swapped for transpose
    input, input_width, input_height,
    &e, &e_w, &e_h);

  //TODO transpose e
  *e_ret = transpose(e, e_w, e_h);

  free(e);
  e = NULL;
  free(c_t);
  c_t = NULL;

  return;
}


//TODO create and use grad_mask
int backwards_demask_ordinary(const FP_TYPE * input, const int input_width, const int input_height,
  const FP_TYPE * input_mask, 
  const FP_TYPE * outputs, const int outputs_width, const int outputs_height,
  const FP_TYPE * weights, const int weights_width, const int weights_height,
  const FP_TYPE * weights_mask,
  FP_TYPE * grad_output, const int grad_output_width, const int grad_output_height,
  FP_TYPE ** d_ret, FP_TYPE ** e_ret, int verbose = 0){

  assert(grad_output_height == input_height);

  FP_TYPE * a_mul_wmask;
  int a_mul_wmask_height;
  int a_mul_wmask_width;

  matrix_multiply(input, input_width, input_height, 
    weights_mask, weights_width, weights_height,
     &a_mul_wmask, &a_mul_wmask_width, &a_mul_wmask_height);

  FP_TYPE * weight_mask_weights = (FP_TYPE *) malloc(sizeof(FP_TYPE) * weights_height * weights_width);
  //diff_term eventually
  FP_TYPE * a_prod_weights_mask = (FP_TYPE *) malloc(sizeof(FP_TYPE) * weights_height * weights_width);
  
  matrix_sub(weights_mask, weights, weights_height * weights_width, weight_mask_weights);

  FP_TYPE * diff_term;
  int diff_term_w, diff_term_h;
  matrix_multiply(input_mask, input_width, input_height,
    weight_mask_weights, weights_width, weights_height,
    &diff_term, &diff_term_w, &diff_term_h);

  matrix_sub(diff_term, a_mul_wmask, diff_term_w*diff_term_h, diff_term);

  free(weight_mask_weights);
  weight_mask_weights = NULL;
  free(a_prod_weights_mask);
  a_prod_weights_mask = NULL;

  FP_TYPE * weight_mask_transpose = transpose(weights_mask, weights_width, weights_height);
  FP_TYPE * weights_transpose = transpose(weights, weights_width, weights_height); //might need later

  free(a_mul_wmask);
  a_mul_wmask = NULL;

  //Mask the grad. output being sent to the GPU
  FP_TYPE * grad_rand_mask = (FP_TYPE *) malloc(sizeof(FP_TYPE) * grad_output_height * outputs_width);
  rand_floats(grad_rand_mask, grad_output_height * outputs_width);
  mask(grad_output, grad_rand_mask, grad_output_width * grad_output_height, grad_output, false);


  FP_TYPE * c_transformed = (FP_TYPE *) malloc(sizeof(FP_TYPE) * grad_output_height * outputs_width);
  transform_and_mult(grad_output, outputs, diff_term, c_transformed, grad_output_height * outputs_width);

  FP_TYPE * weightmask_b = (FP_TYPE *) malloc(sizeof(FP_TYPE) * weights_height * weights_width);
  matrix_sub(weights_mask, weights, weights_height*weights_width, weightmask_b);

  //Now transpose
  FP_TYPE * wrm_b_t = transpose(weightmask_b, weights_width, weights_height);

  free(weightmask_b);
  weightmask_b = NULL;

  FP_TYPE * diffc_diffa;
  int diffc_diffa_w, diffc_diffa_h;  

  //Apply transform to grad_rand_mask
  //transform(grad_rand_mask, diff_term, grad_output_height*grad_output_width, grad_rand_mask);
  //FP_TYPE * res = transform(grad_rand_mask, diff_term, grad_output_height*grad_output_width);
  //free(res);
  //res = NULL;


  free(diff_term);
  diff_term = NULL;

  matrix_multiply(grad_rand_mask, outputs_width, grad_output_height, 
   wrm_b_t, weights_height, weights_width,
   &diffc_diffa, &diffc_diffa_w, &diffc_diffa_h);

  free(wrm_b_t);
  wrm_b_t = NULL;
    
  FP_TYPE * diffb;
  int diffb_w, diffb_h;

  matrix_multiply(c_transformed, grad_output_width, grad_output_height, 
    weight_mask_transpose, weights_height, weights_width,
    &diffb, &diffb_w, &diffb_h, 0, 1);

  free(weight_mask_transpose);
  weight_mask_transpose = NULL;

  //Send c_transformed to GPU 
  //First, round it
  //round_floatmat(c_transformed, grad_output_height*grad_output_width);
  if(send_to_gpu(c_transformed, grad_output_height, grad_output_width, verbose)){
    print_out((char *) &("Failed to send c_transformed"[0]), true);
    return 1;
  }
  //Receive d from GPU  
  FP_TYPE * d; //Do not free - returned
  int d_w, d_h;
  if(receive_from_gpu(&d, &d_w, &d_h, verbose)){
    print_out((char *) &("Failed to receive d from GPU"[0]), true);
    return 1;
  }
  
  //Verify
  if(verify_frievald(c_transformed, weights_transpose, d,
      grad_output_width, grad_output_height, 
      weights_height, weights_width,
      d_w, d_h)){
    print_out((char *) &("Frievalds' algorithm failed on d"[0]), true);
    return 1;
  }

  free(weights_transpose);
  weights_transpose = NULL; 
  
  FP_TYPE * difftemp = (FP_TYPE *) malloc(sizeof(FP_TYPE) * grad_output_height*weights_width);
  
  matrix_sub(diffc_diffa, diffb, grad_output_height*weights_width, difftemp);

  /*
  for(int i = 0; i < grad_output_height*weights_width; i++){
    d[i] += diffc_diffa[i] - diffb[i];
  }
  */

  free(diffc_diffa);
  diffc_diffa = NULL;
  free(diffb);
  diffb = NULL;
  
  //matrix_add(d, difftemp, grad_output_height*weights_width, d);

  //free(difftemp);
  //difftemp = NULL;

  FP_TYPE * diffx;
  int diffx_w, diffx_h;

  FP_TYPE * c_transpose = transpose(c_transformed, outputs_width, outputs_height);

  matrix_multiply(c_transpose, outputs_height, outputs_width,
    input_mask, input_width, input_height,
    &diffx, &diffx_w, &diffx_h);

  

  FP_TYPE * grm_transposed = transpose(grad_rand_mask, grad_output_width, outputs_height);

  free(grad_rand_mask);
  grad_rand_mask = NULL;


  FP_TYPE * rand_mask_a = (FP_TYPE *) malloc(sizeof(FP_TYPE) * input_height*input_width);
  //matrix_sub
  matrix_sub(input_mask, input, input_height*input_width, rand_mask_a);


  FP_TYPE * diffz_diffy;
  int diffz_diffy_w, diffz_diffy_h;
  matrix_multiply(grm_transposed, grad_output_height, outputs_width, 
    rand_mask_a, input_width, input_height, //Reversed due to transposition
     &diffz_diffy, &diffz_diffy_w, &diffz_diffy_h);

  free(grm_transposed);
  grm_transposed = NULL;
  free(rand_mask_a);
  rand_mask_a = NULL;


  //Get e from GPU
  FP_TYPE * e;
  int e_w, e_h;
  if(receive_from_gpu(&e, &e_w, &e_h, verbose)){
    print_out((char *) &("Failed to receive e from GPU"[0]), true);
    return 1;
  }
  //Verify that e == c_transformed.t() @ a
  //First, get transposed of c_transformed
  if(verify_frievald(c_transpose, input, e,
      grad_output_height, grad_output_width, 
      input_width, input_height,
      e_w, e_h)){
    print_out((char *) &("Frievalds' algorithm failed on e"[0]), true);
    return 1;
  }

  free(c_transpose);
  c_transpose = NULL;

  free(c_transformed);
  c_transformed = NULL;

  assert(e_w == weights_height);
  assert(e_h == weights_width);

  /*
  for(int i = 0; i < input_height*input_width; i++){
    e[i] += diffz_diffy - diffx;
  }
  */
  
  

  matrix_sub(diffz_diffy, diffx, grad_output_height*input_width, diffz_diffy);
  matrix_add(e, diffz_diffy, input_height*input_width, e);

  //Transpose error
  *e_ret = transpose(e, e_w, e_h);

  free(diffz_diffy);
  diffz_diffy = NULL;

  free(diffx);
  diffx = NULL;

  free(e);
  e = NULL;

  *d_ret = d;
  return 0;
}

void forward_demask(const FP_TYPE * input, const int input_width, const int input_height,
  const FP_TYPE * input_masks, 
  const FP_TYPE * weights_unmasked, const int weights_width, const int weights_height,
  const FP_TYPE * weights_masks,
  const FP_TYPE * gpu_output,
  FP_TYPE ** result, int * result_width, int * result_height){

  assert(input_width == weights_height);

  int w_dummy, h_dummy;

  FP_TYPE * d3_d = NULL;
  matrix_multiply(input_masks, input_width, input_height,
   weights_unmasked, weights_width, weights_height,
   &d3_d, &w_dummy, &h_dummy, 1);

  //diff2 goes in result
  matrix_multiply(input, input_width, input_height,
    weights_masks, weights_width, weights_height,
    result, result_width, result_height);
  
  matrix_sub(d3_d, *result, (*result_width)*(*result_height), *result);

  free(d3_d);
  d3_d = NULL;

}

//Actual is ground truth - 0 or 1 for each label
FP_TYPE crossentropy_loss(const unsigned int * actual, const FP_TYPE * predicted,
 const int num_possible_labels, const int batchsize){
  FP_TYPE sum = 0.0f;
  for(int i = 0; i < batchsize; i++){
    assert((int)actual[i] < num_possible_labels);
    sum -= log(predicted[(i*num_possible_labels)+actual[i]]);
  }
  return sum/(FP_TYPE)batchsize;
}

//Allocates memory
FP_TYPE * crossentropy_derivative(const unsigned int * actual, const FP_TYPE * predicted,
 const int num_possible_labels, const int batchsize){
  FP_TYPE * ret = (FP_TYPE *) calloc(batchsize*num_possible_labels, sizeof(FP_TYPE));
  for(int i = 0; i < batchsize; i++){
    int idx = (i*num_possible_labels)+actual[i];
    ret[idx] = (-1.0f/predicted[idx])/batchsize;
  }
  return ret;
}


void update_weights(FP_TYPE * weights, const FP_TYPE * weights_gradient, const int total_elts, const FP_TYPE learning_rate){
  for(int i = 0; i < total_elts; i++){
    weights[i] -= learning_rate*weights_gradient[i];
  }
  return;
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
  unsigned int num_epochs = 1;
  bool skip_masking = true; 
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
  if(verbose >= 2){
    print_out((char *) &("Initialized I/O streams"[0]), false);
  }
  

  if(parse_structure(network_structure_fname, layer_files, 
    num_inputs, num_pixels, batchsize, num_possible_labels, num_epochs)){
    print_out((char *) &("Network parsing failed!"[0]), true);
    return 1;
  }
  if(verbose >= 2){
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

  FP_TYPE ** layer_data;
  layer_data = (FP_TYPE **) malloc(sizeof(FP_TYPE *) * num_layers);
  //Read in all layer data
  if(read_all_weights(layer_files, layer_data, num_pixels)){
    print_out((char *) &("Failed to read weights"[0]), true);
    return 1;
  }
  if(verbose >= 2){
    print_out((char *) &("Read in weights"[0]), false);
  }

  unsigned num_batches = (num_inputs / batchsize) + ((num_inputs % batchsize) ? 0 : 1);

  FP_TYPE * input_data;

#ifdef NENCLAVE
  if(start_timing(TASK_ALL)){
    return 1;
  }
#else
  ocall_status = start_timing(&ocall_ret, TASK_ALL);
  if(ocall_ret){
    return 1;
  }
#endif  

  for(unsigned int epoch_idx = 0; epoch_idx < num_epochs; epoch_idx++){

    if(verbose >= 2){
      std::string eph_msg = "Enclave beginning epoch " + std::to_string(epoch_idx);
      print_out((char *) &(eph_msg[0]), false);
    }

    bool epoch_reset = true;

    for(unsigned int batch_idx = 0; batch_idx < num_batches; batch_idx++){
    //Get images into a matrix
    unsigned num_images_this_batch = (batch_idx != num_batches-1) ? (batchsize) : (num_inputs % num_batches);
    input_data = (FP_TYPE *) malloc(sizeof(FP_TYPE) * num_images_this_batch * num_pixels);
    FP_TYPE * image_data_csv_ptr = input_data;
    unsigned int * data_labels = (unsigned int *) malloc(sizeof(unsigned int) * num_inputs);
    unsigned int * data_labels_ptr = data_labels;

    //FP_TYPE * final_data = NULL;
    FP_TYPE * gpu_unmasked_result = NULL;

    for(unsigned int image_idx = 0; image_idx < num_images_this_batch; image_idx++){
#ifdef NENCLAVE        
      if(csv_getline(input_csv_filename, image_data_csv_ptr, data_labels_ptr, num_pixels, epoch_reset)){
        print_out((char *) &("Failed to read input .csv"[0]), true);
        return 1;
      }
#else
      ocall_status = csv_getline(&ocall_ret, input_csv_filename, image_data_csv_ptr, data_labels_ptr, num_pixels, epoch_reset);
      if(ocall_ret){
        print_out((char *) &("Failed to read input .csv"[0]), true);
        return 1;
      }
#endif      
      epoch_reset = false; //Only reset stream on first go of an epoch  
      
      image_data_csv_ptr += num_pixels; //Increment pointer
      data_labels_ptr++;
    }

    if(verbose >= 2){
        std::string read_str = "Read " + std::to_string(num_images_this_batch) + " inputs from " + std::string(input_csv_filename);
        print_out((char *) &(read_str.c_str()[0]), false);
      }

    FP_TYPE ** gpu_inputs = (FP_TYPE **) malloc(sizeof(FP_TYPE *) * num_layers); //Slight misnomer - activated only by the prev. layer
    FP_TYPE ** gpu_outputs = (FP_TYPE **) malloc(sizeof(FP_TYPE *) * (num_layers)); 
    FP_TYPE ** input_masks = (FP_TYPE **) malloc(sizeof(FP_TYPE *) * num_layers);
    FP_TYPE ** weights_mask = (FP_TYPE **) malloc(sizeof(FP_TYPE *) * num_layers);

#ifdef NENCLAVE
    if(start_timing(TASK_FORWARD)){
      return 1;
    }
#else
    ocall_status = start_timing(&ocall_ret, TASK_FORWARD);
    if(ocall_ret){
      return 1;
    }
#endif  

    for(unsigned int i = 0; i < num_layers; i++){
      gpu_inputs[i] = input_masks[i] = weights_mask[i] = gpu_outputs[i] = NULL;
    }
    
    //Now we have the whole batch in a single array
    for(unsigned int layer_idx = 0; layer_idx < num_layers; layer_idx++){
      int num_neurons;
      num_neurons = layer_idx ? layer_files[layer_idx-1].neurons : num_pixels;
      FP_TYPE * input_masking_target = NULL; //TODO remove
      if(!layer_idx){
        //TODO remove this
        input_masking_target = gpu_inputs[layer_idx] = input_data;
        input_data = NULL;
      }
      else{
        input_masking_target = gpu_inputs[layer_idx] = gpu_unmasked_result;
        gpu_unmasked_result = NULL;
      }

      //Round off input
      round_floatmat(input_masking_target, num_neurons*num_images_this_batch);

      //Mask the current input
      //First, get the random mask
      input_masks[layer_idx] = (FP_TYPE *) malloc(sizeof(FP_TYPE)*num_neurons*num_images_this_batch);
      rand_floats(input_masks[layer_idx], num_neurons*num_images_this_batch);

      //Next, mask the data
      if(!skip_masking){
        mask(input_masking_target, input_masks[layer_idx], num_neurons*num_images_this_batch, gpu_inputs[layer_idx], false);
      }
      input_masking_target = NULL;

      if(verbose >= 2){
        print_out((char *) &("Finished masking input"[0]), false);
      }    
     
      //Send masked input to the GPU
      if(send_to_gpu(gpu_inputs[layer_idx], num_images_this_batch, num_neurons, verbose)){
        print_out((char *) &("Failed to send input data"[0]), true);
        return 1;
      }

      int num_weights = num_neurons * layer_files[layer_idx].neurons;
      //Round off weights
      round_floatmat(layer_data[layer_idx], num_weights);
      //Mask weights
      weights_mask[layer_idx] = (FP_TYPE *) malloc(sizeof(FP_TYPE) * num_weights);
      rand_floats(weights_mask[layer_idx], num_weights, WEIGHTS_SCALE);
      //Cast should be explicit, for the non-SGX version

      //Quantize weights before masking
      round_floatmat(layer_data[layer_idx], num_weights);

      if(!skip_masking){
        mask(layer_data[layer_idx], weights_mask[layer_idx], num_weights, layer_data[layer_idx], false);
        
      }
      if(verbose >= 2){
        print_out((char *) &("Finished masking weights"[0]), false);
      }

      //Send weights to GPU
      if(send_to_gpu(layer_data[layer_idx], num_neurons, layer_files[layer_idx].neurons, verbose)){
        print_out((char *) &("Failed to send weights data"[0]), true);
        return 1;
      }
      //Receive result back
      int num_result_neurons;
      int result_batchsize;
      if(receive_from_gpu(&gpu_outputs[layer_idx], &num_result_neurons, &result_batchsize, verbose)){
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

      //Validate result with Frievalds' algorithm
      //If it fails, send {-1, -1} back to the GPU and exit
      if(verify_frievald(gpu_inputs[layer_idx], layer_data[layer_idx], gpu_outputs[layer_idx], 
          num_neurons, num_images_this_batch,
          layer_files[layer_idx].neurons, num_neurons,
          num_result_neurons, result_batchsize)){
        //Verification failed!
        print_out((char *) &("Frievalds' algorithm failed!"[0]), true);
      }
      else{

        if(verbose >= 2){
#ifdef NENCLAVE          
          cout << "Frievalds' algorithm succeeded!" << endl;
#else
          print_out((char *) &("Frievalds' algorithm succeeded!"[0]), false);          
#endif              
        }
    
      }

      //Result has been verified
      //Unmask (forward) the GPU result
      //Recall that input and weights are currently masked
      
      int gpu_unmasked_w, gpu_unmasked_h;
      if(!skip_masking){
        //Unmask weights NVM
        //mask(layer_data[layer_idx], weights_mask[layer_idx], num_neurons, layer_data[layer_idx], true);

        forward_demask(gpu_inputs[layer_idx], num_neurons, num_images_this_batch,
          input_masks[layer_idx], 
          layer_data[layer_idx], layer_files[layer_idx].neurons, num_neurons,
          weights_mask[layer_idx], 
          gpu_outputs[layer_idx],
          &gpu_unmasked_result, &gpu_unmasked_w, &gpu_unmasked_h);
        
      }
      else{
         gpu_unmasked_result = (FP_TYPE *) malloc(sizeof(FP_TYPE)*result_batchsize*layer_files[layer_idx].neurons);
         memcpy(gpu_unmasked_result, gpu_outputs[layer_idx], sizeof(FP_TYPE)*result_batchsize*layer_files[layer_idx].neurons);
         gpu_unmasked_h = result_batchsize;
         gpu_unmasked_w = layer_files[layer_idx].neurons;
      }
      

      //Save result from GPU and activate
      if(layer_idx != num_layers-1){
        activate(gpu_unmasked_result, gpu_unmasked_h, gpu_unmasked_w);
      }
      else{
        assert(gpu_unmasked_h == (int) num_images_this_batch);
        assert(gpu_unmasked_w == (int) num_possible_labels);
        softmax(gpu_unmasked_result, gpu_unmasked_w, gpu_unmasked_h);   
      }
      
      //Also, don't deallocate weights either, also needed in backprop

    } //layer_idx (forward pass)

#ifdef NENCLAVE
  if(finish_timing(TASK_FORWARD)){
    return 1;
  }
#else
  ocall_status = finish_timing(&ocall_ret, TASK_FORWARD);
  if(ocall_ret){
    return 1;
  }
#endif 

    if(backprop){

      if(verbose >= 2){
#ifdef NENCLAVE
        cout << "Starting backpropagation...\n";
#endif        
      }

#ifdef NENCLAVE
  if(start_timing(TASK_BACKPROP)){
    return 1;
  }
#else
  ocall_status = start_timing(&ocall_ret, TASK_BACKPROP);
  if(ocall_ret){
    return 1;
  }
#endif 

      FP_TYPE * derivative = crossentropy_derivative(data_labels, gpu_unmasked_result, num_possible_labels, num_images_this_batch);

      //Print output
      if(verbose >= 2){
        //Loss for the whole batch
        FLOAT_RAW_TYPE loss_raw = crossentropy_loss(data_labels, gpu_unmasked_result, num_possible_labels, num_images_this_batch);
        std::string loss_str = "Loss: " + std::to_string(loss_raw) + " batch: " + std::to_string(batch_idx) + " epoch: " + std::to_string(epoch_idx);
        print_out((char *) &(loss_str.c_str()[0]), false);
      }

      for(int rev_layer_idx = num_layers-1; rev_layer_idx >= 0; rev_layer_idx--){

        if(rev_layer_idx == (int)num_layers-1){
          FP_TYPE * d_ret = NULL;
          FP_TYPE * e_ret = NULL;
          //First, unmask the inputs and weights
          mask(gpu_inputs[rev_layer_idx], input_masks[rev_layer_idx], layer_files[rev_layer_idx-1].neurons*num_images_this_batch, gpu_inputs[rev_layer_idx], true);
          mask(layer_data[rev_layer_idx], weights_mask[rev_layer_idx], 
            layer_files[rev_layer_idx-1].neurons*layer_files[rev_layer_idx].neurons, layer_data[rev_layer_idx], true);
          
          backwards_demask_lastlayer(gpu_inputs[rev_layer_idx], layer_files[rev_layer_idx-1].neurons, num_images_this_batch,
            gpu_unmasked_result, layer_files[rev_layer_idx].neurons, num_images_this_batch,
            layer_data[rev_layer_idx], layer_files[rev_layer_idx].neurons, layer_files[rev_layer_idx-1].neurons,
            derivative, num_possible_labels, num_images_this_batch,
            &d_ret, &e_ret);

          free(gpu_unmasked_result);
          gpu_unmasked_result = NULL;

          //Update weights - already unmasked
          update_weights(layer_data[rev_layer_idx], e_ret, 
            layer_files[rev_layer_idx-1].neurons*layer_files[rev_layer_idx].neurons, LEARNING_RATE);      

          free(e_ret);
          e_ret = NULL;
          free(input_data);
          input_data = NULL;

          free(derivative);
          derivative = NULL;
          derivative = d_ret;
          d_ret = NULL;
        }
        else{
          int num_neurons = rev_layer_idx ? (int) layer_files[rev_layer_idx-1].neurons : (int) num_pixels;

          FP_TYPE * d_ret = NULL;
          FP_TYPE * e_ret = NULL;
          int demask_result = backwards_demask_ordinary(gpu_inputs[rev_layer_idx], num_neurons, num_images_this_batch,
            input_masks[rev_layer_idx], 
            gpu_outputs[rev_layer_idx], layer_files[rev_layer_idx].neurons, num_images_this_batch,
            layer_data[rev_layer_idx], layer_files[rev_layer_idx].neurons, num_neurons,
            weights_mask[rev_layer_idx],
            derivative, layer_files[rev_layer_idx].neurons, num_images_this_batch,
            &d_ret, &e_ret, verbose);
     

          if(demask_result){
            std::string out_str = "Failed in demasking at layer " + std::to_string(rev_layer_idx) 
              + ", epoch " + std::to_string(epoch_idx) + ", batch " + std::to_string(batch_idx);
            print_out((char *) &(out_str[0]), true);
            return 1;
          }     

          free(derivative);
          derivative = NULL;
          derivative = d_ret;
          d_ret = NULL;

          //Update weights with e_ret - first unmask
          mask(layer_data[rev_layer_idx], weights_mask[rev_layer_idx], 
            num_neurons*layer_files[rev_layer_idx].neurons, layer_data[rev_layer_idx], true);
          update_weights(layer_data[rev_layer_idx], e_ret, 
            num_neurons*layer_files[rev_layer_idx].neurons, LEARNING_RATE);


          free(e_ret);
          e_ret = NULL;
        }

        free(gpu_inputs[rev_layer_idx]);
        gpu_inputs[rev_layer_idx] = NULL;
        free(input_masks[rev_layer_idx]);
        input_masks[rev_layer_idx] = NULL;
        free(weights_mask[rev_layer_idx]);
        weights_mask[rev_layer_idx] = NULL;
        //TODO review - do we actually need num_layers gpu_outputs?
        free(gpu_outputs[rev_layer_idx]);
        gpu_outputs[rev_layer_idx] = NULL;
        
      } //rev_layer_idx

#ifdef NENCLAVE
      if(finish_timing(TASK_BACKPROP)){
        return 1;
      }
#else
      ocall_status = finish_timing(&ocall_ret, TASK_BACKPROP);
      if(ocall_ret){
        return 1;
      }
#endif 

        free(derivative);
        derivative = NULL;

      } //if(backprop)
      else{
        for(int rev_layer_idx = num_layers-1; rev_layer_idx >= 0; rev_layer_idx--){
          free(gpu_inputs[rev_layer_idx]);
          gpu_inputs[rev_layer_idx] = NULL;
          free(input_masks[rev_layer_idx]);
          input_masks[rev_layer_idx] = NULL;
          free(gpu_outputs[rev_layer_idx]);
          gpu_outputs[rev_layer_idx] = NULL;
          free(weights_mask[rev_layer_idx]);
          weights_mask[rev_layer_idx] = NULL;
        }
      }

      //Free up labels
      free(data_labels);
      data_labels = NULL;

      //Free saved data buffers
      free(gpu_inputs);
      gpu_inputs = NULL;
      free(input_masks);
      input_masks = NULL;
      free(gpu_outputs);
      gpu_outputs = NULL;
      free(weights_mask);
      weights_mask = NULL;
      
    } //batch_idx
  } //epoch_idx

  

#ifdef NENCLAVE
  if(finish_timing(TASK_ALL)){
    return 1;
  }
#else
  ocall_status = finish_timing(&ocall_ret, TASK_ALL);
  if(ocall_ret){
    return 1;
  }
#endif  

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

  //Close streams
#ifdef NENCLAVE
  if(close_streams()){
    return 1;
  }
#else
  ocall_status = close_streams(&ocall_ret);
  if(ocall_ret){
    return 1;
  }
#endif
  return 0;

}

//Verbosity levels:
//0 no output (except in case of error)
//1 timing data
//2 logging
//3 data
