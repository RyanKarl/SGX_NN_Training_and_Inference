//enclave_functions.h
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

#ifndef ENCLAVE_FUNCTIONS_H
#define ENCLAVE_FUNCTIONS_H

#define DATA_DIMENSIONS 3

//Return 1 if verification fails, 0 if successful
int verify_frievald(float * data, int datashape[DATA_DIMENSIONS]){
  return 0;
}

//Return 1 if activation fails, 0 if successful
int activate(float * data_in, int data_inshape[DATA_DIMENSIONS], float ** data_out, int * data_outshape[DATA_DIMENSIONS]){
  //Use the below if things are done in-place
  //data_outshape must have DATA_DIMENSIONS elements
  *data_out = data_in;
  *data_outshape = data_inshape;
  return 0;
}

#endif
