//enclave_functions.h
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni, Kathryn Hund

#ifndef ENCLAVE_FUNCTIONS_H
#define ENCLAVE_FUNCTIONS_H


//Return 1 if verification fails, 0 if successful
int verify_frievald(char * data, int datalen){
  return 0;
}

//Return 1 if activation fails, 0 if successful
int activate(char * data_in, int data_inlen, char ** data_out, int * data_outlen){
  //Use the below if things are done in-place
  *data_out = data_in;
  *data_outlen = data_inlen;
  return 0;
}

#endif
