//enclave_driver.c
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni
//gcc ./enclave_driver.c -pedantic -Wall -Werror -O3 -std=gnu99 -o enclave_driver

//Defined first to go before other standard libs
/*
#define malloc_consolidate(){\
fprintf(stdout, "%s %s \n", __FILE__, __LINE__);\
malloc_consolidate();\
}
*/



#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
//<sys/types.h> used for pipes
#include <sys/types.h>
#include <fcntl.h> 
#include <string.h>
#include <errno.h>
#include <getopt.h>
#include <malloc.h>

/*
#include <execinfo.h>
#define NUM_FRAMES 10

void __real_malloc_consolidate();

void __wrap_malloc_consolidate(){
  void * arr[NUM_FRAMES];
  printf("malloc_consolidate called\n");
  fflush(stdout);
  size_t bt_size = backtrace(arr, NUM_FRAMES);
  backtrace_symbols_fd(arr, bt_size, STDOUT_FILENO);
  fflush(stdout);
  __real_malloc_consolidate();
  return;
}
*/

#include "enclave_functions.h"

int main(int argc, char ** argv){

  //i and o are filenames of named pipe
  //sleepy not implemented
  int sleepy = 0;
  int use_std_io = 0;
  char * input_pipe_path = NULL;
  char * output_pipe_path = NULL;
  int verbose = 0;

  char c;
  while((c = getopt(argc, argv, "sfi:o:v")) != -1){
    switch(c){
      case 'v':{
        verbose += 1;
        break;
      }
      case 's':{
        sleepy = 1;
        break;
      }
      case 'f':{
        use_std_io = 1;
        break;
      }
      case 'i':{
        input_pipe_path = optarg;
        break;
      }
      case 'o':{
        output_pipe_path = optarg;
        break;  
      }
      default:{
        fprintf(stderr, "ERROR: unrecognized argument %c\n", c);
        return 0;
      }
    }
  }

  
  int input_pipe = 0;
  if(!use_std_io){
    input_pipe = open(input_pipe_path, O_RDONLY);
  } 
  if(input_pipe == -1){
    fprintf(stderr, "ERROR: could not open input pipe %s\n", input_pipe_path);
    return 1;
  }
  int output_pipe = 0;
  if(!use_std_io){
    output_pipe = open(output_pipe_path, O_WRONLY);
  }  
  if(output_pipe == -1){
    fprintf(stderr, "ERROR: could not open output pipe %s\n", output_pipe_path);
    return 1;
  }

  //c file object which is wrapper for file descriptor
  FILE * instream;
  FILE * outstream;
  if(use_std_io){
    if(verbose >= 2){
      fprintf(stdout, "Assigning I/O to std\n");
      fflush(stdout);
    }
    instream = stdin;
    outstream = stdout;
  }
  else{
    if(verbose >= 2){
      fprintf(stdout, "Assigning I/O to file descriptors\n");
      fflush(stdout);
    }
    instream = fdopen(input_pipe, "r");
    outstream = fdopen(output_pipe, "w");
  }
   
  if(instream == NULL){
    fprintf(stderr, "ERROR: unable to open input pipe as file stream\n");
    fflush(stderr);
    return 1;
  }
  if(outstream == NULL){
    fprintf(stderr, "ERROR: unable to open output pipe as file stream\n");
    fflush(stderr);
    return 1;
  }


  if(sleepy){
    fprintf(stderr, "WARNING: Sleeping not yet implemented!\n");
    fflush(stderr);
  }

  //Track how many rounds we go through
  unsigned int round = 0;
  while(1){
    //Shape of input data
    //Assumes a linear buffer
    int matrix_n[NUM_MATRICES][MAT_DIM];
    //Get number of bytes from pipe
    if(verbose){
      fprintf(stdout, "About to get header (%li bytes)\n", sizeof(matrix_n));
      fflush(stdout);
    }
    
    //Dangerous cast - works because array is entirely contiguous
    if(!fread((int*) matrix_n, sizeof(int), NUM_MATRICES*MAT_DIM, instream)){
      fprintf(stderr, "ERROR: could not read header\n");
      fflush(stderr);
      return 1;
    }

    if(verbose >= 2){
      fprintf(stdout, "Matrix dimensions received: ");
      for(unsigned int i = 0; i < NUM_MATRICES; i++){
        for(unsigned int j = 0; j < MAT_DIM; j++){
          fprintf(stdout, "%d ", matrix_n[i][j]);
        }
      }
      fprintf(stdout, "\n");
    }
    
    //Now we know how many bytes to receive
    //Read in data
    int num_in = 0;
    for(unsigned int i = 0; i < NUM_MATRICES; i++){
      int mat_size = 1;
      for(unsigned int j = 0; j < MAT_DIM; j++){
        mat_size *= matrix_n[i][j];
      }
      num_in += mat_size;
    }
    
    //num_bytes_in should be product of matrix dimensions i.e. 3*4*5 = 60 times sizeof(float)
    if(verbose){
      fprintf(stdout, "Total matrix elements to receive: %d \n", num_in);
      fflush(stdout);
    }

    //Allocate array of floats for data
    float * input = malloc((unsigned int) num_in * sizeof(float));
    if(!input){
      fprintf(stderr, "ERROR: malloc failed\n");
      return -1;
    }
    
    //Read in data from array to dynamic memory (note num_in is finite number of dimensions)
    if(!fread(input, sizeof(float), num_in, instream)){
      fprintf(stderr, "ERROR: could not read bytes\n");
      return 1;
    }

    if(verbose >= 2){
      fprintf(stdout, "Finished reading input from pipe: ");
      if(verbose >= 3){
        for(int i = 0; i < num_in; i++){
          fprintf(stdout, "%f ", input[i]);
        }
      }
      fprintf(stdout, "\n");
      fflush(stdout);
    }
    
    //First verify data (TODO consider optimizing freivalds)
    if(verify_frievald(input, matrix_n[0], matrix_n[1], matrix_n[2])){
      //Frievald's algorithm failed - send back -1
      int frievald_result[MAT_DIM];
      frievald_result[0] = frievald_result[1] = -1;
      if(verbose){
        fprintf(stdout, "Frievald's algorithm failed on round %d\n", round);
        fprintf(stdout, "Sending response: %d %d\n", frievald_result[0], frievald_result[1]);
        fflush(stdout);
      }
      if((!fwrite((int *) frievald_result, sizeof(frievald_result[0]), MAT_DIM, outstream)) || fflush(outstream)){
        fprintf(stderr, "ERROR: could not write failed verification\n");
        return 1;
      }
    }
    else{
    
      if(verbose){
        fprintf(stdout, "Frievald's algorithm succeeded on round %d\n", round);
        fflush(stdout);
      }
    
      //Now compute activation
      float * activation_data = NULL;
      int activation_n[MAT_DIM] = {0, 0};
      float * c_mat_addr = input + ((matrix_n[0][0]*matrix_n[0][1])+(matrix_n[1][0]*matrix_n[1][1])); //Address of the start of C
      int new_act_data = 0;
      if(activate(c_mat_addr, matrix_n[2], (float **) &activation_data, (int *) activation_n, &new_act_data)){
        fprintf(stderr, "ERROR: activation failed on round %d\n", round);
        fflush(stderr);
        return 1;
      }
      else{
        if(verbose >= 2){
          fprintf(stdout, "Activation successful on round %d\n", round);
          fflush(stdout);
        }
      }
      
      if(verbose >= 2){
        fprintf(stdout, "Activated data (%d x %d): ", activation_n[0], activation_n[1]);
        if(verbose >= 3){
          for(int i = 0; i < activation_n[0]*activation_n[1]; i++){
            fprintf(stdout, "%f ", activation_data[i]);
          }
        }
        fprintf(stdout, "\n");
        fflush(stdout);
      }
      
      
      
      //Send header with length
      //Another dangerous cast
      if((!fwrite((int *) activation_n, sizeof(activation_n[0]), MAT_DIM, outstream)) || fflush(outstream)){
        fprintf(stderr, "ERROR: could not write header\n");
        fflush(stderr);
        return 1;
      }
      else{
        if(verbose >=2){
          fprintf(stdout, "Sent header!\n");
          fflush(stdout);
        }
      }
      
      int activated_size = activation_n[0]*activation_n[1];
      
      if(verbose >= 2){
        fprintf(stdout, "Sent output header: %ld bytes\n", MAT_DIM*sizeof(activation_n[0]));
        if(verbose >= 3){
          fprintf(stdout, "Sending %d float outputs: ", activated_size);
          for(int i = 0; i < activated_size; i++){
            fprintf(stdout, "%f ", activation_data[i]);
          }
        }
        fprintf(stdout, "\n");
        fflush(stdout);
      }
      
      //Now send actual data
      if((!fwrite((float *) activation_data, sizeof(float), activated_size, outstream)) || fflush(outstream)){
        fprintf(stderr, "ERROR: could not write out\n");
        fflush(stderr);
        return 1;
      }
      //Clean up memory (this is the data i.e. activation data)
      
      //Check that the activated data is a valid pointer, and not in range of input (i.e. fresh)
      if((activation_data != NULL) && new_act_data){
        if(verbose >= 2){
          fprintf(stdout, "Freeing activated data\n");
          fflush(stdout);
        }
        free(activation_data);
        activation_data = NULL;
      }
      

      if(verbose){
        fprintf(stdout, "Finished writing\n\n");
        fflush(stdout);
      }
      
    }
    
    if(verbose >= 2){
      fprintf(stdout, "Freeing input data\n");
      fflush(stdout);
    }
    free(input);
    input = NULL;

    round++;
  }

  //Close the given pipes, clean up memory
  //This code will rarely be called, as the program will be killed by the Python driver
  
  fclose(instream);
  fclose(outstream);
  free(instream);
  instream = NULL;
  free(outstream);
  outstream = NULL;

  if(use_std_io){
    close(input_pipe);
    close(output_pipe);
  }
  
  return 0;

}
