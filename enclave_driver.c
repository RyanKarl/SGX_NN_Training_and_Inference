//enclave_driver.c
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

//gcc ./enclave_driver.c -pedantic -Wall -Werror -O3 -o enclave_driver
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
//<sys/types.h> used for pipes
#include <sys/types.h>
#include <fcntl.h> 
#include <string.h>
#include <errno.h>
#include <getopt.h>

#include "enclave_functions.h"

//Assumes input file with num. entries on first line, and each entry being a space-separated pair of numbers of bytes
//Returns 0 on error, number of entries otherwise
unsigned int io_sizes(char * infile_name, unsigned int ** input_sizes, unsigned int ** output_sizes){
 //Initialize infile
  FILE * infile = NULL;
  if(infile_name != NULL){
    infile = fopen(infile_name, "r");
    if(infile == NULL){
      fprintf(stderr, "ERROR: failed opening %s\n", infile_name);
      return 1;
    }
  }

  unsigned int num_entries;
  if(fscanf(infile, "%u", &num_entries) != 1){
    fprintf(stderr, "ERROR: failed reading num. entries from %s\n", infile_name);
    return 0;
  }
  *input_sizes = (unsigned int *) malloc(num_entries*sizeof(**input_sizes));
  *output_sizes = (unsigned int *) malloc(num_entries*sizeof(**output_sizes));
  if(*input_sizes == NULL || *output_sizes == NULL){
    fprintf(stderr, "ERROR: cannot allocate memory\n");
    return 0;
  }
  for(unsigned int i = 0; i < num_entries; i++){
    int scanresult;
    scanresult = fscanf(infile, "%u", (*input_sizes)+i);
    if(scanresult == EOF){
      fprintf(stderr, "ERROR: EOF reached %u\n", i);
      return 0;
    }
    if(scanresult != 1){
      fprintf(stderr, "ERROR: entry %u %i\n", i, scanresult);
      return 0;
    }

    scanresult = fscanf(infile, "%u", (*output_sizes)+i);
    if(scanresult == EOF){
      fprintf(stderr, "ERROR: EOF reached %u\n", i);
      return 0;
    }
    if(scanresult != 1){
      fprintf(stderr, "ERROR: entry %u %i\n", i, scanresult);
      return 0;
    }
  }

  if(fclose(infile)){
    fprintf(stderr, "ERROR: could not close %s\n", infile_name);
    return 0;
  }
  
  free(infile);

  return num_entries;  
}

int main(int argc, char ** argv){

  //i and o are filenames of named pipe
  //sleepy not implemented
  int sleepy = 0;
  char * infile_name = NULL;
  char * input_pipe_path = NULL;
  char * output_pipe_path = NULL;
  int verbose = 0;

  char c;
  while((c = getopt(argc, argv, "sf:i:o:v")) != -1){
    switch(c){
      case 'v':{
        verbose = 1;
        break;
      }
      case 's':{
        sleepy = 1;
        break;
      }
      case 'f':{
        infile_name = optarg;
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

  
  int input_pipe = open(input_pipe_path, O_RDONLY);
  if(input_pipe == -1){
    fprintf(stderr, "ERROR: could not open input pipe %s\n", input_pipe_path);
    return 1;
  }
  int output_pipe = open(output_pipe_path, O_WRONLY);
  if(output_pipe == -1){
    fprintf(stderr, "ERROR: could not open output pipe %s\n", output_pipe_path);
    return 1;
  }

  //c file object which is wrapper for file descriptor
  FILE * instream = fdopen(input_pipe, "r");
  FILE * outstream = fdopen(output_pipe, "w");
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

  unsigned int * input_sizes;
  unsigned int * output_sizes;
  unsigned int num_entries;
  if(infile_name != NULL){    
    num_entries = io_sizes(infile_name, &input_sizes, &output_sizes);
    if(!num_entries){
      fprintf(stderr, "ERROR: reading I/O sizes from %s failed\n", infile_name);
      fflush(stderr);
      return 1;
    }
    
    //TODO implement file-based?
    fprintf(stdout, "ERROR: file-based I/O sizes not implemented!\n");
    return 0;
  }

  //Track how many rounds we go through
  unsigned int round = 0;
  while(1){
    //Shape of input data
    int data_shape[DATA_DIMENSIONS];
    //Get number of bytes either from file or from pipe
    if(infile_name != NULL){
      fprintf(stderr, "WARNING: file input not yet tested\n");
      fflush(stderr);
      //num_bytes_in = (int) input_sizes[round];
    }
    //Grab 3 dimensions of matrix
    else{
      if(verbose){
        fprintf(stdout, "About to get header (%li bytes)\n", sizeof(data_shape[0]) * DATA_DIMENSIONS);
        fflush(stdout);
      }
      
      if(!fread(&data_shape, sizeof(data_shape[0]), DATA_DIMENSIONS, instream)){
        fprintf(stderr, "ERROR: could not read header\n");
        fflush(stderr);
        return 1;
      }
    }
    
/*    if(verbose){
      fprintf(stdout, "Going to read %u bytes \n", (unsigned int) (DATA_DIMENSIONS * sizeof(data_shape[0])));
      fflush(stdout);
    }
*/    
    //Now we know how many bytes to receive
    //Read in data
    int num_in = 1;
    for(unsigned int i = 0; i < DATA_DIMENSIONS; i++){
      num_in *= data_shape[i];
    }
    
    //num_in should be product of matrix dimensions i.e. 3*4*5 = 60
    if(verbose){
      fprintf(stdout, "Data dimensions: ");
      for(unsigned int i = 0; i < DATA_DIMENSIONS; i++){
        fprintf(stdout, "%d ", data_shape[i]);
      }
      fprintf(stdout, "\n");
    }
    
    //Allocate array of floats for data
    float * input = malloc((unsigned int) num_in * sizeof(float));
    //Read in data from array to dynamic memory (note num_in is finite number of dimensions)
    if(!fread(input, sizeof(float), num_in, instream)){
      fprintf(stderr, "ERROR: could not read bytes\n");
      return 1;
    }

    if(verbose){
      fprintf(stderr, "Finished reading input from pipe\n");
    }
    
    //First verify data (TODO consider optimizing freivalds)
    if(verify_frievald(input, data_shape)){
      //Frievald's algorithm failed - send back -1
      int frievald_result[DATA_DIMENSIONS];
      for(unsigned int i = 0; i < DATA_DIMENSIONS; i++){
        frievald_result[i] = -1;
      }
      if(!fwrite(&frievald_result, sizeof(frievald_result[0]), DATA_DIMENSIONS, outstream)){
        fprintf(stderr, "ERROR: could not write failed verification\n");
        return 1;
      }
      if(verbose){
        fprintf(stdout, "Frievald's algorithm failed on round %d, ending program\n", round);
      }
      break;
    }
    else{
      //Now compute activation
      float * activation_data = NULL;
      int * activation_shape;
      if(activate(input, data_shape, &activation_data, &activation_shape)){
        fprintf(stderr, "ERROR: activation failed on round %d\n", round);
        return 1;
      }
      int activated_data_size = 1;
      //May need to account for conv dimensions
      for(unsigned int i = 0; i < DATA_DIMENSIONS; i++){
        activated_data_size *= activation_shape[i];
      }
      
      //Send header with length
      if(!fwrite(activation_shape, sizeof(activation_shape[0]), DATA_DIMENSIONS, outstream)){
        fprintf(stderr, "ERROR: could not write header\n");
        return 1;
      }
      
      if(verbose){
        fprintf(stdout, "Sent output header: %d bytes\n", (int) sizeof(activation_shape[0])*DATA_DIMENSIONS);
        fprintf(stdout, "Sending %d float outputs: ", activated_data_size);
        for(int i = 0; i < activated_data_size; i++){
          fprintf(stdout, "%f ", activation_data[i]);
        }
        fprintf(stdout, "\n");
      }
      
      //Now send actual data
      if((!fwrite(activation_data, sizeof(activation_data[0]), activated_data_size, outstream)) || fflush(outstream)){
        fprintf(stderr, "ERROR: could not write out\n");
        return 1;
      }
      //Clean up memory (this is the data i.e. activation data)
      free(input);
      if(activation_data != NULL){
        free(activation_data);
      }
      if(verbose){
        fprintf(stdout, "Finished writing\n");
      }
      
    }

    round++;
  }

  //Close the given pipes, clean up memory
  
  fclose(instream);
  fclose(outstream);
  free(instream);
  free(outstream);

  close(input_pipe);
  close(output_pipe);

  if(input_sizes){
    free(input_sizes);
  }
  if(output_sizes){
    free(output_sizes);
  }
  
  return 0;

}
