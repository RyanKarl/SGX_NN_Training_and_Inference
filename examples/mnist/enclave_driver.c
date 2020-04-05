//enclave_driver.c
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni
//gcc ./enclave_driver.c -pedantic -Wall -Werror -O3 -std=gnu99 -o enclave_driver -lm -DNENCLAVE

//Defined first to go before other standard libs
/*
#define malloc_consolidate(){\
fprintf(stdout, "%s %s \n", __FILE__, __LINE__);\
malloc_consolidate();\
}
*/

//Flag NENCLAVE used to compile without enclave things


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h> 
#include <string.h>
#include <errno.h>
#include <getopt.h>
#include <malloc.h>

#ifndef NENCLAVE

# include <pwd.h>
# define MAX_PATH FILENAME_MAX

//Enclave definitions
# define TOKEN_FILENAME   "enclave.token"
# define ENCLAVE_FILENAME "enclave.signed.so"

extern sgx_enclave_id_t global_eid;    /* global enclave id */

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }
    
    if (idx == ttl)
    	printf("Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer Reference\" for more details.\n", ret);
}

/* Initialize the enclave:
 *   Step 1: try to retrieve the launch token saved by last transaction
 *   Step 2: call sgx_create_enclave to initialize an enclave instance
 *   Step 3: save the launch token if it is updated
 */
int initialize_enclave(void)
{
    char token_path[MAX_PATH] = {'\0'};
    sgx_launch_token_t token = {0};
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    int updated = 0;
    
    /* Step 1: try to retrieve the launch token saved by last transaction 
     *         if there is no token, then create a new one.
     */
    /* try to get the token saved in $HOME */
    const char *home_dir = getpwuid(getuid())->pw_dir;
    
    if (home_dir != NULL && 
        (strlen(home_dir)+strlen("/")+sizeof(TOKEN_FILENAME)+1) <= MAX_PATH) {
        /* compose the token path */
        strncpy(token_path, home_dir, strlen(home_dir));
        strncat(token_path, "/", strlen("/"));
        strncat(token_path, TOKEN_FILENAME, sizeof(TOKEN_FILENAME)+1);
    } else {
        /* if token path is too long or $HOME is NULL */
        strncpy(token_path, TOKEN_FILENAME, sizeof(TOKEN_FILENAME));
    }

    FILE *fp = fopen(token_path, "rb");
    if (fp == NULL && (fp = fopen(token_path, "wb")) == NULL) {
        printf("Warning: Failed to create/open the launch token file \"%s\".\n", token_path);
    }

    if (fp != NULL) {
        /* read the token from saved file */
        size_t read_num = fread(token, 1, sizeof(sgx_launch_token_t), fp);
        if (read_num != 0 && read_num != sizeof(sgx_launch_token_t)) {
            /* if token is invalid, clear the buffer */
            memset(&token, 0x0, sizeof(sgx_launch_token_t));
            printf("Warning: Invalid launch token read from \"%s\".\n", token_path);
        }
    }
    /* Step 2: call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        if (fp != NULL) fclose(fp);
        return -1;
    }

    /* Step 3: save the launch token if it is updated */
    if (updated == FALSE || fp == NULL) {
        /* if the token is not updated, or file handler is invalid, do not perform saving */
        if (fp != NULL) fclose(fp);
        return 0;
    }

    /* reopen the file with write capablity */
    fp = freopen(token_path, "wb", fp);
    if (fp == NULL) return 0;
    size_t write_num = fwrite(token, 1, sizeof(sgx_launch_token_t), fp);
    if (write_num != sizeof(sgx_launch_token_t))
        printf("Warning: Failed to save launch token to \"%s\".\n", token_path);
    fclose(fp);
    return 0;
}

#endif

#include "enclave_functions.h"


#ifndef NENCLAVE
int SGX_CDECL main(int argc, char ** argv){
#else
int main(int argc, char ** argv){
#endif


  //i and o are filenames of named pipe
  //sleepy not implemented
  int sleepy = 0;
  int use_std_io = 0;
  char * input_pipe_path = NULL;
  char * output_pipe_path = NULL;
  int verbose = 0;

  char c;
  while((c = getopt(argc, argv, "sfi:o:vd")) != -1){
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
  
#ifndef NENCLAVE  
  
  /* Initialize the enclave */
  if (dummy){
    if(initialize_enclave() < 0){
        printf("Enter a character before exit ...\n");
        getchar();
        return -1; 
    }
  }
  
#endif  
    

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
    
    float * activation_data = NULL;
    int activation_n[MAT_DIM] = {0, 0};
    
    int enclave_retcode = verify_and_activate(input, matrix_n[0], matrix_n[1], matrix_n[2], (float **) &activation_data, activation_n);
    
    if(enclave_retcode){
    
      int frievald_result[MAT_DIM];
      frievald_result[0] = frievald_result[1] = -1;
      
      if(verbose){
        fprintf(stdout, "Frievald's algorithm and activation failed on round %d\n", round);
        fprintf(stdout, "Sending response: %d %d\n", frievald_result[0], frievald_result[1]);
        fflush(stdout);
      }

      if((!fwrite((int *) frievald_result, sizeof(frievald_result[0]), MAT_DIM, outstream)) || fflush(outstream)){
        fprintf(stderr, "ERROR: could not write failed verification\n");
        return 1;
      }
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
    if((activation_data != NULL) 
    && (activation_data < input) 
    && (activation_data >= input+num_in)){
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
  
  /* Destroy the enclave */
#ifndef NENCLAVE  
  sgx_destroy_enclave(global_eid);
#endif  
  
  
  
  return 0;

}
