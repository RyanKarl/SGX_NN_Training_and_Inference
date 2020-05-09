//App.cpp
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni
//g++ App/App.cpp Enclave/Enclave.cpp -pedantic -Wall  -O3 -o ./app -lm -DNENCLAVE

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

#include <iostream>

using std::cout;
using std::endl;

#ifndef NENCLAVE

# include "sgx_error.h"       /* sgx_status_t */
# include "sgx_eid.h"     /* sgx_enclave_id_t */
# include "sgx_urts.h"

# include "Enclave_u.h" //Enclave headers
# include "ocalls.h" //OCALL implementations

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
    if ( (!updated) || fp == NULL) {
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

#else
# include "../Enclave/Enclave.h"
#endif

#ifndef NENCLAVE
int SGX_CDECL main(int argc, char ** argv){
#else
int main(int argc, char ** argv){
#endif

  //Temporary method of randomness
#define SEED 5
  srand(SEED);


  //i and o are filenames of named pipe
  char * input_pipe_path = NULL;
  char * output_pipe_path = NULL;
  char * network_structure_fname = NULL;
  char * input_csv_filename = NULL;
  int verbose = 0;

  char c;
  while((c = getopt(argc, argv, "s:c:i:o:v")) != -1){
    switch(c){
      case 'v':{
        verbose += 1;
        break;
      }
      case 's':{
        network_structure_fname = optarg;
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
      case 'c':{
        input_csv_filename = optarg;
        break;
      }
      default:{
        fprintf(stderr, "ERROR: unrecognized argument %c\n", c);
        return 0;
      }
    }
  }
  
  if(verbose){
    cout << "Args parsed" << endl;
  }
  
#ifndef NENCLAVE  
  
  /* Initialize the enclave */
  if(initialize_enclave() < 0){
      fprintf(stderr, "Enclave initialization FAILED, exiting\n");
      return -1; 
  }
  else if (verbose){
    cout << "Successfully initialized enclave" << endl;
  }
  
#endif  

#ifdef NENCLAVE
  int enclave_result = enclave_main(network_structure_fname, input_csv_filename, input_pipe_path, output_pipe_path, verbose);
  
#else
  int enclave_result;
  sgx_enclave_id_t eid = global_eid;
  sgx_status_t sgx_enclave_stat = enclave_main(eid, &enclave_result, network_structure_fname, input_csv_filename, input_pipe_path, output_pipe_path, verbose); 
#endif  

  if(verbose){
    cout << "Enclave returned " << enclave_result << endl;
  }
  
    
  
  /* Destroy the enclave */
#ifndef NENCLAVE  
  sgx_destroy_enclave(global_eid);
#endif  
  
  
  
  return enclave_result;

}
