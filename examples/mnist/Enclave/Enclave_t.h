#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

int verify_and_activate(float* data_in, int a_height, int a_width, int b_height, int b_width, int c_height, int c_width, float* data_out, int out_height, int out_width);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
