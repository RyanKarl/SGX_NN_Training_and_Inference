#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */

#include "../App/App.h"

#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

void printf_helloworld(void);
void init_array(void);
int input(char* data);
int learning_process(void);
double square_error(void);
void save_nn(void);
void load_model(void);
int predict(void);
void decryptMessage(char* encMessageIn, size_t len, char* decMessageOut, size_t lenOut);
void encryptMessage(char* decMessageIn, size_t len, char* encMessageOut, size_t lenOut);

sgx_status_t SGX_CDECL print_error_message(sgx_status_t ret);
sgx_status_t SGX_CDECL ocall_print_string(const char* str);
sgx_status_t SGX_CDECL ocall_print_num(uint32_t num);
sgx_status_t SGX_CDECL ocall_read_image(char* number);
sgx_status_t SGX_CDECL ocall_read_label(char* number);
sgx_status_t SGX_CDECL ocall_save_nn(int* retval, const uint8_t* sealed_data, size_t sealed_size);
sgx_status_t SGX_CDECL ocall_load_nn(int* retval, uint8_t* sealed_data, size_t sealed_size);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
