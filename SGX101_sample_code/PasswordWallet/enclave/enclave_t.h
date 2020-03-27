#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */

#include "wallet.h"

#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

int ecall_create_wallet(const char* master_password);
int ecall_show_wallet(const char* master_password, wallet_t* wallet, size_t wallet_size);
int ecall_change_master_password(const char* old_password, const char* new_password);
int ecall_add_item(const char* master_password, const item_t* item, size_t item_size);
int ecall_remove_item(const char* master_password, int index);

sgx_status_t SGX_CDECL ocall_save_wallet(int* retval, const uint8_t* sealed_data, size_t sealed_size);
sgx_status_t SGX_CDECL ocall_load_wallet(int* retval, uint8_t* sealed_data, size_t sealed_size);
sgx_status_t SGX_CDECL ocall_is_wallet(int* retval);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
