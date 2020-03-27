#include "enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_create_wallet_t {
	int ms_retval;
	const char* ms_master_password;
	size_t ms_master_password_len;
} ms_ecall_create_wallet_t;

typedef struct ms_ecall_show_wallet_t {
	int ms_retval;
	const char* ms_master_password;
	size_t ms_master_password_len;
	wallet_t* ms_wallet;
	size_t ms_wallet_size;
} ms_ecall_show_wallet_t;

typedef struct ms_ecall_change_master_password_t {
	int ms_retval;
	const char* ms_old_password;
	size_t ms_old_password_len;
	const char* ms_new_password;
	size_t ms_new_password_len;
} ms_ecall_change_master_password_t;

typedef struct ms_ecall_add_item_t {
	int ms_retval;
	const char* ms_master_password;
	size_t ms_master_password_len;
	const item_t* ms_item;
	size_t ms_item_size;
} ms_ecall_add_item_t;

typedef struct ms_ecall_remove_item_t {
	int ms_retval;
	const char* ms_master_password;
	size_t ms_master_password_len;
	int ms_index;
} ms_ecall_remove_item_t;

typedef struct ms_ocall_save_wallet_t {
	int ms_retval;
	const uint8_t* ms_sealed_data;
	size_t ms_sealed_size;
} ms_ocall_save_wallet_t;

typedef struct ms_ocall_load_wallet_t {
	int ms_retval;
	uint8_t* ms_sealed_data;
	size_t ms_sealed_size;
} ms_ocall_load_wallet_t;

typedef struct ms_ocall_is_wallet_t {
	int ms_retval;
} ms_ocall_is_wallet_t;

static sgx_status_t SGX_CDECL enclave_ocall_save_wallet(void* pms)
{
	ms_ocall_save_wallet_t* ms = SGX_CAST(ms_ocall_save_wallet_t*, pms);
	ms->ms_retval = ocall_save_wallet(ms->ms_sealed_data, ms->ms_sealed_size);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL enclave_ocall_load_wallet(void* pms)
{
	ms_ocall_load_wallet_t* ms = SGX_CAST(ms_ocall_load_wallet_t*, pms);
	ms->ms_retval = ocall_load_wallet(ms->ms_sealed_data, ms->ms_sealed_size);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL enclave_ocall_is_wallet(void* pms)
{
	ms_ocall_is_wallet_t* ms = SGX_CAST(ms_ocall_is_wallet_t*, pms);
	ms->ms_retval = ocall_is_wallet();

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[3];
} ocall_table_enclave = {
	3,
	{
		(void*)enclave_ocall_save_wallet,
		(void*)enclave_ocall_load_wallet,
		(void*)enclave_ocall_is_wallet,
	}
};
sgx_status_t ecall_create_wallet(sgx_enclave_id_t eid, int* retval, const char* master_password)
{
	sgx_status_t status;
	ms_ecall_create_wallet_t ms;
	ms.ms_master_password = master_password;
	ms.ms_master_password_len = master_password ? strlen(master_password) + 1 : 0;
	status = sgx_ecall(eid, 0, &ocall_table_enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_show_wallet(sgx_enclave_id_t eid, int* retval, const char* master_password, wallet_t* wallet, size_t wallet_size)
{
	sgx_status_t status;
	ms_ecall_show_wallet_t ms;
	ms.ms_master_password = master_password;
	ms.ms_master_password_len = master_password ? strlen(master_password) + 1 : 0;
	ms.ms_wallet = wallet;
	ms.ms_wallet_size = wallet_size;
	status = sgx_ecall(eid, 1, &ocall_table_enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_change_master_password(sgx_enclave_id_t eid, int* retval, const char* old_password, const char* new_password)
{
	sgx_status_t status;
	ms_ecall_change_master_password_t ms;
	ms.ms_old_password = old_password;
	ms.ms_old_password_len = old_password ? strlen(old_password) + 1 : 0;
	ms.ms_new_password = new_password;
	ms.ms_new_password_len = new_password ? strlen(new_password) + 1 : 0;
	status = sgx_ecall(eid, 2, &ocall_table_enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_add_item(sgx_enclave_id_t eid, int* retval, const char* master_password, const item_t* item, size_t item_size)
{
	sgx_status_t status;
	ms_ecall_add_item_t ms;
	ms.ms_master_password = master_password;
	ms.ms_master_password_len = master_password ? strlen(master_password) + 1 : 0;
	ms.ms_item = item;
	ms.ms_item_size = item_size;
	status = sgx_ecall(eid, 3, &ocall_table_enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t ecall_remove_item(sgx_enclave_id_t eid, int* retval, const char* master_password, int index)
{
	sgx_status_t status;
	ms_ecall_remove_item_t ms;
	ms.ms_master_password = master_password;
	ms.ms_master_password_len = master_password ? strlen(master_password) + 1 : 0;
	ms.ms_index = index;
	status = sgx_ecall(eid, 4, &ocall_table_enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

