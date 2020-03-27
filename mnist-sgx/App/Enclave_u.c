#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_input_t {
	int ms_retval;
	char* ms_data;
} ms_input_t;

typedef struct ms_learning_process_t {
	int ms_retval;
} ms_learning_process_t;

typedef struct ms_square_error_t {
	double ms_retval;
} ms_square_error_t;

typedef struct ms_predict_t {
	int ms_retval;
} ms_predict_t;

typedef struct ms_decryptMessage_t {
	char* ms_encMessageIn;
	size_t ms_len;
	char* ms_decMessageOut;
	size_t ms_lenOut;
} ms_decryptMessage_t;

typedef struct ms_encryptMessage_t {
	char* ms_decMessageIn;
	size_t ms_len;
	char* ms_encMessageOut;
	size_t ms_lenOut;
} ms_encryptMessage_t;

typedef struct ms_print_error_message_t {
	sgx_status_t ms_ret;
} ms_print_error_message_t;

typedef struct ms_ocall_print_string_t {
	const char* ms_str;
} ms_ocall_print_string_t;

typedef struct ms_ocall_print_num_t {
	uint32_t ms_num;
} ms_ocall_print_num_t;

typedef struct ms_ocall_read_image_t {
	char* ms_number;
} ms_ocall_read_image_t;

typedef struct ms_ocall_read_label_t {
	char* ms_number;
} ms_ocall_read_label_t;

typedef struct ms_ocall_save_nn_t {
	int ms_retval;
	const uint8_t* ms_sealed_data;
	size_t ms_sealed_size;
} ms_ocall_save_nn_t;

typedef struct ms_ocall_load_nn_t {
	int ms_retval;
	uint8_t* ms_sealed_data;
	size_t ms_sealed_size;
} ms_ocall_load_nn_t;

static sgx_status_t SGX_CDECL Enclave_print_error_message(void* pms)
{
	ms_print_error_message_t* ms = SGX_CAST(ms_print_error_message_t*, pms);
	print_error_message(ms->ms_ret);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_print_string(void* pms)
{
	ms_ocall_print_string_t* ms = SGX_CAST(ms_ocall_print_string_t*, pms);
	ocall_print_string(ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_print_num(void* pms)
{
	ms_ocall_print_num_t* ms = SGX_CAST(ms_ocall_print_num_t*, pms);
	ocall_print_num(ms->ms_num);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_read_image(void* pms)
{
	ms_ocall_read_image_t* ms = SGX_CAST(ms_ocall_read_image_t*, pms);
	ocall_read_image(ms->ms_number);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_read_label(void* pms)
{
	ms_ocall_read_label_t* ms = SGX_CAST(ms_ocall_read_label_t*, pms);
	ocall_read_label(ms->ms_number);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_save_nn(void* pms)
{
	ms_ocall_save_nn_t* ms = SGX_CAST(ms_ocall_save_nn_t*, pms);
	ms->ms_retval = ocall_save_nn(ms->ms_sealed_data, ms->ms_sealed_size);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_load_nn(void* pms)
{
	ms_ocall_load_nn_t* ms = SGX_CAST(ms_ocall_load_nn_t*, pms);
	ms->ms_retval = ocall_load_nn(ms->ms_sealed_data, ms->ms_sealed_size);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[7];
} ocall_table_Enclave = {
	7,
	{
		(void*)Enclave_print_error_message,
		(void*)Enclave_ocall_print_string,
		(void*)Enclave_ocall_print_num,
		(void*)Enclave_ocall_read_image,
		(void*)Enclave_ocall_read_label,
		(void*)Enclave_ocall_save_nn,
		(void*)Enclave_ocall_load_nn,
	}
};
sgx_status_t printf_helloworld(sgx_enclave_id_t eid)
{
	sgx_status_t status;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, NULL);
	return status;
}

sgx_status_t init_array(sgx_enclave_id_t eid)
{
	sgx_status_t status;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, NULL);
	return status;
}

sgx_status_t input(sgx_enclave_id_t eid, int* retval, char* data)
{
	sgx_status_t status;
	ms_input_t ms;
	ms.ms_data = data;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t learning_process(sgx_enclave_id_t eid, int* retval)
{
	sgx_status_t status;
	ms_learning_process_t ms;
	status = sgx_ecall(eid, 3, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t square_error(sgx_enclave_id_t eid, double* retval)
{
	sgx_status_t status;
	ms_square_error_t ms;
	status = sgx_ecall(eid, 4, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t save_nn(sgx_enclave_id_t eid)
{
	sgx_status_t status;
	status = sgx_ecall(eid, 5, &ocall_table_Enclave, NULL);
	return status;
}

sgx_status_t load_model(sgx_enclave_id_t eid)
{
	sgx_status_t status;
	status = sgx_ecall(eid, 6, &ocall_table_Enclave, NULL);
	return status;
}

sgx_status_t predict(sgx_enclave_id_t eid, int* retval)
{
	sgx_status_t status;
	ms_predict_t ms;
	status = sgx_ecall(eid, 7, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t decryptMessage(sgx_enclave_id_t eid, char* encMessageIn, size_t len, char* decMessageOut, size_t lenOut)
{
	sgx_status_t status;
	ms_decryptMessage_t ms;
	ms.ms_encMessageIn = encMessageIn;
	ms.ms_len = len;
	ms.ms_decMessageOut = decMessageOut;
	ms.ms_lenOut = lenOut;
	status = sgx_ecall(eid, 8, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t encryptMessage(sgx_enclave_id_t eid, char* decMessageIn, size_t len, char* encMessageOut, size_t lenOut)
{
	sgx_status_t status;
	ms_encryptMessage_t ms;
	ms.ms_decMessageIn = decMessageIn;
	ms.ms_len = len;
	ms.ms_encMessageOut = encMessageOut;
	ms.ms_lenOut = lenOut;
	status = sgx_ecall(eid, 9, &ocall_table_Enclave, &ms);
	return status;
}

