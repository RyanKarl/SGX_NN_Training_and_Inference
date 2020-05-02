#include "Enclave_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define ADD_ASSIGN_OVERFLOW(a, b) (	\
	((a) += (b)) < (b)	\
)


typedef struct ms_verify_and_activate_t {
	int ms_retval;
	float* ms_data_in;
	int ms_a_height;
	int ms_a_width;
	int ms_b_height;
	int ms_b_width;
	int ms_c_height;
	int ms_c_width;
	float* ms_data_out;
	int ms_out_height;
	int ms_out_width;
} ms_verify_and_activate_t;

static sgx_status_t SGX_CDECL sgx_verify_and_activate(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_verify_and_activate_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_verify_and_activate_t* ms = SGX_CAST(ms_verify_and_activate_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_data_in = ms->ms_data_in;
	float* _tmp_data_out = ms->ms_data_out;



	ms->ms_retval = verify_and_activate(_tmp_data_in, ms->ms_a_height, ms->ms_a_width, ms->ms_b_height, ms->ms_b_width, ms->ms_c_height, ms->ms_c_width, _tmp_data_out, ms->ms_out_height, ms->ms_out_width);


	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[1];
} g_ecall_table = {
	1,
	{
		{(void*)(uintptr_t)sgx_verify_and_activate, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
} g_dyn_entry_table = {
	0,
};


