#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_build_network_t {
	char* ms_cfg;
	int ms_cfg_length;
	char* ms_weights;
	int ms_weights_length;
} ms_ecall_build_network_t;

typedef struct ms_ecall_test_network_t {
	float* ms_image;
	float* ms_output;
	int ms_output_size;
	int ms_size_test_file;
	int ms_num_threads;
} ms_ecall_test_network_t;

typedef struct ms_ecall_thread_enter_enclave_waiting_t {
	int ms_thread_id;
} ms_ecall_thread_enter_enclave_waiting_t;

typedef struct ms_ocall_print_string_t {
	const char* ms_str;
} ms_ocall_print_string_t;

typedef struct ms_ocall_spawn_threads_t {
	int ms_n;
} ms_ocall_spawn_threads_t;

static sgx_status_t SGX_CDECL Enclave_ocall_print_string(void* pms)
{
	ms_ocall_print_string_t* ms = SGX_CAST(ms_ocall_print_string_t*, pms);
	ocall_print_string(ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_spawn_threads(void* pms)
{
	ms_ocall_spawn_threads_t* ms = SGX_CAST(ms_ocall_spawn_threads_t*, pms);
	ocall_spawn_threads(ms->ms_n);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[2];
} ocall_table_Enclave = {
	2,
	{
		(void*)Enclave_ocall_print_string,
		(void*)Enclave_ocall_spawn_threads,
	}
};
sgx_status_t ecall_build_network(sgx_enclave_id_t eid, char* cfg, int cfg_length, char* weights, int weights_length)
{
	sgx_status_t status;
	ms_ecall_build_network_t ms;
	ms.ms_cfg = cfg;
	ms.ms_cfg_length = cfg_length;
	ms.ms_weights = weights;
	ms.ms_weights_length = weights_length;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_test_network(sgx_enclave_id_t eid, float* image, float* output, int output_size, int size_test_file, int num_threads)
{
	sgx_status_t status;
	ms_ecall_test_network_t ms;
	ms.ms_image = image;
	ms.ms_output = output;
	ms.ms_output_size = output_size;
	ms.ms_size_test_file = size_test_file;
	ms.ms_num_threads = num_threads;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}
 
sgx_status_t ecall_thread_enter_enclave_waiting(sgx_enclave_id_t eid, int thread_id)
{
	sgx_status_t status;
	ms_ecall_thread_enter_enclave_waiting_t ms;
	ms.ms_thread_id = thread_id;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

 