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

static sgx_status_t SGX_CDECL sgx_ecall_build_network(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_build_network_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_build_network_t* ms = SGX_CAST(ms_ecall_build_network_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_cfg = ms->ms_cfg;
	int _tmp_cfg_length = ms->ms_cfg_length;
	size_t _len_cfg = _tmp_cfg_length * sizeof(char);
	char* _in_cfg = NULL;
	char* _tmp_weights = ms->ms_weights;
	int _tmp_weights_length = ms->ms_weights_length;
	size_t _len_weights = _tmp_weights_length * sizeof(char);
	char* _in_weights = NULL;

	if (sizeof(*_tmp_cfg) != 0 &&
		(size_t)_tmp_cfg_length > (SIZE_MAX / sizeof(*_tmp_cfg))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_weights) != 0 &&
		(size_t)_tmp_weights_length > (SIZE_MAX / sizeof(*_tmp_weights))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_cfg, _len_cfg);
	CHECK_UNIQUE_POINTER(_tmp_weights, _len_weights);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_cfg != NULL && _len_cfg != 0) {
		if ( _len_cfg % sizeof(*_tmp_cfg) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_cfg = (char*)malloc(_len_cfg);
		if (_in_cfg == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_cfg, _len_cfg, _tmp_cfg, _len_cfg)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_weights != NULL && _len_weights != 0) {
		if ( _len_weights % sizeof(*_tmp_weights) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_weights = (char*)malloc(_len_weights);
		if (_in_weights == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_weights, _len_weights, _tmp_weights, _len_weights)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ecall_build_network(_in_cfg, _tmp_cfg_length, _in_weights, _tmp_weights_length);

err:
	if (_in_cfg) free(_in_cfg);
	if (_in_weights) free(_in_weights);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_test_network(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_test_network_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_test_network_t* ms = SGX_CAST(ms_ecall_test_network_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_image = ms->ms_image;
	int _tmp_size_test_file = ms->ms_size_test_file;
	size_t _len_image = _tmp_size_test_file * sizeof(float);
	float* _in_image = NULL;
	float* _tmp_output = ms->ms_output;
	int _tmp_output_size = ms->ms_output_size;
	size_t _len_output = _tmp_output_size * sizeof(float);
	float* _in_output = NULL;

	if (sizeof(*_tmp_image) != 0 &&
		(size_t)_tmp_size_test_file > (SIZE_MAX / sizeof(*_tmp_image))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_output) != 0 &&
		(size_t)_tmp_output_size > (SIZE_MAX / sizeof(*_tmp_output))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_image, _len_image);
	CHECK_UNIQUE_POINTER(_tmp_output, _len_output);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_image != NULL && _len_image != 0) {
		if ( _len_image % sizeof(*_tmp_image) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_image = (float*)malloc(_len_image);
		if (_in_image == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_image, _len_image, _tmp_image, _len_image)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_output != NULL && _len_output != 0) {
		if ( _len_output % sizeof(*_tmp_output) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_output = (float*)malloc(_len_output)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_output, 0, _len_output);
	}

	ecall_test_network(_in_image, _in_output, _tmp_output_size, _tmp_size_test_file, ms->ms_num_threads);
	if (_in_output) {
		if (memcpy_s(_tmp_output, _len_output, _in_output, _len_output)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_image) free(_in_image);
	if (_in_output) free(_in_output);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_thread_enter_enclave_waiting(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_thread_enter_enclave_waiting_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_thread_enter_enclave_waiting_t* ms = SGX_CAST(ms_ecall_thread_enter_enclave_waiting_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	ecall_thread_enter_enclave_waiting(ms->ms_thread_id);


	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[3];
} g_ecall_table = {
	3,
	{
		{(void*)(uintptr_t)sgx_ecall_build_network, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_test_network, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_thread_enter_enclave_waiting, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[2][3];
} g_dyn_entry_table = {
	2,
	{
		{0, 0, 0, },
		{0, 0, 0, },
	}
};


sgx_status_t SGX_CDECL ocall_print_string(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_ocall_print_string_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_print_string_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(str, _len_str);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (str != NULL) ? _len_str : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_print_string_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_print_string_t));
	ocalloc_size -= sizeof(ms_ocall_print_string_t);

	if (str != NULL) {
		ms->ms_str = (const char*)__tmp;
		if (_len_str % sizeof(*str) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_s(__tmp, ocalloc_size, str, _len_str)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_str);
		ocalloc_size -= _len_str;
	} else {
		ms->ms_str = NULL;
	}
	
	status = sgx_ocall(0, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_spawn_threads(int n)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_spawn_threads_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_spawn_threads_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_spawn_threads_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_spawn_threads_t));
	ocalloc_size -= sizeof(ms_ocall_spawn_threads_t);

	ms->ms_n = n;
	status = sgx_ocall(1, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

