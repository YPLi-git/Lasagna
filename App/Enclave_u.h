#ifndef ENCLAVE_U_H__
#define ENCLAVE_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "sgx_edger8r.h" /* for sgx_status_t etc. */

#include "user_types.h"

#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OCALL_PRINT_STRING_DEFINED__
#define OCALL_PRINT_STRING_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_print_string, (const char* str));
#endif
#ifndef OCALL_SPAWN_THREADS_DEFINED__
#define OCALL_SPAWN_THREADS_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_spawn_threads, (int n));
#endif

sgx_status_t ecall_build_network(sgx_enclave_id_t eid, char* cfg, int cfg_length, char* weights, int weights_length);
sgx_status_t ecall_test_network(sgx_enclave_id_t eid, float* image, float* output, int output_size, int size_test_file, int num_threads);
sgx_status_t ecall_thread_enter_enclave_waiting(sgx_enclave_id_t eid, int thread_id);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
