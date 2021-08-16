#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */

#include "user_types.h"

#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

void ecall_build_network(char* cfg, int cfg_length, char* weights, int weights_length);
void ecall_test_network(float* image, float* output, int output_size, int size_test_file, int num_threads);
void ecall_thread_enter_enclave_waiting(int thread_id);

sgx_status_t SGX_CDECL ocall_print_string(const char* str);
sgx_status_t SGX_CDECL ocall_spawn_threads(int n);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
