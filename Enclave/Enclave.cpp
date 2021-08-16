/*
 * Copyright (C) 2011-2019 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "Enclave.h"
#include "Enclave_t.h" /* print_string */
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
extern "C"
{
#include "TrustedLibrary/custom-darknet/src/standard.h"
}

int g_num_threads;
sgx_spinlock_t *g_spin_locks;
gemm_args *g_gemm_args_pointer;
volatile int *g_finished;

network *final_net;

void printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
}

void free_data(data d)
{
    if (!d.shallow)
    {
        free_matrix(d.X);
        free_matrix(d.y);
    }
    else
    {
        free(d.X.vals);
        free(d.y.vals);
    }
}

void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = (float *)calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void read_bytes(void *ptr, size_t size, size_t nmemb, char* input, int *offset) {
    memcpy(ptr, input + *offset, size * nmemb);
    *offset +=  size * nmemb;
}

void load_connected_weights(layer l, char *file_bytes, int *offset)
{
    read_bytes(l.biases, sizeof(float), l.outputs, file_bytes, offset);
    read_bytes(l.weights, sizeof(float), l.outputs*l.inputs, file_bytes, offset);
    if (l.batch_normalize && (!l.dontloadscales)){
        read_bytes(l.scales, sizeof(float), l.outputs, file_bytes, offset);
        read_bytes(l.rolling_mean, sizeof(float), l.outputs, file_bytes, offset);
        read_bytes(l.rolling_variance, sizeof(float), l.outputs, file_bytes, offset);
    }
}

void load_convolutional_weights(layer l, char *file_bytes, int *offset)
{
    int num = l.c/l.groups*l.n*l.size*l.size;
    read_bytes(l.biases, sizeof(float), l.n, file_bytes, offset);
    if (l.batch_normalize && (!l.dontloadscales)){
        read_bytes(l.scales, sizeof(float), l.n, file_bytes, offset);
        read_bytes(l.rolling_mean, sizeof(float), l.n, file_bytes, offset);
        read_bytes(l.rolling_variance, sizeof(float), l.n, file_bytes, offset);
    }
    read_bytes(l.weights, sizeof(float), num, file_bytes, offset);
    if (l.flipped) {
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }
}

void load_weights(network *net, char *file_bytes)
{
    int major = 0;
    int minor = 0;
    int revision = 0;

    int offset = 0;

    int i;
    for(i = 0; i < net->n; ++i){
       layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            // printf("load -------------------------------1\n");
            load_convolutional_weights(l, file_bytes, &offset);
            // printf("load over-------------------------------1\n");
        }
        if(l.type == CONNECTED){
            // printf("load -------------------------------1\n");
            load_connected_weights(l, file_bytes, &offset);
        }
        // printf("layer:%d; end offset is:%d;\n", i, offset);
    }
}

void ecall_build_network(char *cfg, int cfg_length, char *weights, int weights_length) {

    network *net = (network *)malloc(sizeof(network));
    list *sections = sgx_file_string_to_list(cfg);

    net = sgx_parse_network_cfg(sections);


    // printf("build over -------------------------------2\n");
    free_list(sections);
    
    if (weights) {
        load_weights(net, weights);
    }


    final_net = net;
    // free_network(final_net);
}

void ecall_thread_enter_enclave_waiting(int thread_id)
{
    while (1) {

        sgx_spin_lock(&g_spin_locks[thread_id]);
        // printf("Thread %d processing (%d,%d) i_start=%d, M=%d, N=%d, K=%d, A[0]=%f, A[999]=%f, B[0]=%f, B[9999]=%f, C[0]=%f\n", thread_id, g_gemm_args_pointer[thread_id].TA, g_gemm_args_pointer[thread_id].TB, g_gemm_args_pointer[thread_id].i_start,  g_gemm_args_pointer[thread_id].M, g_gemm_args_pointer[thread_id].N, g_gemm_args_pointer[thread_id].K,  g_gemm_args_pointer[thread_id].A[0], g_gemm_args_pointer[thread_id].A[999],  g_gemm_args_pointer[thread_id].B[0], g_gemm_args_pointer[thread_id].B[9999], g_gemm_args_pointer[thread_id].C[0]);
        // ocall_start_measuring_training(thread_id+3, 10);
        gemm_cpu(
            g_gemm_args_pointer[thread_id].TA,
            g_gemm_args_pointer[thread_id].TB,
            g_gemm_args_pointer[thread_id].i_start,
            g_gemm_args_pointer[thread_id].M,
            g_gemm_args_pointer[thread_id].N,
            g_gemm_args_pointer[thread_id].K,
            g_gemm_args_pointer[thread_id].ALPHA, 
            g_gemm_args_pointer[thread_id].A,
            g_gemm_args_pointer[thread_id].lda, 
            g_gemm_args_pointer[thread_id].B, 
            g_gemm_args_pointer[thread_id].ldb,
            g_gemm_args_pointer[thread_id].BETA,
            g_gemm_args_pointer[thread_id].C, 
            g_gemm_args_pointer[thread_id].ldc);
        g_finished[thread_id] = 1;
        sgx_spin_unlock(&g_spin_locks[thread_id]);
        while (g_finished[thread_id] == 1){
            if (g_num_threads == 0) {
                break;
            }
        }

    
        if (g_num_threads == 0) {
            g_finished[thread_id] = 2;
            break;
        }
        
    }

}

void ecall_test_network(float *image, float *output, int output_size, int size_test_file, int num_threads) {
    
    
    g_num_threads = num_threads;
    g_spin_locks = (sgx_spinlock_t *)calloc(g_num_threads, sizeof(sgx_spinlock_t));
    g_gemm_args_pointer = (gemm_args *)calloc(g_num_threads, sizeof(gemm_args));
    g_finished = (volatile int *)calloc(g_num_threads, sizeof(int));


    // printf("inference -------------------------------1\n");
    int i;
    for (i = 0; i < g_num_threads; i++)
        sgx_spin_lock(&g_spin_locks[i]);
    
    ocall_spawn_threads(g_num_threads);

    // printf("inference -------------------------------2\n");

    float *out = network_predict(final_net, image);

    // printf("length is: %d \n", sizeof(out));
    for (i = 0; i < g_num_threads; i++){
        sgx_spin_unlock(&g_spin_locks[i]);
    }
        // sgx_spin_unlock(&g_spin_locks[i]);
        // sgx_spin_lock(&g_spin_locks[i]);
        
        // sgx_spin_unlock(&g_spin_locks[i]);

    g_num_threads = 0;

    while(1){
        int finished = 0;
        for (i = 0; i < num_threads; i++){
            if (g_finished[i] != 2){
                finished = finished + 1;
            };
        }
        if (finished == 0){
            break;
        }
    }


    // printf("result is: %f \n", out[1]);
    memcpy(output, out, output_size);
    // g_num_threads = 0;
    free_network(final_net);
}