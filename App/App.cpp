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

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <assert.h>

#include <pthread.h>
#include <fstream>
#include <time.h>
#include <sys/time.h>
#include <stdarg.h>

# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX
#include "Image.cpp"
#include "utils.cpp"
#include <vector>

#include <math.h>

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"


int interval_duration = 1;

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

struct Group
{
    std::vector<int>layer_group;
};

struct Policy
{
    std::vector<Group>partition_policy;
};


struct NN
{
    float *image;
    char *weights;
    int image_size;
    Policy policy;
    std::vector<char*> layer_cfg;
    std::vector<int> layer_outputs;
    std::vector<int*> layer_weights_offsets;
    std::vector<int*> layer_shapes;
};



/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }
    
    if (idx == ttl)
    	printf("Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer Reference\" for more details.\n", ret);
}

long clockFilter(timespec ts, timespec tf){
    long startTime, endTime, duration;
    startTime = (long)ts.tv_sec * 1000 * 1000 + (long)ts.tv_nsec / 1000;
    endTime = (long)tf.tv_sec * 1000 * 1000 + (long)tf.tv_nsec / 1000;
    duration = endTime - startTime;
    return duration;
}

void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}

/* Initialize the enclave:
 *   Call sgx_create_enclave to initialize an enclave instance
 */
int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    
    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }

    return 0;
}

void *enter_enclave_waiting(void *thread_id_void)
{
    int thread_id = *((int *)thread_id_void);
    ecall_thread_enter_enclave_waiting(global_eid, thread_id);
}

void ocall_spawn_threads(int n)
{
    pthread_t *threads = (pthread_t *)calloc(n, sizeof(pthread_t));

    int i;
    for (i = 0; i < n; i++)
    {
        int *pass_i = (int *)malloc(sizeof(*pass_i));
        *pass_i = i;
        if (pthread_create(&threads[i], 0, enter_enclave_waiting, (void *)pass_i))
            printf("Thread creation failed");
    }
}



int layers[] = {14, 27, 69, 138, 205, 205, 25, 107};

int get_layers(char *model_name){
    int layer_length = 0;
    if (strcmp("alexnet", model_name) == 0){
        return layers[0];
    }
    if (strcmp("googlenet", model_name) == 0){
        return layers[1];
    }
    if (strcmp("resnet50", model_name) == 0){
        return layers[2];
    }
    if (strcmp("resnet101", model_name) == 0){
        return layers[3];
    }
    if (strcmp("resnet152", model_name) == 0){
        return layers[4];
    }
    if (strcmp("resnext152", model_name) == 0){
        return layers[5];
    }
    if (strcmp("vgg-16", model_name) == 0){
        return layers[6];
    }
    if (strcmp("yolov3", model_name) == 0){
        return layers[7];
    }
}

void generate_test_partition(char* model_name, Policy *policy){

    int layer_length = get_layers(model_name);

    int interval = interval_duration;

    for (int i = 0; i < ((layer_length - 1) /interval) + 1; i++)
    {   
        Group group;
        int index = i * interval;
        for (int j = 0; j < interval; j++)
        {
            if (index + j + 1 <= layer_length){
                group.layer_group.push_back(index + j + 1);
            }
        }
        policy->partition_policy.push_back(group);
    } 

    // printf("partition layer size is: %d \n");
}

void generate_nn(char *net_cfg, char *weights, char *weights_cfg, char *outputs_cfg_str, char *layer_shapes_str, NN *nn, char *model_name){
    
    int layer_num = get_layers(model_name);

    // std::vector<char*>layer_weights;
    weights_partition(weights_cfg, nn->layer_weights_offsets, layer_num);

    // std::vector<char*>layer_cfg;
    cfg_partition(net_cfg, nn->layer_cfg);

    // printf("------------------------ 1\n");
    // std::vector<int>layer_outputs;
    outputs_partition(outputs_cfg_str, nn->layer_outputs);

    // printf("------------------------ 2\n");
    // Policy policy;
    generate_test_partition(model_name, &nn->policy);

    // printf("------------------------ 3\n");
    // layer shapes
    get_layers_shape(layer_shapes_str, nn->layer_shapes, layer_num);
    // printf("------------------------ 4\n");
}

void partition_scheduling(NN *nn){

    if(initialize_enclave() < 0){
        printf("Something error in enclave inilialization ...\n");
        exit(0); 
    }

    float *input = nn->image;
    int inputs = nn->image_size;

    timespec ts, tf;
    long init_duration = 0;
    long inference_duration = 0;
    
    printf("Total layer is: %d \n", nn->policy.partition_policy.size());

    for (int i = 0; i < nn->policy.partition_policy.size(); i++)
    {   
        // int inputss = input_length /sizeof(float);
        printf("--------------------Layer %d--------------------\n", i+1);

        char *group_cfg = cfg_filter(nn->policy.partition_policy[i].layer_group, nn->layer_cfg, nn->layer_shapes);
        
        // printf("net cfg is: %s\n", group_cfg);
        char *group_weights = weights_filter(nn->policy.partition_policy[i].layer_group, nn->layer_weights_offsets, nn->weights);
        // printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
        int cfg_length = sizeof(char) * strlen(group_cfg);

        int s_offset = nn->layer_weights_offsets[nn->policy.partition_policy[i].layer_group[0] - 1][0];
        int e_offset = nn->layer_weights_offsets[nn->policy.partition_policy[i].layer_group[nn->policy.partition_policy[i].layer_group.size() -1] - 1][1];
        int weights_length = e_offset - s_offset;

        // 

        // printf("weights is: %s\n", group_weights);

        printf("begin build network\n");
        clock_gettime(CLOCK_REALTIME,&ts);
        sgx_status_t status = ecall_build_network(global_eid, group_cfg, cfg_length + 1, group_weights, weights_length + 1);
        clock_gettime(CLOCK_REALTIME,&tf);
        init_duration = init_duration + clockFilter(ts, tf);
        printf("finish build network\n");

        int outputs = get_outputs(nn->policy.partition_policy[i].layer_group, nn->layer_outputs);
        float* output = (float*)malloc(sizeof(float) * outputs);

        printf("begin inference\n");
        clock_gettime(CLOCK_REALTIME,&ts);
        status = ecall_test_network(global_eid, input, output, outputs, inputs, 3);
        clock_gettime(CLOCK_REALTIME,&tf);
        inference_duration = inference_duration + clockFilter(ts, tf);
        printf("end inference \n");

        free(input);
        free(group_cfg);
        free(group_weights);

        input = (float*)malloc(sizeof(float) * outputs);
        memcpy(input, output, sizeof(float) * outputs);
        inputs = outputs;
        free(output);
    }
    printf("Init time consumption is: %ld us \n", init_duration);
    printf("Inference time consumption is: %ld us \n", inference_duration);
    sgx_destroy_enclave(global_eid);
}


/* Application entry */
int SGX_CDECL main(int argc, char *argv[]){
    // (void)(argc);
    // (void)(argv);

    char *cfgFile = argv[1];
    char *weightFile = argv[2];
    char *imageFile = argv[3];
    char *w_cfg = argv[4];
    char *outputs_cfg_file = argv[5];
    char *layer_shapes = argv[6];

    // Reading model config file
    FILE *f = fopen(cfgFile, "rb");
    fseek(f, 0L, SEEK_END);
    
    int cfg_length = ftell(f);
    rewind(f);
    
    char *net_cfg = (char *)malloc(cfg_length + 1);

    fread(net_cfg, cfg_length, 1, f);
    fclose(f); 


    // Reading the weights file
    f = fopen(weightFile, "rb");
    fseek(f, 0L, SEEK_END);
    int weights_length = ftell(f);
    rewind(f);
    
    char *weights = (char *)malloc(weights_length + 1);
    
    fread(weights, weights_length, 1, f);
    fclose(f);
    // printf("weights length is: %d \n", weights_length);

    //Reading images
    char *input = imageFile;
    int w, h;
    sscanf(argv[7], "%d", &w);
    sscanf(argv[8], "%d", &h);
    image im = load_image_color(input,0,0);
    image sized = letterbox_image(im, w, h);
    int image_length = int(w * h * im.c);
    
    //reading weights config
    f = fopen(w_cfg, "rb");
    fseek(f, 0L, SEEK_END);
    int weights_cfg_length = ftell(f);
    rewind(f);
    
    char *weights_cfg = (char *)malloc(weights_cfg_length + 1);
    
    fread(weights_cfg, weights_cfg_length, 1, f);
    fclose(f);

    //Reading outputs
    f = fopen(outputs_cfg_file, "rb");
    fseek(f, 0L, SEEK_END);
    int outputs_cfg_length = ftell(f);
    rewind(f);
    
    char *outputs_cfg_str = (char *)malloc(outputs_cfg_length + 1);
    
    fread(outputs_cfg_str, outputs_cfg_length, 1, f);
    fclose(f);

    // Reading shapes
    f = fopen(layer_shapes, "rb");
    fseek(f, 0L, SEEK_END);
    int layer_shapes_length = ftell(f);
    rewind(f);
    
    char *layer_shapes_str = (char *)malloc(layer_shapes_length + 1);
    
    fread(layer_shapes_str, layer_shapes_length, 1, f);
    fclose(f);


    interval_duration = atoi(argv[10]);

    char *model_name = argv[9];
    NN nn;
    nn.image = sized.data;
    nn.image_size = image_length;
    nn.weights = weights;
    generate_nn(net_cfg, weights, weights_cfg, outputs_cfg_str, layer_shapes_str, &nn, model_name);

    partition_scheduling(&nn);

    return 0;
}

