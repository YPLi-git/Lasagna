#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <vector>

void outputs_partition(char* outputs, std::vector<int> &result){

    char *line = strtok(outputs, "\n");
    while(line){
        int data = atoi(line);
        result.push_back(data);
        line = strtok(NULL, "\n");
    }
}

int get_outputs(std::vector<int> &layers, std::vector<int> &outputs){
    
    int index = layers[layers.size() - 1] - 1;
    return outputs[index];
    
}

void get_layers_shape(char* file_string, std::vector<int*> &result, int layer_num){

    // printf("------------------------ 1.1\n");
    char* data = (char *)malloc(sizeof(char) * strlen(file_string) + 1); 
    sprintf(data, "%s", file_string);
    char* line = strtok(data, "\n");

    // printf("------------------------ 1.2\n");
    for (int i = 0; i < layer_num; i++)
    {
        // printf("------------------------ 1.3\n");
        
        int h = atoi(line);
        // printf("%s\n", line);
        
        
        line = strtok(NULL, "\n");
        int w = atoi(line);
        // printf("%s\n", line);


        line = strtok(NULL, "\n");
        int c = atoi(line);
        // printf("%s\n", line);

        int *shape = (int *)malloc(3 * sizeof(int));
        shape[0] = h;
        shape[1] = w;
        shape[2] = c;
        result.push_back(shape);

        line = strtok(NULL, "\n");
    }

}

void cfg_partition(char* file_string, std::vector<char*> &result){
    
    
    char* data = (char *)malloc(sizeof(char) * strlen(file_string) + 1); 
    sprintf(data, "%s", file_string);

    char *line = strtok(data, "[");
    char *net = (char *)malloc(sizeof(char) * (strlen(line) + 1)  + 1); 
    sprintf(net, "[%s", line);
    
    result.push_back(net);
 
    while(line){
        line = strtok(NULL, "[");
        if (line){
            char *layer = (char *)malloc(sizeof(char) * (strlen(line) + 1) + 1);
            sprintf(layer, "[%s", line);
            result.push_back(layer);
        }
        
    }

    free(data);
}

int get_length(int integer){
    int n = 0;
    while(integer > 0){
        integer = integer/10;
        n++;
    }
    // printf("n is %d\n",n);
    return n;
}

char *deal_header(char *header, int *shape){
    
    char *result;
    char *line = strtok(header, "\n");
    int h_len = get_length(shape[0]);
    int w_len = get_length(shape[1]);
    int c_len = get_length(shape[2]);
    int length = 0;

    // printf("inputs is %d, inputs length %d \n", inputs, inputs_length);

    result = (char *)malloc(sizeof(char) + 1);
    sprintf(result, "\n");

    while (line)
    {
        char *above_data = result;
        if (strlen(line) > 4){
            if (line[0] == 'h' && line[1] == 'e' && line[2] == 'i' && line[3] == 'g' ){
                result = (char *)malloc((8 +h_len +strlen(above_data)) * sizeof(char) + 1);
                sprintf(result, "%sheight=%d\n", above_data, shape[0]);
            }
            else if (line[0] == 'w' && line[1] == 'i' && line[2] == 'd' && line[3] == 't' ){
                result = (char *)malloc((7 + w_len +strlen(above_data)) * sizeof(char) + 1);
                sprintf(result, "%swidth=%d\n", above_data, shape[1]);
            }
            else if (line[0] == 'c' && line[1] == 'h' && line[2] == 'a' && line[3] == 'n' ){
                result = (char *)malloc((10 + c_len +strlen(above_data)) * sizeof(char) + 1);
                sprintf(result, "%schannels=%d\n", above_data, shape[2]);
            }
            else{
                result = (char *)malloc((strlen(line) + 1 +strlen(above_data)) * sizeof(char) + 1);
                sprintf(result, "%s%s\n", above_data, line);
            }
        }
        else{
            result = (char *)malloc((strlen(line) + 1 +strlen(above_data)) * sizeof(char));
            sprintf(result, "%s%s\n", above_data, line);
        }
        // free(above_data);
        line = strtok(NULL, "\n");
    }

    return result;
}

char *cfg_filter(std::vector<int> &layers, std::vector<char*> &data, std::vector<int*> &layer_shapes){
    
    // printf("%s", data[0]);

    char *header_message = (char *)malloc(strlen(data[0]) * sizeof(char) + 1);
    sprintf(header_message, "%s", data[0]);
    int input_layer = layers[0];
    char* adjust_header = deal_header(header_message, layer_shapes[input_layer - 1]);
    // char* adjust_header = header_message;
    // printf("header is !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");

    // printf("header is \n %s \n", adjust_header);

    // printf("header is !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");

    char *result;
    int length = strlen(adjust_header);
    // printf("result is ***********************1 \n");
    result = (char *)malloc(length * sizeof(char) + 1);
    // printf("result is ***********************2 \n");
    sprintf(result, "%s", adjust_header);

    // printf("result is --###################- \n %s \n", result);

    for (int i=0; i < layers.size(); i++){
        length = length + strlen(data[layers[i]]);
        char *layer = result;
        result = (char *)malloc(length * sizeof(char) + 1);
        sprintf(result, "%s%s", layer, data[layers[i]]);
        free(layer);
    }
    return result;
}

void weights_partition(char* cfg_string, std::vector<int*> &result, int layer_num){
    
    std::vector<char*>lines;
    char *line = strtok(cfg_string, "\n");
    
    // printf("weigth partition ----------- 1 \n");
    for (int i = 0; i < layer_num; i++)
    {
        // printf("weighe \n %s \n", line);
        char *line_data = (char *)malloc(sizeof(float) * (strlen(line)));
        
        sprintf(line_data, "%s", line);
        lines.push_back(line_data);

        line = strtok(NULL, "\n");
    }

    for(int i=0; i<lines.size(); i++){
        
        int index = atoi(strtok(lines[i], ","));
        int s_offset = atoi(strtok(NULL, ","));
        int e_offset = atoi(strtok(NULL, ","));
        int *offsets = (int *)malloc(2 * sizeof(int));
        offsets[0] = s_offset;
        offsets[1] = e_offset;

        result.push_back(offsets);
    }
}

char *weights_filter(std::vector<int> &layers, std::vector<int*> &offsets, char* weigths){
    
    int s_offset = offsets[layers[0] - 1][0];
    int e_offset = offsets[layers[layers.size() -1] - 1][1];

    char *result = (char *)malloc(e_offset - s_offset);
    memcpy(result, weigths + s_offset, e_offset - s_offset);
    
    // printf("start layer %d, end layer: %d \n", layers[0], layers[layers.size() -1]);

    // printf("start offset %d, end offset: %d \n", s_offset, e_offset);

    // for (int i=0; i < layers.size(); i++){
    //     length = length + strlen(data[layers[i] - 1]);
    // }

    //         char *layer = result;
    //     result = (char *)malloc(length * sizeof(char));
    //     sprintf(result, "%s%s", layer, data[layers[i] -1]);
    //     free(layer);

    return result;
}