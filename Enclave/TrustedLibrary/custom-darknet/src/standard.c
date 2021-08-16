#include "standard.h"
// FUNCTINOS FROM PARSER AND NETWORK
// from parser

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

box float_to_box(float *f, int stride)
{
    box b = {0};
    b.x = f[0];
    b.y = f[1*stride];
    b.w = f[2*stride];
    b.h = f[3*stride];
    return b;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random") == 0)
        return RANDOM;
    if (strcmp(s, "poly") == 0)
        return POLY;
    if (strcmp(s, "constant") == 0)
        return CONSTANT;
    if (strcmp(s, "step") == 0)
        return STEP;
    if (strcmp(s, "exp") == 0)
        return EXP;
    if (strcmp(s, "sigmoid") == 0)
        return SIG;
    if (strcmp(s, "steps") == 0)
        return STEPS;
    //fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    return net;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch", 1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions", 1);
    net->time_steps = option_find_int_quiet(options, "time_steps", 1);
    net->notruth = option_find_int_quiet(options, "notruth", 0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, "random", 0);

    net->adam = option_find_int_quiet(options, "adam", 0);
    if (net->adam)
    {
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }

    net->h = option_find_int_quiet(options, "height", 0);
    net->w = option_find_int_quiet(options, "width", 0);
    net->c = option_find_int_quiet(options, "channels", 0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop", net->w * 2);
    net->min_crop = option_find_int_quiet(options, "min_crop", net->w);
    net->max_ratio = option_find_float_quiet(options, "max_ratio", (float)net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, "min_ratio", (float)net->min_crop / net->w);
    net->center = option_find_int_quiet(options, "center", 0);
    net->clip = option_find_float_quiet(options, "clip", 0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    // if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
    if (net->policy == STEP)
    {
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    }
    else if (net->policy == STEPS)
    {
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        //if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i)
        {
            if (l[i] == ',')
                ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for (i = 0; i < n; ++i)
        {
            int step = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',') + 1;
            p = strchr(p, ',') + 1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    }
    else if (net->policy == EXP)
    {
        net->gamma = option_find_float(options, "gamma", 1);
    }
    else if (net->policy == SIG)
    {
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    }
    else if (net->policy == POLY || net->policy == RANDOM)
    {
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

LAYER_TYPE string_to_layer_type(char *type)
{

    if (strcmp(type, "[shortcut]") == 0)
        return SHORTCUT;
    if (strcmp(type, "[crop]") == 0)
        return CROP;
    if (strcmp(type, "[cost]") == 0)
        return COST;
    if (strcmp(type, "[detection]") == 0)
        return DETECTION;
    if (strcmp(type, "[region]") == 0)
        return REGION;
    if (strcmp(type, "[yolo]") == 0)
        return YOLO;
    if (strcmp(type, "[local]") == 0)
        return LOCAL;
    if (strcmp(type, "[conv]") == 0 || strcmp(type, "[convolutional]") == 0)
        return CONVOLUTIONAL;
    if (strcmp(type, "[deconv]") == 0 || strcmp(type, "[deconvolutional]") == 0)
        return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]") == 0)
        return ACTIVE;
    if (strcmp(type, "[logistic]") == 0)
        return LOGXENT;
    if (strcmp(type, "[l2norm]") == 0)
        return L2NORM;
    if (strcmp(type, "[net]") == 0 || strcmp(type, "[network]") == 0)
        return NETWORK;
    if (strcmp(type, "[crnn]") == 0)
        return CRNN;
    if (strcmp(type, "[gru]") == 0)
        return GRU;
    if (strcmp(type, "[lstm]") == 0)
        return LSTM;
    if (strcmp(type, "[rnn]") == 0)
        return RNN;
    if (strcmp(type, "[conn]") == 0 || strcmp(type, "[connected]") == 0)
        return CONNECTED;
    if (strcmp(type, "[max]") == 0 || strcmp(type, "[maxpool]") == 0)
        return MAXPOOL;
    if (strcmp(type, "[reorg]") == 0)
        return REORG;
    if (strcmp(type, "[avg]") == 0 || strcmp(type, "[avgpool]") == 0)
        return AVGPOOL;
    if (strcmp(type, "[dropout]") == 0)
        return DROPOUT;
    if (strcmp(type, "[lrn]") == 0 || strcmp(type, "[normalization]") == 0)
        return NORMALIZATION;
    if (strcmp(type, "[batchnorm]") == 0)
        return BATCHNORM;
    if (strcmp(type, "[soft]") == 0 || strcmp(type, "[softmax]") == 0)
        return SOFTMAX;
    if (strcmp(type, "[route]") == 0)
        return ROUTE;
    if (strcmp(type, "[upsample]") == 0)
        return UPSAMPLE;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while (n)
    {
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) printf("ERRPR: Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride,padding);
    return layer;
}


convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;
    
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) printf("ERROR: Layer before convolutional layer must output image.\n");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam);
    
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) printf("Layer before crop layer must output image.");

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    crop_layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',')+1;
        }
        *num = n;
    }
    return mask;
}

layer parse_yolo(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes);
    if (l.outputs == params.inputs) printf("error in yolo");

    l.max_boxes = option_find_int_quiet(options, "max",90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);

    // char *map_file = option_find_str(options, "map", 0);
    // if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

layer parse_connected(list *options, size_params params)
{   
    int output = option_find_int(options, "output", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
    return l;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale", 1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio = option_find_float_quiet(options, "ratio", 0);
    layer.noobject_scale = option_find_float_quiet(options, "noobj", 1);
    layer.thresh = option_find_float_quiet(options, "thresh", 0);
    return layer;
}

layer get_network_output_layer(network *net)
{
    int i;
    for (i = net->n - 1; i >= 0; --i)
    {
        if (net->layers[i].type != COST)
            break;
    }
    return net->layers[i];
}

softmax_layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups", 1);
    softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
    //if (tree_file) layer.softmax_tree = read_tree(tree_file);
    layer.w = params.w;
    layer.h = params.h;
    layer.c = params.c;
    layer.spatial = option_find_float_quiet(options, "spatial", 0);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) printf("Layer before avgpool layer must output image.");

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}


layer parse_shortcut(list *options, size_params params, network *net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if(index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net->layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    s.alpha = option_find_float_quiet(options, "alpha", 1);
    s.beta = option_find_float_quiet(options, "beta", 1);
    return s;
}

route_layer parse_route(list *options, size_params params, network *net)
{
    char *l = option_find(options, "layers");
 
    int len = strlen(l);
    if(!l) printf("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

 
    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}

layer parse_upsample(list *options, size_params params, network *net)
{

    int stride = option_find_int(options, "stride",2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
}

char *read_line(char* msg, char* label, int index){
    int i = 0;
    int startPos = 0;
    int endPos = -1;
    int ariseTimes = 0;
    int finished = 0;

    while(msg[i] != '\0'){
        if (msg[i] == label[0]){
            ariseTimes += 1;
            
            if (index -1 == ariseTimes) {
                startPos = i + 1;
            }
            if (index == ariseTimes) {
                endPos = i;
                break;
            }
        }
        i++;
    }

    // printf("startPos:%d,  endPos:%d, index:%d, msg data is: ...%c\n", startPos, endPos, index, msg[i]);

    if (endPos == -1 && msg[i] == '\0'){
        return NULL;
    }
    else if (startPos == 0 && endPos == -1){
        // printf("first if , char data is: %c ;...\n", msg[i]);
        char* result = (char*)malloc(1);
        result[0] = '\0';
        return result;
    }

    else if (startPos == endPos){
        // printf("second if , char data is: %c ;...\n", msg[i]);
        char* result = (char*)malloc(1);
        result[0] = '\0';
        return result;
    }
    else{
        // printf("char length is: %d;", endPos - startPos);
        char* result = (char*)malloc((endPos - startPos) * sizeof(char) + 1);
        int j = 0;
        for (int i = startPos; i < endPos; i++){
            result[j] = msg[i];
            // printf("result char is: %c ,.... \n", result[j]);
            j++;
        }
        result[endPos - startPos] = '\0';
        // printf("result is: %s ,... \n", result);
        return result;
    }
}


list *sgx_file_string_to_list(char *file_string)
{
    char* line = NULL;
    char * pchStrTmpIn = NULL;

    // printf("string to list.... \n");
    
    // line = strtok(file_string, "\n");
    int pos = 1;
    line = read_line(file_string, "\n", pos); 


    list *options = make_list();
    section *current = 0;

    // ocall_print_string("reading cfg: \n");
    while (line != NULL)
    {   
        // printf("line data is: %s ;...\n", line);
        switch (line[0])
        {
        case '[':
            current = (section *)malloc(sizeof(section));
            list_insert(options, current);
            current->options = make_list();
            current->type = line;
            break;
        case '\0':
        case '#':
        case ';':
            free(line);
            break;
        default:
            read_option(line, current->options);
            break;
        }
        pos++;
        line = read_line(file_string, "\n", pos);
        // line = strtok(NULL, "\n");
        
        
    }

    return options;
}

network *sgx_parse_network_cfg(list *sections)
{   
    // list *sections = read_cfg(filename);
    node *n = sections->front;
    //if(!n) error("Config file has no sections");
    network *net = make_network(sections->size - 1);
    // printf("section size is: %d \n", sections->size);
    net->gpu_index = -1;
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    //if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    // printf("h %d,w %d,c %d \n", params.h, params.w, params.c);
    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    int index = 1;
    while (n)
    {
        params.index = count;
        //fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = {0};
        // printf("layer %d type is: %s\n", index, s->type);
        // printf("layer inputs is: %d \n", params.inputs);
        index ++;
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL){
             l = parse_convolutional(options, params);
        }else if(lt == CONNECTED){
            l = parse_connected(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net->layers[count-1].output;
            l.delta = net->layers[count-1].delta;
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == SHORTCUT){
            l = parse_shortcut(options, params, net);
        }else if(lt == CROP){
            l = parse_crop(options, params);
        }else if(lt == YOLO){
            l = parse_yolo(options, params);
        }else if(lt == UPSAMPLE){
            l = parse_upsample(options, params, net);
        }else if(lt == ROUTE){
            l = parse_route(options, params, net);
        }else{
            printf("layer type not recognized: %s\n", s->type);
        }


        l.clip = net->clip;
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, "dontsave", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size)
            workspace_size = l.workspace_size;
        // free_section(s);
        n = n->next;
        ++count;

        if (n)
        {
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
        // printf("layer outputs is: %d", l.outputs);
    }

    // free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if (net->layers[net->n - 1].truths)
        net->truths = net->layers[net->n - 1].truths;
    net->output = out.output;
    net->input = calloc(net->inputs * net->batch, sizeof(float));
    net->truth = calloc(net->truths * net->batch, sizeof(float));

#ifdef GPU
    net->output_gpu = out.output_gpu;
    net->input_gpu = cuda_make_array(net->input, net->inputs * net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths * net->batch);
#endif
    if (workspace_size)
    {
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if (gpu_index >= 0)
        {
            net->workspace = cuda_make_array(0, (workspace_size - 1) / sizeof(float) + 1);
        }
        else
        {
            net->workspace = calloc(1, workspace_size);
        }
#else
        net->workspace = calloc(1, workspace_size);
#endif
    }
    return net;
}

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}