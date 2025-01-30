#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "quantize.h"
#include "../nn.h"

void print_usage() {
    printf("Usage: quantize <input_model> <output_model>\n");
    printf("  input_model: path to the floating-point neural network model\n");
    printf("  output_model: path where to save the quantized model\n");
}

// Helper function to find min and max values in a layer
static void find_layer_minmax(float* values, int size, float* min_val, float* max_val) {
    *min_val = values[0];
    *max_val = values[0];
    for (int i = 1; i < size; i++) {
        if (values[i] < *min_val) *min_val = values[i];
        if (values[i] > *max_val) *max_val = values[i];
    }
}

// Helper function to quantize a single float value to int8
static int8_t quantize_value(float value, float scale, float zero_point) {
    float quantized = value / scale + zero_point;
    if (quantized > 127) return 127;
    if (quantized < -128) return -128;
    return (int8_t)round(quantized);
}

nn_quantized_t* nn_quantize(nn_t* network, quantization_method_t method, int bit_depth) {
    if (!network || bit_depth != 8) {  // Currently only supporting 8-bit quantization
        return NULL;
    }

    nn_quantized_t* quantized = malloc(sizeof(nn_quantized_t));
    if (!quantized) return NULL;

    quantized->original_network = network;
    
    // Allocate memory for quantized weights and scales
    quantized->quantized_weights = malloc(sizeof(int8_t**) * network->depth);
    quantized->weight_scales = malloc(sizeof(float*) * network->depth);
    quantized->quantized_biases = malloc(sizeof(int8_t*) * network->depth);
    quantized->bias_scales = malloc(sizeof(float) * network->depth);

    for (int layer = 1; layer < network->depth; layer++) {
        int prev_width = network->width[layer-1];
        int curr_width = network->width[layer];

        // Allocate memory for this layer
        quantized->quantized_weights[layer] = malloc(sizeof(int8_t*) * curr_width);
        quantized->weight_scales[layer] = malloc(sizeof(float) * curr_width);
        quantized->quantized_biases[layer] = malloc(sizeof(int8_t) * curr_width);

        // Quantize weights for each neuron in this layer
        for (int neuron = 0; neuron < curr_width; neuron++) {
            quantized->quantized_weights[layer][neuron] = malloc(sizeof(int8_t) * prev_width);
            
            // Find min/max for weights of this neuron
            float min_val, max_val;
            find_layer_minmax(network->weight[layer][neuron], prev_width, &min_val, &max_val);
            
            // Calculate scale and zero point for symmetric quantization
            float scale = (float)fmax(fabs(min_val), fabs(max_val)) / 127.0f;
            quantized->weight_scales[layer][neuron] = scale;
            float zero_point = 0.0f;  // For symmetric quantization

            // Quantize weights
            for (int w = 0; w < prev_width; w++) {
                quantized->quantized_weights[layer][neuron][w] = 
                    quantize_value(network->weight[layer][neuron][w], scale, zero_point);
            }
        }

        // Quantize biases
        float bias_min, bias_max;
        find_layer_minmax(network->bias + layer, curr_width, &bias_min, &bias_max);
        float bias_scale = (float)fmax(fabs(bias_min), fabs(bias_max)) / 127.0f;
        quantized->bias_scales[layer] = bias_scale;

        for (int neuron = 0; neuron < curr_width; neuron++) {
            quantized->quantized_biases[layer][neuron] = 
                quantize_value(network->bias[layer], bias_scale, 0.0f);
        }
    }

    return quantized;
}

int nn_save_quantized(nn_quantized_t* quantized_network, const char* path) {
    if (!quantized_network || !path) return -1;

    FILE* file = fopen(path, "w");
    if (!file) return -1;

    nn_t* network = quantized_network->original_network;

    // Save network architecture
    fprintf(file, "%d\n", network->depth);
    for (int i = 0; i < network->depth; i++) {
        fprintf(file, "%d %d %f\n", network->width[i], network->activation[i], network->bias[i]);
    }

    // Save quantized weights and scales
    for (int layer = 1; layer < network->depth; layer++) {
        for (int neuron = 0; neuron < network->width[layer]; neuron++) {
            // Save weight scale
            fprintf(file, "%f\n", quantized_network->weight_scales[layer][neuron]);
            
            // Save quantized weights
            for (int w = 0; w < network->width[layer-1]; w++) {
                fprintf(file, "%d\n", quantized_network->quantized_weights[layer][neuron][w]);
            }
        }
        
        // Save bias scale and quantized biases
        fprintf(file, "%f\n", quantized_network->bias_scales[layer]);
        for (int neuron = 0; neuron < network->width[layer]; neuron++) {
            fprintf(file, "%d\n", quantized_network->quantized_biases[layer][neuron]);
        }
    }

    fclose(file);
    return 0;
}

float* nn_predict_quantized(nn_quantized_t* qmodel, float* input) {
    if (!qmodel || !input) return NULL;

    nn_t* original = qmodel->original_network;
    int depth = original->depth;
    float* activations = malloc(sizeof(float) * original->width[0]);
    memcpy(activations, input, sizeof(float) * original->width[0]);

    for (int layer = 1; layer < depth; layer++) {
        int curr_width = original->width[layer];
        float* new_activations = malloc(sizeof(float) * curr_width);

        for (int neuron = 0; neuron < curr_width; neuron++) {
            // Dequantize weights and compute dot product
            float sum = 0.0f;
            for (int w = 0; w < original->width[layer-1]; w++) {
                float weight = qmodel->quantized_weights[layer][neuron][w] * 
                              qmodel->weight_scales[layer][neuron];
                sum += weight * activations[w];
            }

            // Dequantize bias
            float bias = qmodel->quantized_biases[layer][neuron] * 
                        qmodel->bias_scales[layer];
            sum += bias;

            // Apply activation
            new_activations[neuron] = activate(sum, original->activation[layer]);
        }

        free(activations);
        activations = new_activations;
    }

    return activations;
}
