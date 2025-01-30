#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "quantize.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        print_usage();
        return 1;
    }

    char* input_model = argv[1];
    char* output_model = argv[2];
    
    // Load the original network
    nn_t* network = nn_load(input_model);
    if (!network) {
        fprintf(stderr, "Failed to load input model: %s\n", input_model);
        return 1;
    }

    // Quantize the network (using symmetric 8-bit quantization)
    nn_quantized_t* quantized = nn_quantize(network, QUANTIZATION_METHOD_SYMMETRIC, 8);
    if (!quantized) {
        fprintf(stderr, "Failed to quantize network\n");
        nn_free(network);
        return 1;
    }

    // Save the quantized network
    if (nn_save_quantized(quantized, output_model) != 0) {
        fprintf(stderr, "Failed to save quantized model: %s\n", output_model);
        nn_free_quantized(quantized);
        nn_free(network);
        return 1;
    }

    printf("Successfully quantized model:\n");
    printf("  Input: %s\n", input_model);
    printf("  Output: %s\n", output_model);

    // Clean up
    nn_free_quantized(quantized);
    nn_free(network);
    
    return 0;
}