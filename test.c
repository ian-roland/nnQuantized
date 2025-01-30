/*
 * Neural Network library
 * Copyright (c) 2019-2024 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>  // For strstr()
#include "nn.h"
#include "data_prep.h"
#include "quantizing/quantize.h"  // Add this

int main(int argc, char *argv[]) {
    nn_t *model = NULL;
    nn_quantized_t *qmodel = NULL;
    data_t *data;
    float *prediction;
    int num_samples, correct, true_positive, false_positive;

    if (argc < 2) {
        printf("Usage: %s <model_file>\n", argv[0]);
        return 1;
    }

    // Load model (quantized or floating-point)
    qmodel = nn_load_quantized(argv[1]);
    if (!qmodel) {
        model = nn_load(argv[1]);
        if (!model) {
            printf("Error: Invalid model file\n");
            return 1;
        }
    }

    // Get input/output dimensions
    int input_size, output_size;
    if (qmodel) {
        input_size = qmodel->original_network->width[0];
        output_size = qmodel->original_network->width[qmodel->original_network->depth - 1];
    } else {
        input_size = model->width[0];
        output_size = model->width[model->depth - 1];
    }

    // Load training data
    data = data_load("train.csv", input_size, output_size);
    num_samples = 0;
    correct = 0;

    for (int i = 0; i < data->num_rows; i++) {
        num_samples++;
        if (qmodel) {
            prediction = nn_predict_quantized(qmodel, data->input[i]);
        } else {
            prediction = nn_predict(model, data->input[i]);
        }

        true_positive = 0;
        false_positive = 0;
        for (int j = 0; j < output_size; j++) {
            if (data->target[i][j] >= 0.5) {
                if (prediction[j] >= 0.5) true_positive++;
            } else {
                if (prediction[j] >= 0.5) false_positive++;
            }
        }
        if ((true_positive == 1) && (false_positive == 0)) correct++;
    }

    printf("Train: %d/%d = %2.2f%%\n", correct, num_samples, (correct * 100.0) / num_samples);
    data_free(data);

    // Repeat for test data
    data = data_load("test.csv", input_size, output_size);
    num_samples = 0;
    correct = 0;

    for (int i = 0; i < data->num_rows; i++) {
        num_samples++;
        if (qmodel) {
            prediction = nn_predict_quantized(qmodel, data->input[i]);
        } else {
            prediction = nn_predict(model, data->input[i]);
        }

        true_positive = 0;
        false_positive = 0;
        for (int j = 0; j < output_size; j++) {
            if (data->target[i][j] >= 0.5) {
                if (prediction[j] >= 0.5) true_positive++;
            } else {
                if (prediction[j] >= 0.5) false_positive++;
            }
        }
        if ((true_positive == 1) && (false_positive == 0)) correct++;
    }

    printf("Test: %d/%d = %2.2f%%\n", correct, num_samples, (correct * 100.0) / num_samples);
    data_free(data);

    // Cleanup
    if (qmodel) {
        nn_free_quantized(qmodel);
    } else {
        nn_free(model);
    }
    return 0;
}