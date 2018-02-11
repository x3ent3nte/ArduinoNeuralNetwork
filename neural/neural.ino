#include <time.h>

#define E 2.71828182845904523536
#define EPSILON 0.005

#define NUM_INPUTS 2
#define NUM_HIDDEN 6
#define NUM_OUTPUTS 1

#define HIDDEN_WIDTH 1

#define NUM_LAYERS 2 + HIDDEN_WIDTH
#define NUM_NEURONS NUM_INPUTS + NUM_OUTPUTS + (HIDDEN_WIDTH * NUM_HIDDEN)

#define NUM_WEIGHTS ((NUM_HIDDEN * NUM_INPUTS) + NUM_HIDDEN) + (((NUM_HIDDEN * NUM_HIDDEN) + NUM_HIDDEN) * (HIDDEN_WIDTH - 1)) + ((NUM_OUTPUTS * NUM_HIDDEN) + NUM_OUTPUTS)

#define DATA_SIZE 40

#define SEED 999

struct Neuron {
    float sum;
    float sig;
    float delta;
};

float randomFloat() {
    return ((float) rand()) / ((float) RAND_MAX);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + powf(E, -x));
}

float sigmoidPrime(float x) {
    float sig_x = sigmoid(x);
    return sig_x * (1.0f - sig_x);
}

void generateData(float input[DATA_SIZE][NUM_INPUTS], float output[DATA_SIZE][NUM_OUTPUTS]) {
    for (int i = 0; i < DATA_SIZE; i++) {
        
        int a = rand() % 2;
        int b = rand() % 2;
        
        int z = a ^ b;

        input[i][0] = (float) a;
        input[i][1] = (float) b;

        output[i][0] = (float) z;
    }
}

void randomizeWeights(float weights[NUM_WEIGHTS]) {
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        weights[i] = (randomFloat() - 0.5f) * 0.9f;
    }
}

void input(Neuron neurons[NUM_NEURONS], float input[NUM_INPUTS]) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        neurons[i].sig = input[i];
    }
}

void feedForward(int layer_widths[NUM_LAYERS], Neuron neurons[NUM_NEURONS], float weights[NUM_WEIGHTS]) {
    int prev_neuron_index = 0;
    int neuron_index = NUM_INPUTS;
    int weight_index = 0;
    for (int i = 1; i < NUM_LAYERS; i++) {
        int prev_layer_width = layer_widths[i - 1];
        int layer_width = layer_widths[i];
        for (int j = 0; j < layer_width; j++) {
            Neuron neuron = neurons[neuron_index];
            float sum = 0.0f;
            for (int k = 0; k < prev_layer_width; k++) {
                sum += neurons[prev_neuron_index++].sig * weights[weight_index++];
            }
            sum += weights[weight_index++];

            neuron.sum = sum;
            neuron.sig = sigmoid(sum);
            neurons[neuron_index] = neuron;

            neuron_index++;
            prev_neuron_index -= prev_layer_width;
        }
        prev_neuron_index += prev_layer_width;
    }
}

void computeOutputDeltas(Neuron neurons[NUM_NEURONS], float outputs[NUM_OUTPUTS]) {
    int neurons_output_offset = NUM_NEURONS - NUM_OUTPUTS;
    for (int i = 0 ; i < NUM_OUTPUTS; i++) {
        int neuron_index = neurons_output_offset + i;
        Neuron neuron = neurons[neuron_index];

        float error = outputs[i] - round(neuron.sum);
        neuron.delta = error * sigmoidPrime(neuron.sum);
        neurons[neuron_index] = neuron;
    }
}

void computeHiddenDeltas(int layer_widths[NUM_LAYERS], Neuron neurons[NUM_NEURONS], float weights[NUM_WEIGHTS]) {
    int next_neuron_index = NUM_NEURONS - 1;
    int neuron_index = next_neuron_index - NUM_OUTPUTS;
    int weight_offset = NUM_WEIGHTS - (NUM_OUTPUTS * (layer_widths[NUM_LAYERS - 1] + 1)) ;

    for (int i = NUM_LAYERS - 2; i >= 0 ; i--) {
        int layer_width = layer_widths[i];
        int next_layer_width = layer_widths[i + 1];
        for (int j = layer_width - 1; j >= 0; j--) {
            float sum_delta_x_weights = 0.0f;
            for (int k = 0; k < next_layer_width; k++) {
                sum_delta_x_weights += neurons[next_neuron_index--].delta * weights[weight_offset + (k * (layer_width + 1)) + j];
            }

            Neuron neuron = neurons[neuron_index];
            neuron.delta = sigmoidPrime(neuron.sum) * sum_delta_x_weights;
            neurons[neuron_index] = neuron;

            neuron_index--;
            next_neuron_index += next_layer_width; 
        }
        weight_offset -= layer_widths[i] * (layer_widths[i - 1] + 1);
        next_neuron_index -= next_layer_width;
    }
}

void adjustWeights(int layer_widths[NUM_LAYERS], Neuron neurons[NUM_NEURONS], float weights[NUM_WEIGHTS], float learning_rate) {
    int weight_index = 0;
    int prev_neuron_index = 0;
    int neuron_index = NUM_INPUTS;

    for (int i = 1; i < NUM_LAYERS; i++) {
        int prev_layer_width = layer_widths[i - 1];
        int layer_width = layer_widths[i];
        float delta = neurons[neuron_index].delta;
        for (int j = 0; j < layer_width; j++) {
            for (int k = 0; k < prev_layer_width; k++) {
                weights[weight_index++] += learning_rate * delta * neurons[prev_neuron_index++].sig;
            }
            weights[weight_index++] += learning_rate * delta;
            
            neuron_index++;
            prev_neuron_index -= prev_layer_width;
        }
        prev_neuron_index += prev_layer_width;
    }
}

void trainRow(int layer_widths[NUM_LAYERS],
    Neuron neurons[NUM_NEURONS], float weights[NUM_WEIGHTS],
    float inputs[NUM_INPUTS], float outputs[NUM_OUTPUTS], float learning_rate) {

    input(neurons, inputs);
    feedForward(layer_widths, neurons, weights);

    computeOutputDeltas(neurons, outputs);
    computeHiddenDeltas(layer_widths, neurons, weights);
    adjustWeights(layer_widths, neurons, weights, learning_rate);
}

void train(int layer_widths[NUM_LAYERS], 
    Neuron neurons[NUM_NEURONS], float weights[NUM_WEIGHTS],
    float inputs[DATA_SIZE][NUM_INPUTS], float outputs[DATA_SIZE][NUM_OUTPUTS], 
    float learning_rate) {

    for (int i = 0; i < DATA_SIZE * 20; i++) {
        int row_num = i % DATA_SIZE;
        trainRow(layer_widths, neurons, weights, inputs[row_num], outputs[row_num], learning_rate);
    }
}

float outputError(Neuron neurons[NUM_NEURONS], float output[NUM_OUTPUTS]) {
    int neurons_output_offset = NUM_NEURONS - NUM_OUTPUTS;
    float sum_errors = 0.0f;
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        float exp = output[i];
        float actual = round(neurons[neurons_output_offset + i].sum);
        float error = exp - actual;
        sum_errors += abs(error);

        /*Serial.print("Expected: ");
        Serial.print(exp);
        Serial.print(" Actual: ");
        Serial.print(actual);
        Serial.print(" Error:");
        Serial.println(error);*/
    }
    return sum_errors;
}

float testRow(int layer_widths[NUM_LAYERS],
    Neuron neurons[NUM_NEURONS], float weights[NUM_WEIGHTS],
    float inputs[NUM_INPUTS], float outputs[NUM_OUTPUTS]) {
    
    input(neurons, inputs);
    feedForward(layer_widths, neurons, weights);
    return outputError(neurons, outputs);
}

float test(int layer_widths[NUM_LAYERS], 
    Neuron neurons[NUM_NEURONS], float weights[NUM_WEIGHTS], 
    float inputs[DATA_SIZE][NUM_INPUTS], float outputs[DATA_SIZE][NUM_OUTPUTS]) {

    float sum_errors = 0.0f;

    for (int i = 0; i < DATA_SIZE; i++) {
        sum_errors += testRow(layer_widths, neurons, weights, inputs[i], outputs[i]);
    }
    return sum_errors / DATA_SIZE;
}

void setup() {
    Serial.begin(57600);

    srand(time(NULL));

    Serial.println("Started");

    Serial.print("Num layers: ");
    Serial.println(NUM_LAYERS);
    Serial.print("Num neurons: ");
    Serial.println(NUM_NEURONS);
    Serial.print("Num layers: ");
    Serial.println(NUM_WEIGHTS);

    int layer_widths[NUM_LAYERS];

    layer_widths[0] = NUM_INPUTS;
    for (int i = 0; i < HIDDEN_WIDTH; i++) {
        layer_widths[i + 1] = NUM_HIDDEN;
    }
    layer_widths[NUM_LAYERS - 1] = NUM_OUTPUTS;

    Neuron neurons[NUM_NEURONS];
    float weights[NUM_WEIGHTS];
    randomizeWeights(weights);

    float training_input[DATA_SIZE][NUM_INPUTS];
    float training_output[DATA_SIZE][NUM_OUTPUTS];
    generateData(training_input, training_output);

    float testing_input[DATA_SIZE][NUM_INPUTS];
    float testing_output[DATA_SIZE][NUM_OUTPUTS];
    generateData(testing_input, testing_output);

    float avg_error = 999.0f;

    float learning_rate = 0.09;
    while (avg_error > EPSILON) {
        train(layer_widths, neurons, weights, training_input, training_output, learning_rate);
        avg_error = test(layer_widths, neurons, weights, testing_input, testing_output);

        Serial.print("Average error: ");
        Serial.println(avg_error);
    }

    Serial.println("Finished");
}

void loop() {
    return;
}
