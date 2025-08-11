#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double activated_output){
    return activated_output * (1.0 - activated_output);
}


typedef struct{
    double* weights;
    double bias;
    
    double z_output;
    double a_output;

    double error_delta;
} Neuron;

typedef struct{
    Neuron* neurons;
    int num_neurons;
    int num_weights_per_neuron;
} Layer;

typedef struct{
    Layer* layers;
    int num_layers;
    double loss;
} NeuralNetwork;

// NeuralNetwork �ʱ�ȭ �Լ�
NeuralNetwork* create_neural_network(const int layer_num, const int neuron_num, const int weight_num);
//
NeuralNetwork* create_neural_network_flexible(const int num_layers, const int* layers_config);
// Forward Propagation �Լ�
void forward_propagate(NeuralNetwork* nn, double* inputs);
// Back Propagation �Լ�
void back_propagate(NeuralNetwork* nn, double* inputs, double* actual_outputs, double learning_rate);

void free_neural_network(NeuralNetwork* nn);


int main() {
    // 1. �غ� (Setup)
    srand(time(NULL)); // ���� �õ� �ʱ�ȭ

    // XOR �н� ������
    double inputs[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double outputs[][1] = {{0}, {1}, {1}, {0}};

    // �Ű�� ���� ����
    int layers_config[] = {2, 4, 4, 1}; // �Է���, ������, ������� ���� ��
    // (create_neural_network�� �� �迭�� �°� �����ߴٰ� ����)
    NeuralNetwork* nn = create_neural_network_flexible(4, layers_config);

    // �н� �Ķ����
    int epochs = 1000000;
    double learning_rate = 0.1;

    // 2. �Ʒ� (Training Loop)
    printf("�Ʒ� ����...\n");
    for (int i = 0; i < epochs; i++) {
        double total_loss = 0;
        // 4���� �����ͼ��� ��� ����Ͽ� �Ʒ�
        for (int j = 0; j < 4; j++) {
            // ������
            forward_propagate(nn, inputs[j]);

            // �ս� ��� (����͸���)
            total_loss += pow(outputs[j][0] - nn->layers[nn->num_layers - 1].neurons[0].a_output, 2);

            // ������
            back_propagate(nn, inputs[j], outputs[j], learning_rate);
        }

        // 1000�� ���� �ս� ���
        if (i % 1000 == 0) {
            printf("Epoch %d, Loss: %f\n", i, total_loss / 4.0);
        }
    }
    printf("�Ʒ� ����!\n\n");

    // 3. ��� Ȯ�� (Testing)
    printf("�׽�Ʈ ���:\n");
    for (int i = 0; i < 4; i++) {
        forward_propagate(nn, inputs[i]);
        double prediction = nn->layers[nn->num_layers - 1].neurons[0].a_output;
        printf("�Է�: [%d, %d], ����: %d, ����: %f\n", (int)inputs[i][0], (int)inputs[i][1], (int)outputs[i][0], prediction);
    }
    
    free_neural_network(nn);
    
    return 0;
}


NeuralNetwork* create_neural_network(const int layer_num, const int neuron_num, const int weight_num) {
    if (layer_num <= 0 || neuron_num <= 0 || weight_num <= 0) {
        return NULL;
    }

    // 1. �Ű�� ����ü �Ҵ�
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    nn->num_layers = layer_num;

    // 2. ���̾� �迭 �Ҵ�
    nn->layers = malloc(sizeof(Layer) * layer_num);


    for (int i = 0; i < layer_num; i++) {
        // �����͸� ���� �ʰ� �ٷ� �迭�� ��ҿ� ����
        nn->layers[i].num_neurons = neuron_num;
        nn->layers[i].num_weights_per_neuron = weight_num;
        nn->layers[i].neurons = malloc(sizeof(Neuron) * neuron_num);

        for (int j = 0; j < neuron_num; j++) {
            nn->layers[i].neurons[j].bias = 0; // ������ 0���� �ʱ�ȭ
            nn->layers[i].neurons[j].weights = malloc(sizeof(double) * weight_num);

            for (int k = 0; k < weight_num; k++) {
                // ����ġ�� -0.5 ~ 0.5 ������ ���� ���� ������ �ʱ�ȭ
                nn->layers[i].neurons[j].weights[k] = ((double)rand() / RAND_MAX) - 0.5;
            }
        }
    }
    return nn;
}

NeuralNetwork* create_neural_network_flexible(const int num_layers, const int* layers_config) {
    if (num_layers <= 1) {
        return NULL; // �ּ� 2�� ��(�Է�, ���)�� �־�� ��
    }

    // 1. �Ű�� ����ü �Ҵ�
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layers = malloc(sizeof(Layer) * num_layers);

    // 2. �� ��(Layer)�� ������� ����
    for (int i = 0; i < num_layers; i++) {
        Layer* current_layer = &nn->layers[i];
        
        // ���� ���� ���� ���� ����
        current_layer->num_neurons = layers_config[i];
        current_layer->neurons = malloc(sizeof(Neuron) * current_layer->num_neurons);

        // ���� ���� �������� ���� ����ġ�� ���� ����
        int num_weights_per_neuron = 0;
        if (i > 0) { // �Է����� �ƴ� ��쿡��
            // ����ġ ���� = ���� ���� ���� ����
            num_weights_per_neuron = layers_config[i - 1];
        }
        current_layer->num_weights_per_neuron = num_weights_per_neuron;

        // 3. ���� ���� �� ����(Neuron)�� �ʱ�ȭ
        for (int j = 0; j < current_layer->num_neurons; j++) {
            Neuron* current_neuron = &current_layer->neurons[j];
            
            // �Է����� �ƴ� ��쿡�� ����ġ�� ������ �Ҵ��ϰ� �ʱ�ȭ
            if (i > 0) {
                current_neuron->bias = 0; // ������ 0���� ����
                current_neuron->weights = malloc(sizeof(double) * num_weights_per_neuron);
                
                for (int k = 0; k < num_weights_per_neuron; k++) {
                    // ����ġ�� -0.5 ~ 0.5 ������ ���� ���� ������ �ʱ�ȭ
                    current_neuron->weights[k] = ((double)rand() / RAND_MAX) - 0.5;
                }
            } else {
                // �Է��� ������ ����ġ�� ������ ����
                current_neuron->weights = NULL;
                current_neuron->bias = 0;
            }
        }
    }
    return nn;
}

void forward_propagate(NeuralNetwork* nn, double* inputs) {
    // 1. �Է����� ���(a_output)�� �Է°� �״�� ����
    Layer* input_layer = &nn->layers[0];
    for (int i = 0; i < input_layer->num_neurons; i++) {
        input_layer->neurons[i].a_output = inputs[i];
    }

    // 2. �� ��° ��(ù ��° ������)���� ������ ������ ������ ���
    for (int i = 1; i < nn->num_layers; i++) {
        Layer* prev_layer = &nn->layers[i - 1]; // ���� ��
        Layer* current_layer = &nn->layers[i];   // ���� ��

        for (int j = 0; j < current_layer->num_neurons; j++) {
            Neuron* neuron = &current_layer->neurons[j];
            double weighted_sum = 0.0;

            // ������ ���: ���� ���� ��� ���� ���(a_output)�� ����ġ�� ����
            for (int k = 0; k < prev_layer->num_neurons; k++) {
                weighted_sum += prev_layer->neurons[k].a_output * neuron->weights[k];
            }
            weighted_sum += neuron->bias;
            neuron->z_output = weighted_sum;

            // Ȱ��ȭ �Լ� ����
            neuron->a_output = sigmoid(neuron->z_output);
        }
    }
}

void back_propagate(NeuralNetwork* nn, double* inputs, double* actual_outputs, double learning_rate) {
    // 1. ������ ������ �������� ��ȸ
    for (int i = nn->num_layers - 1; i >= 0; i--) {
        Layer* layer = &nn->layers[i];
        
        // 2. ���� ��(i)�� �� ����(j)�� ���� ���� ��ȣ(delta) ���
        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron* neuron = &layer->neurons[j];
            double error_signal;

            if (i == nn->num_layers - 1) { // ������� ���
                double error = neuron->a_output - actual_outputs[j];
                error_signal = error * sigmoid_derivative(neuron->a_output);
            } else { // ������ ���� ���
                double propagated_error = 0.0;
                Layer* next_layer = &nn->layers[i + 1];
                
                for (int k = 0; k < next_layer->num_neurons; k++) {
                    // ���� �� ���� k�� ����� error_delta ���� ���
                    propagated_error += next_layer->neurons[k].error_delta * next_layer->neurons[k].weights[j];
                }
                error_signal = propagated_error * sigmoid_derivative(neuron->a_output);
            }
            // ���� ���� ��ȣ�� ���� ���ο� ����
            neuron->error_delta = error_signal; 
        }

        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron* neuron = &layer->neurons[j];

            for (int k = 0; k < layer->num_weights_per_neuron; k++) {
                // �� ����ġ(k)�� �ش��ϴ� �Է��� ���� ã��
                double input_for_this_weight;
                if (i == 0) {
                    // ù ��° ���� ���, ���� �Է��� ���
                    input_for_this_weight = inputs[k];
                } else {
                    // ������ ���� ���, ���� ���� k��° ���� ����� ���
                    input_for_this_weight = nn->layers[i - 1].neurons[k].a_output;
                }
                
                // �׷����Ʈ ��� �� ����ġ ������Ʈ
                double gradient = neuron->error_delta * input_for_this_weight;
                neuron->weights[k] -= learning_rate * gradient;
            }
            
            // ���� ������Ʈ
            neuron->bias -= learning_rate * neuron->error_delta;
        }
    }
}

void free_neural_network(NeuralNetwork* nn) {
    // �Ű�� �����Ͱ� NULL�̸� �ƹ��͵� ���� ����
    if (nn == NULL) {
        return;
    }

    // 1. ���� ���ʺ���, �� ��(Layer)�� ��ȸ
    for (int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &nn->layers[i];
        
        // �Է����� �ƴ� ��쿡�� ����ġ �迭 ����
        if (i > 0) {
            // 2. �� ����(Neuron)�� ��ȸ�ϸ� ����ġ(weights) �迭 ����
            for (int j = 0; j < layer->num_neurons; j++) {
                if (layer->neurons[j].weights != NULL) {
                    free(layer->neurons[j].weights);
                }
            }
        }
        
        // 3. ����(Neuron) �迭 ����
        if (layer->neurons != NULL) {
            free(layer->neurons);
        }
    }

    // 4. ���̾�(Layer) �迭 ����
    if (nn->layers != NULL) {
        free(nn->layers);
    }

    // 5. ���������� �Ű��(NeuralNetwork) ����ü ��ü�� ����
    free(nn);
}