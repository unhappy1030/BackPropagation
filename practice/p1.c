#include<stdio.h>
#include<stdio.h>
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
    int num_weights;
} Layer;

typedef struct{
    Layer* layers;
    int num_layers;
    double loss;
} NeuralNetwork;

// NeuralNetwork 초기화 함수
NeuralNetwork* create_neural_network(const int num_layers, const int* layers_config);

void forward_propagate(NeuralNetwork* nn, double* inputs);
void back_propagate(NeuralNetwork* nn, double* inputs, double* actual_outputs, double learning_rate);
void free_neural_network(NeuralNetwork* nn);

int main(){
    return 0;
}

NeuralNetwork* create_neural_network(const int num_layers, const int* layers_config){
    if(num_layers <= 1){
        return NULL;
    }

    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layers = malloc(sizeof(Layer) * num_layers);

    for(int i = 0; i < num_layers; i++){
        Layer* current_layer = &nn->layers[i];

        current_layer->num_neurons = layers_config[i];
        current_layer->neurons = malloc(sizeof(Neuron) * current_layer->num_neurons);

        // 첫번째 레이어를 단순히 입력을 그대로 옮기는 레이어로 사용
        int num_weights = 0;
        if(i > 0){
            //가중치 개수 = 이전 층의 뉴런 개수
            num_weights = layers_config[i-1];
        }
        current_layer->num_weights = num_weights;

        for(int j = 0; j < current_layer->num_neurons; j++){
            Neuron* current_neuron = &current_layer->neurons[j];

            //입력층이 아닌 경우에만 가중치와 편향을 할당하고 초기화
            if(i > 0){
                current_neuron->bias = 0;
                current_neuron->weights = malloc(sizeof(double) * current_layer->num_weights);

                for(int k = 0; k < num_weights; k++){
                    // 가중치를 -0.5 ~ 0.5 사이의 작은 랜덤 값으로 초기화
                    current_neuron->weights[k] = ((double)rand() / RAND_MAX) - 0.5;
                }
            }
            else{
                current_neuron->weights = NULL;
                current_neuron->bias = 0;
            }
        }
    }
    return nn;
}

void forward_propagate(NeuralNetwork* nn, double* inputs){
    // 1. 입력층의 출력(a_output)은 입력값 그대로 설정
    Layer* input_layer = &nn->layers[0];
    for(int i = 0; i < input_layer->num_neurons; i++){
        input_layer->neurons[i].a_output = inputs[i];
    }

    // 2. 두 번째 층(첫 번째 은닉층)부터 마지막 층까지 순전파 계산
    for(int i = 1; i < nn->num_layers; i++){
        Layer* prev_layer = &nn->layers[i-1];
        Layer* current_layer = &nn->layers[i];

        for(int j = 0; j < current_layer->num_neurons; j++){
            Neuron* neuron = &current_layer->neurons[j];
            double weighted_sum = 0.0;

            // 가중합 계산: 이전 층의 모든 뉴런 출력(a_output)과 가중치를 곱함
            for(int k = 0; k < prev_layer->num_neurons; k++){
                weighted_sum += prev_layer->neurons[k].a_output * neuron->weights[k];
            }
            weighted_sum += neuron->bias;
            neuron->z_output = weighted_sum;

            // 활성화 함수 적용
            neuron->a_output = sigmoid(weighted_sum);
        }
    }
}

void back_propagate(NeuralNetwork* nn, double* inputs, double* actual_outputs, double learning_rate){
    // 1. 마지막 층부터 역순으로 순회
    for(int i = nn->num_layers - 1; i >= 0; i++){
        Layer* layer = &nn->layers[i];

        for(int j = 0; j < layer->num_neurons; j++){
            Neuron* neuron = &layer->neurons[j];
            double error_signal;

            if(i == nn->layers - 1){ // 출력층의 경우
                double error = neuron->a_output - actual_outputs[j];
                error_signal = error * sigmoid_derivative(neuron_>a_output);
            } else{ // 히든 레이어의 경우
                double propagated_error = 0.0;
                Layer* next_layer = &nn->layers[i+1];

                for(int k = 0; k < next_layer->num_neurons; k++){
                    // 다음 층 뉴런 k의 저장된 error_delta 값을 사용
                    propagated_error += next_layer->neurons[k].error_delta * next_layer->neurons[k].weights[j];
                }
            }
        }
    }
}