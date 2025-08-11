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

// NeuralNetwork 초기화 함수
NeuralNetwork* create_neural_network(const int layer_num, const int neuron_num, const int weight_num);
//
NeuralNetwork* create_neural_network_flexible(const int num_layers, const int* layers_config);
// Forward Propagation 함수
void forward_propagate(NeuralNetwork* nn, double* inputs);
// Back Propagation 함수
void back_propagate(NeuralNetwork* nn, double* inputs, double* actual_outputs, double learning_rate);

void free_neural_network(NeuralNetwork* nn);


int main() {
    // 1. 준비 (Setup)
    srand(time(NULL)); // 랜덤 시드 초기화

    // XOR 학습 데이터
    double inputs[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double outputs[][1] = {{0}, {1}, {1}, {0}};

    // 신경망 구조 정의
    int layers_config[] = {2, 2, 1}; // 입력층, 은닉층, 출력층의 뉴런 수
    // (create_neural_network를 이 배열에 맞게 수정했다고 가정)
    NeuralNetwork* nn = create_neural_network_flexible(3, layers_config);

    // 학습 파라미터
    int epochs = 50000;
    double learning_rate = 0.1;

    // 2. 훈련 (Training Loop)
    printf("훈련 시작...\n");
    for (int i = 0; i < epochs; i++) {
        double total_loss = 0;
        // 4개의 데이터셋을 모두 사용하여 훈련
        for (int j = 0; j < 4; j++) {
            // 순전파
            forward_propagate(nn, inputs[j]);

            // 손실 계산 (모니터링용)
            total_loss += pow(outputs[j][0] - nn->layers[nn->num_layers - 1].neurons[0].a_output, 2);

            // 역전파
            back_propagate(nn, inputs[j], outputs[j], learning_rate);
        }

        // 1000번 마다 손실 출력
        if (i % 1000 == 0) {
            printf("Epoch %d, Loss: %f\n", i, total_loss / 4.0);
        }
    }
    printf("훈련 종료!\n\n");

    // 3. 결과 확인 (Testing)
    printf("테스트 결과:\n");
    for (int i = 0; i < 4; i++) {
        forward_propagate(nn, inputs[i]);
        double prediction = nn->layers[nn->num_layers - 1].neurons[0].a_output;
        printf("입력: [%d, %d], 정답: %d, 예측: %f\n", (int)inputs[i][0], (int)inputs[i][1], (int)outputs[i][0], prediction);
    }
    
    free_neural_network(nn);
    
    return 0;
}


NeuralNetwork* create_neural_network(const int layer_num, const int neuron_num, const int weight_num) {
    if (layer_num <= 0 || neuron_num <= 0 || weight_num <= 0) {
        return NULL;
    }

    // 1. 신경망 구조체 할당
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    nn->num_layers = layer_num;

    // 2. 레이어 배열 할당
    nn->layers = malloc(sizeof(Layer) * layer_num);


    for (int i = 0; i < layer_num; i++) {
        // 포인터를 쓰지 않고 바로 배열의 요소에 접근
        nn->layers[i].num_neurons = neuron_num;
        nn->layers[i].num_weights_per_neuron = weight_num;
        nn->layers[i].neurons = malloc(sizeof(Neuron) * neuron_num);

        for (int j = 0; j < neuron_num; j++) {
            nn->layers[i].neurons[j].bias = 0; // 편향은 0으로 초기화
            nn->layers[i].neurons[j].weights = malloc(sizeof(double) * weight_num);

            for (int k = 0; k < weight_num; k++) {
                // 가중치를 -0.5 ~ 0.5 사이의 작은 랜덤 값으로 초기화
                nn->layers[i].neurons[j].weights[k] = ((double)rand() / RAND_MAX) - 0.5;
            }
        }
    }
    return nn;
}

NeuralNetwork* create_neural_network_flexible(const int num_layers, const int* layers_config) {
    if (num_layers <= 1) {
        return NULL; // 최소 2개 층(입력, 출력)은 있어야 함
    }

    // 1. 신경망 구조체 할당
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layers = malloc(sizeof(Layer) * num_layers);

    // 2. 각 층(Layer)을 순서대로 생성
    for (int i = 0; i < num_layers; i++) {
        Layer* current_layer = &nn->layers[i];
        
        // 현재 층의 뉴런 개수 설정
        current_layer->num_neurons = layers_config[i];
        current_layer->neurons = malloc(sizeof(Neuron) * current_layer->num_neurons);

        // 현재 층의 뉴런들이 가질 가중치의 개수 결정
        int num_weights_per_neuron = 0;
        if (i > 0) { // 입력층이 아닌 경우에만
            // 가중치 개수 = 이전 층의 뉴런 개수
            num_weights_per_neuron = layers_config[i - 1];
        }
        current_layer->num_weights_per_neuron = num_weights_per_neuron;

        // 3. 현재 층의 각 뉴런(Neuron)을 초기화
        for (int j = 0; j < current_layer->num_neurons; j++) {
            Neuron* current_neuron = &current_layer->neurons[j];
            
            // 입력층이 아닌 경우에만 가중치와 편향을 할당하고 초기화
            if (i > 0) {
                current_neuron->bias = 0; // 편향은 0으로 시작
                current_neuron->weights = malloc(sizeof(double) * num_weights_per_neuron);
                
                for (int k = 0; k < num_weights_per_neuron; k++) {
                    // 가중치를 -0.5 ~ 0.5 사이의 작은 랜덤 값으로 초기화
                    current_neuron->weights[k] = ((double)rand() / RAND_MAX) - 0.5;
                }
            } else {
                // 입력층 뉴런은 가중치나 편향이 없음
                current_neuron->weights = NULL;
                current_neuron->bias = 0;
            }
        }
    }
    return nn;
}

void forward_propagate(NeuralNetwork* nn, double* inputs) {
    // 1. 입력층의 출력(a_output)은 입력값 그대로 설정
    Layer* input_layer = &nn->layers[0];
    for (int i = 0; i < input_layer->num_neurons; i++) {
        input_layer->neurons[i].a_output = inputs[i];
    }

    // 2. 두 번째 층(첫 번째 은닉층)부터 마지막 층까지 순전파 계산
    for (int i = 1; i < nn->num_layers; i++) {
        Layer* prev_layer = &nn->layers[i - 1]; // 이전 층
        Layer* current_layer = &nn->layers[i];   // 현재 층

        for (int j = 0; j < current_layer->num_neurons; j++) {
            Neuron* neuron = &current_layer->neurons[j];
            double weighted_sum = 0.0;

            // 가중합 계산: 이전 층의 모든 뉴런 출력(a_output)과 가중치를 곱함
            for (int k = 0; k < prev_layer->num_neurons; k++) {
                weighted_sum += prev_layer->neurons[k].a_output * neuron->weights[k];
            }
            weighted_sum += neuron->bias;
            neuron->z_output = weighted_sum;

            // 활성화 함수 적용
            neuron->a_output = sigmoid(neuron->z_output);
        }
    }
}

void back_propagate(NeuralNetwork* nn, double* inputs, double* actual_outputs, double learning_rate) {
    // 1. 마지막 층부터 역순으로 순회
    for (int i = nn->num_layers - 1; i >= 0; i--) {
        Layer* layer = &nn->layers[i];
        
        // 2. 현재 층(i)의 각 뉴런(j)에 대해 오차 신호(delta) 계산
        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron* neuron = &layer->neurons[j];
            double error_signal;

            if (i == nn->num_layers - 1) { // 출력층의 경우
                double error = neuron->a_output - actual_outputs[j];
                error_signal = error * sigmoid_derivative(neuron->a_output);
            } else { // 숨겨진 층의 경우
                double propagated_error = 0.0;
                Layer* next_layer = &nn->layers[i + 1];
                
                for (int k = 0; k < next_layer->num_neurons; k++) {
                    // 다음 층 뉴런 k의 저장된 error_delta 값을 사용
                    propagated_error += next_layer->neurons[k].error_delta * next_layer->neurons[k].weights[j];
                }
                error_signal = propagated_error * sigmoid_derivative(neuron->a_output);
            }
            // 계산된 오차 신호를 뉴런 내부에 저장
            neuron->error_delta = error_signal; 
        }

        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron* neuron = &layer->neurons[j];

            for (int k = 0; k < layer->num_weights_per_neuron; k++) {
                // 이 가중치(k)에 해당하는 입력을 직접 찾음
                double input_for_this_weight;
                if (i == 0) {
                    // 첫 번째 층의 경우, 최초 입력을 사용
                    input_for_this_weight = inputs[k];
                } else {
                    // 숨겨진 층의 경우, 이전 층의 k번째 뉴런 출력을 사용
                    input_for_this_weight = nn->layers[i - 1].neurons[k].a_output;
                }
                
                // 그래디언트 계산 및 가중치 업데이트
                double gradient = neuron->error_delta * input_for_this_weight;
                neuron->weights[k] -= learning_rate * gradient;
            }
            
            // 편향 업데이트
            neuron->bias -= learning_rate * neuron->error_delta;
        }
    }
}

void free_neural_network(NeuralNetwork* nn) {
    // 신경망 포인터가 NULL이면 아무것도 하지 않음
    if (nn == NULL) {
        return;
    }

    // 1. 가장 안쪽부터, 각 층(Layer)을 순회
    for (int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &nn->layers[i];
        
        // 입력층이 아닌 경우에만 가중치 배열 해제
        if (i > 0) {
            // 2. 각 뉴런(Neuron)을 순회하며 가중치(weights) 배열 해제
            for (int j = 0; j < layer->num_neurons; j++) {
                if (layer->neurons[j].weights != NULL) {
                    free(layer->neurons[j].weights);
                }
            }
        }
        
        // 3. 뉴런(Neuron) 배열 해제
        if (layer->neurons != NULL) {
            free(layer->neurons);
        }
    }

    // 4. 레이어(Layer) 배열 해제
    if (nn->layers != NULL) {
        free(nn->layers);
    }

    // 5. 마지막으로 신경망(NeuralNetwork) 구조체 자체를 해제
    free(nn);
}