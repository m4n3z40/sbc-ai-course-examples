import SimplePerceptron from './SimplePerceptron.mjs';
import SimpleNeuralNetwork from './SimpleNeuralNetwork.mjs';

// Dados de treinamento para a operação AND
const andData = [
    {inputs: [0, 0], target: 0},
    {inputs: [0, 1], target: 0},
    {inputs: [1, 0], target: 0},
    {inputs: [1, 1], target: 1},
];

// criar e treinar o perceptron
const andPerceptron = new SimplePerceptron(2);
andPerceptron.train(andData, 100);

// Testar as previsões
console.log('PERCEPTRON AND(0,0):', andPerceptron.predict([0, 0])); // 0
console.log('PERCEPTRON AND(0,1):', andPerceptron.predict([0, 1])); // 0
console.log('PERCEPTRON AND(1,0):', andPerceptron.predict([1, 0])); // 0
console.log('PERCEPTRON AND(1,1):', andPerceptron.predict([1, 1])); // 1

// Dados de treinamento para a operação XOR
const xorData = [
    {inputs: [0, 0], target: 0},
    {inputs: [0, 1], target: 1},
    {inputs: [1, 0], target: 1},
    {inputs: [1, 1], target: 0},
];

// Criar e treinar o perceptron
const xorPerceptron = new SimplePerceptron(2);
xorPerceptron.train(xorData, 100);

// Testar as previsões
console.log('PERCEPTRON XOR(0,0):', xorPerceptron.predict([0, 0])); // 0
console.log('PERCEPTRON XOR(0,1):', xorPerceptron.predict([0, 1])); // 1
console.log('PERCEPTRON XOR(1,0):', xorPerceptron.predict([1, 0])); // 1
console.log('PERCEPTRON XOR(1,1):', xorPerceptron.predict([1, 1])); // 0

// Nao é possível resolver o problema XOR com um único perceptron
// Isso ocorre porque o XOR não é linearmente separável
// No entanto, podemos resolver o problema XOR com uma rede neural de múltiplas camadas

// Criar uma rede neural com 2 entradas, 4 neurônios ocultos e 1 saída
const nn = new SimpleNeuralNetwork(2, 4, 1);

// Dados de treinamento (operação XOR)
const trainingData = [
    { inputs: [0, 0], targets: [0] },
    { inputs: [0, 1], targets: [1] },
    { inputs: [1, 0], targets: [1] },
    { inputs: [1, 1], targets: [0] }
];

// Treinar a rede
for(let i = 0; i < 100000; i++) {
    const data = trainingData[Math.floor(Math.random() * trainingData.length)];
    nn.train(data.inputs, data.targets);
}

// Testar a rede
console.log('NN XOR(0,0):', nn.predict([0, 0])); // ~0
console.log('NN XOR(0,1):', nn.predict([0, 1])); // ~1
console.log('NN XOR(1,0):', nn.predict([1, 0])); // ~1
console.log('NN XOR(1,1):', nn.predict([1, 1])); // ~0
