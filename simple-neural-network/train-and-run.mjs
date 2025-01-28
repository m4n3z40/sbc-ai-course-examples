import SimpleNeuralNetwork from './SimpleNeuralNetwork.mjs';

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
console.log(nn.predict([0, 0])); // Deve ser próximo de 0
console.log(nn.predict([0, 1])); // Deve ser próximo de 1
console.log(nn.predict([1, 0])); // Deve ser próximo de 1
console.log(nn.predict([1, 1])); // Deve ser próximo de 0
