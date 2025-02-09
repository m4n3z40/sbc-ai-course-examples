import fs from 'fs';
import { parse } from 'csv-parse/sync';
import SimplePerceptron from './SimplePerceptron.mjs';
import SimpleNeuralNetwork from './SimpleNeuralNetwork.mjs';
import RevenuePredictor from './RevenuePredictor.mjs';

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

// Função para ler e parsear o CSV
function loadCSVData(filename) {
    const fileContent = fs.readFileSync(filename, 'utf-8');
    const records = parse(fileContent, {
        columns: true,
        skip_empty_lines: true,
        cast: true // Converte automaticamente strings para números quando possível
    });
    
    return records.map(record => ({
        dayOfYear: parseInt(record.dayOfYear),
        dayOfMonth: parseInt(record.dayOfMonth),
        dayOfWeek: parseInt(record.dayOfWeek),
        month: parseInt(record.month),
        isHoliday: parseInt(record.isHoliday),
        revenue: parseFloat(record.revenue)
    }));
}

// Carregar dados do CSV
const salesData = loadCSVData('./simple-neural-network/sales-data.csv');

// Criar e treinar o preditor
const predictor = new RevenuePredictor();
predictor.train(salesData, 1000); // aumentar o número de épocas para melhorar a precisão

// Testar algumas previsões

// Prever faturamento para Natal (25/12)
const christmasPrediction = predictor.predict({
    dayOfYear: 359,
    dayOfMonth: 25,
    dayOfWeek: 4,
    month: 12,
    isHoliday: 1
});

// Prever faturamento para um dia comum (15/07)
const regularDayPrediction = predictor.predict({
    dayOfYear: 196,
    dayOfMonth: 15,
    dayOfWeek: 3,
    month: 7,
    isHoliday: 0
});

// Prever faturamento para Ano Novo (01/01)
const newYearPrediction = predictor.predict({
    dayOfYear: 1,
    dayOfMonth: 1,
    dayOfWeek: 1,
    month: 1,
    isHoliday: 1
});

console.log('\nPrevisões de Faturamento:');
console.log('-------------------------');
console.log(`Natal (25/12): R$ ${christmasPrediction.toFixed(2)}`);
console.log(`Dia Comum (15/07): R$ ${regularDayPrediction.toFixed(2)}`);
console.log(`Ano Novo (01/01): R$ ${newYearPrediction.toFixed(2)}`);

// Validação com dados reais
const realChristmas = salesData.find(d => d.dayOfYear === 359);
const realRegularDay = salesData.find(d => d.dayOfYear === 196);
const realNewYear = salesData.find(d => d.dayOfYear === 1);

console.log('\nComparação com Valores Reais:');
console.log('-------------------------');
console.log(`Natal - Real: R$ ${realChristmas.revenue.toFixed(2)} | Previsto: R$ ${christmasPrediction.toFixed(2)}`);
console.log(`Dia Comum - Real: R$ ${realRegularDay.revenue.toFixed(2)} | Previsto: R$ ${regularDayPrediction.toFixed(2)}`);
console.log(`Ano Novo - Real: R$ ${realNewYear.revenue.toFixed(2)} | Previsto: R$ ${newYearPrediction.toFixed(2)}`);

// Calcular erro médio percentual
const predictions = [
    { real: realChristmas.revenue, predicted: christmasPrediction },
    { real: realRegularDay.revenue, predicted: regularDayPrediction },
    { real: realNewYear.revenue, predicted: newYearPrediction }
];

const averageError = predictions.reduce((acc, curr) => {
    return acc + Math.abs((curr.predicted - curr.real) / curr.real) * 100;
}, 0) / predictions.length;

console.log(`\nErro Médio Percentual: ${averageError.toFixed(2)}%`);
