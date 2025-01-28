// Classe da Rede Neural
export default class SimpleNeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        
        // Inicializar pesos com valores aleatórios
        this.weightsIH = Array(this.hiddenNodes).fill().map(() => 
            Array(this.inputNodes).fill().map(() => Math.random() * 2 - 1)
        );
        this.weightsHO = Array(this.outputNodes).fill().map(() => 
            Array(this.hiddenNodes).fill().map(() => Math.random() * 2 - 1)
        );
        
        // Taxa de aprendizado
        this.learningRate = 0.1;
    }
    
    // Função de ativação (sigmoid)
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    // Derivada da sigmoid
    sigmoidDerivative(x) {
        return x * (1 - x);
    }
    
    // Feedforward
    predict(inputArray) {
        // Calcular saídas da camada oculta
        let hidden = this.weightsIH.map(row => 
            row.reduce((sum, weight, i) => sum + weight * inputArray[i], 0)
        ).map(this.sigmoid);
        
        // Calcular saídas finais
        let outputs = this.weightsHO.map(row =>
            row.reduce((sum, weight, i) => sum + weight * hidden[i], 0)
        ).map(this.sigmoid);
        
        return outputs;
    }
    
    // Treinar a rede
    train(inputArray, targetArray) {
        // Feedforward
        let hidden = this.weightsIH.map(row => 
            row.reduce((sum, weight, i) => sum + weight * inputArray[i], 0)
        ).map(this.sigmoid);
        
        let outputs = this.weightsHO.map(row =>
            row.reduce((sum, weight, i) => sum + weight * hidden[i], 0)
        ).map(this.sigmoid);
        
        // Calcular erros
        let outputErrors = targetArray.map((target, i) => target - outputs[i]);
        
        // Backpropagation
        // Ajustar pesos camada oculta -> saída
        for(let i = 0; i < this.outputNodes; i++) {
            for(let j = 0; j < this.hiddenNodes; j++) {
                this.weightsHO[i][j] += this.learningRate * 
                    outputErrors[i] * 
                    this.sigmoidDerivative(outputs[i]) * 
                    hidden[j];
            }
        }
        
        // Calcular erros da camada oculta
        let hiddenErrors = Array(this.hiddenNodes).fill(0);
        for(let i = 0; i < this.hiddenNodes; i++) {
            for(let j = 0; j < this.outputNodes; j++) {
                hiddenErrors[i] += outputErrors[j] * this.weightsHO[j][i];
            }
        }
        
        // Ajustar pesos entrada -> camada oculta
        for(let i = 0; i < this.hiddenNodes; i++) {
            for(let j = 0; j < this.inputNodes; j++) {
                this.weightsIH[i][j] += this.learningRate * 
                    hiddenErrors[i] * 
                    this.sigmoidDerivative(hidden[i]) * 
                    inputArray[j];
            }
        }
    }
}