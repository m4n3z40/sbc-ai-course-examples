import SimpleNeuralNetwork from './SimpleNeuralNetwork.mjs';

export default class RevenuePredictor {
    constructor() {
        // 5 nós de entrada (removidos orderCount e revenue do input)
        // 10 nós na camada oculta
        // 1 nó de saída (previsão de faturamento)
        this.network = new SimpleNeuralNetwork(5, 10, 1);
        
        // Armazenar valores máximos e mínimos para normalização
        this.maxValues = {
            dayOfYear: 365,
            dayOfMonth: 31,
            dayOfWeek: 7,
            month: 12,
            isHoliday: 1,
            revenue: 0 // mantido apenas para normalização do target
        };
        
        this.minValues = {
            dayOfYear: 1,
            dayOfMonth: 1,
            dayOfWeek: 1,
            month: 1,
            isHoliday: 0,
            revenue: 0 // mantido apenas para normalização do target
        };
    }

    // Normalizar dados entre 0 e 1
    normalize(value, min, max) {
        return (value - min) / (max - min);
    }

    // Desnormalizar dados
    denormalize(normalizedValue, min, max) {
        return normalizedValue * (max - min) + min;
    }

    // Processar dados do CSV
    processData(data) {
        // Encontrar valor máximo apenas para revenue
        data.forEach(row => {
            this.maxValues.revenue = Math.max(this.maxValues.revenue, row.revenue);
        });

        return data.map(row => ({
            input: [
                this.normalize(row.dayOfYear, this.minValues.dayOfYear, this.maxValues.dayOfYear),
                this.normalize(row.dayOfMonth, this.minValues.dayOfMonth, this.maxValues.dayOfMonth),
                this.normalize(row.dayOfWeek, this.minValues.dayOfWeek, this.maxValues.dayOfWeek),
                this.normalize(row.month, this.minValues.month, this.maxValues.month),
                row.isHoliday // já está entre 0 e 1
            ],
            target: [this.normalize(row.revenue, this.minValues.revenue, this.maxValues.revenue)]
        }));
    }

    // Treinar o modelo
    train(data, epochs = 1000) {
        const processedData = this.processData(data);
        
        for (let i = 0; i < epochs; i++) {
            processedData.forEach(data => {
                this.network.train(data.input, data.target);
            });
        }
    }

    // Fazer previsão
    predict(input) {
        const normalizedInput = [
            this.normalize(input.dayOfYear, this.minValues.dayOfYear, this.maxValues.dayOfYear),
            this.normalize(input.dayOfMonth, this.minValues.dayOfMonth, this.maxValues.dayOfMonth),
            this.normalize(input.dayOfWeek, this.minValues.dayOfWeek, this.maxValues.dayOfWeek),
            this.normalize(input.month, this.minValues.month, this.maxValues.month),
            input.isHoliday
        ];

        const prediction = this.network.predict(normalizedInput);
        return this.denormalize(prediction[0], this.minValues.revenue, this.maxValues.revenue);
    }
}
