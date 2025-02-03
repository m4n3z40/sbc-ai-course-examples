/**
 * Simple Perceptron implementation for educational purposes
 * 
 * A perceptron is a fundamental unit of neural networks that learns weights
 * through training to make binary classifications
 */
export default class SimplePerceptron {
  /**
   * @param {number} inputSize - Number of input features
   * @param {number} [learningRate=0.1] - Learning rate (0 < lr <= 1)
   */
  constructor(inputSize, learningRate = 0.1) {
    // Initialize weights with small random values (including bias weight)
    this.weights = Array.from(
      { length: inputSize + 1 }, // +1 for bias term
      () => Math.random() * 2 - 1 // Random values between -1 and 1
    );
    this.learningRate = learningRate;
  }

  /**
   * Step activation function - returns 1 if input >= 0, else 0
   * @param {number} sum - Weighted sum of inputs
   * @returns {number} 1 or 0
   */
  activation(sum) {
    return sum >= 0 ? 1 : 0;
  }

  /**
   * Make prediction for given inputs
   * @param {number[]} inputs - Array of input values (length must match inputSize)
   * @returns {number} Prediction (0 or 1)
   */
  predict(inputs) {
    // Add bias input (1) to beginning of inputs array
    const extendedInputs = [1, ...inputs];
    
    // Calculate weighted sum: sum(weight_i * input_i)
    const sum = this.weights.reduce(
      (total, weight, i) => total + weight * extendedInputs[i],
      0
    );
    
    return this.activation(sum);
  }

  /**
   * Train the perceptron on a single example
   * @param {number[]} inputs - Training inputs
   * @param {number} target - Expected output (0 or 1)
   */
  trainSingle(inputs, target) {
    const prediction = this.predict(inputs);
    const error = target - prediction;
    
    // Update weights: w_i += learningRate * error * x_i
    this.weights = this.weights.map((weight, i) => {
      const input = i === 0 ? 1 : inputs[i - 1]; // First weight is bias (input=1)
      return weight + this.learningRate * error * input;
    });
  }

  /**
   * Train the perceptron on a dataset
   * @param {Array} trainingData - Array of {inputs: number[], target: number}
   * @param {number} epochs - Number of training passes through the dataset
   */
  train(trainingData, epochs) {
    for (let epoch = 1; epoch <= epochs; epoch++) {
      for (const {inputs, target} of trainingData) {
        this.trainSingle(inputs, target);
      }
    }
  }
}
