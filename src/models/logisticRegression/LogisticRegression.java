package models.logisticRegression;

import data.DataPoint;
import java.util.List;
import java.util.Random;

/**
 * A from-scratch implementation of Multi-class Logistic Regression (Softmax Regression).
 */
public class LogisticRegression {

    private double[][] weights;
    private double[] biases;
    private final double learningRate;
    private final int epochs;

    public LogisticRegression(int numFeatures, int numClasses, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.weights = new double[numFeatures][numClasses];
        this.biases = new double[numClasses];

        Random rand = new Random();
        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numClasses; j++) {
                this.weights[i][j] = (rand.nextDouble() - 0.5) / 50.0;
            }
        }
    }

    /**
     * The main training method using Stochastic Gradient Descent.
     * @param trainingData The list of DataPoints to learn from.
     */
    public void train(List<DataPoint> trainingData) { // <-- FIX 1: Return type is void
        int numFeatures = weights.length;
        int numClasses = biases.length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (DataPoint dp : trainingData) {
                double[] features = dp.getFeatures();

                // Step 1: Forward Pass
                double[] scores = calculateScores(features);
                double[] probabilities = softmax(scores);

                // Step 2: Calculate Error
                double[] errorSignal = new double[numClasses];
                int trueLabel = dp.getLabel();
                for (int j = 0; j < numClasses; j++) {
                    double y_true = (j == trueLabel) ? 1.0 : 0.0;
                    errorSignal[j] = probabilities[j] - y_true;
                }

                // Step 3 & 4: Update Parameters
                for (int j = 0; j < numClasses; j++) {
                    biases[j] -= learningRate * errorSignal[j];
                }
                for (int i = 0; i < numFeatures; i++) {
                    for (int j = 0; j < numClasses; j++) {
                        weights[i][j] -= learningRate * features[i] * errorSignal[j];
                    }
                }
            }
        }
    }

    /**
     * The "Forward Pass" for a single data point to make a prediction.
     * @param dataPoint The data point to classify.
     * @return The predicted class index (0, 1, or 2).
     */
    public int predict(DataPoint dataPoint) {
        // --- Step 1: Calculate the raw scores (Z) for each class ---
        double[] scores = calculateScores(dataPoint.getFeatures()); // <-- FIX 3: Re-use helper method

        // --- Step 2: Convert scores to probabilities (P) using softmax ---
        double[] probabilities = softmax(scores);

        // --- Step 3: Find the class with the highest probability ---
        int bestClass = -1;
        double maxProbability = -1.0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProbability) {
                maxProbability = probabilities[i];
                bestClass = i;
            }
        }
        return bestClass;
    }

    // =================================================================
    // HELPER METHODS
    // =================================================================

    /**
     * A private helper to calculate the raw scores. (Z = X • W + b)
     */
    private double[] calculateScores(double[] features) {
        double[] scores = new double[biases.length];
        for (int j = 0; j < biases.length; j++) {
            scores[j] = biases[j];
            for (int i = 0; i < features.length; i++) {
                scores[j] += features[i] * weights[i][j];
            }
        }
        return scores;
    }

    /**
     * A helper method to compute the Softmax function.
     */
    private double[] softmax(double[] scores) {
        double[] probabilities = new double[scores.length];
        double maxScore = scores[0];
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
            }
        }
        double sum = 0.0;
        for (int i = 0; i < scores.length; i++) {
            probabilities[i] = Math.exp(scores[i] - maxScore);
            sum += probabilities[i];
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
        return probabilities;
    }
}