import data.DataLoader;
import data.DataPoint;
import data.Preprocessor;
import models.decisionTree.*;
import models.logisticRegression.*;
import models.knn.*;

import java.util.List;

public class Main {

    /**
     * A helper method to convert integer labels back to human-readable strings.
     */
    private static String getLabelString(int label) {
        if (label == 0) {
            return "Low Stress";
        } else if (label == 1) {
            return "Moderate Stress";
        } else if (label == 2) {
            return "High Perceived Stress";
        } else {
            return "Unknown";
        }
    }

    public static void main(String[] args) {
        System.out.println("--- Mental Health Stress Prediction System ---");

        // --- 1. Load and Split Data ---
        System.out.println("Step 1: Loading and splitting data...");
        DataLoader loader = new DataLoader();
        String filePath = "Processed.csv";
        List<DataPoint> allData = loader.loadData(filePath);

        List<List<DataPoint>> splitData = Preprocessor.splitData(allData, 0.8);
        List<DataPoint> trainingSet = splitData.get(0);
        List<DataPoint> testingSet = splitData.get(1);

        System.out.println("=> Loaded " + allData.size() + " data points.");
        System.out.println("=> Training set size: " + trainingSet.size());
        System.out.println("=> Testing set size: " + testingSet.size());

        // Get data dimensions needed for model initialization
        int numFeatures = trainingSet.get(0).getFeatureCount();
        int numClasses = 3; // Low, Moderate, High

        // --- 2. Train All Models ---
        System.out.println("\nStep 2: Training all models...");

        // Train Decision Tree
        DecisionTree tree = new DecisionTree(10, 2); // maxDepth=10, minSamplesSplit=2
        tree.train(trainingSet);
        System.out.println("=> Decision Tree training complete.");

        // Train Logistic Regression
        LogisticRegression logReg = new LogisticRegression(numFeatures, numClasses, 0.01, 100);
        logReg.train(trainingSet);
        System.out.println("=> Logistic Regression training complete.");

        // Train KNN
        KNN knn = new KNN(5); // Using K=5 as a starting point
        knn.train(trainingSet);
        System.out.println("=> KNN training complete (data stored).");

        // --- 3. Perform a Single Prediction "Smoke Test" ---
        System.out.println("\n--- Performing a single prediction test on one unseen student ---");
        if (!testingSet.isEmpty()) {
            DataPoint studentToTest = testingSet.get(0);

            System.out.println("  Student Features: " + java.util.Arrays.toString(studentToTest.getFeatures()));
            System.out.println("--------------------------------------------------");
            System.out.println("  Actual Label:       " + getLabelString(studentToTest.getLabel()));
            System.out.println("--------------------------------------------------");

            // Get a prediction from each model
            int dtPrediction = tree.predict(studentToTest);
            System.out.println("  Decision Tree Prediction:     " + getLabelString(dtPrediction));

            int lrPrediction = logReg.predict(studentToTest);
            System.out.println("  Logistic Regression Prediction: " + getLabelString(lrPrediction));

            int knnPrediction = knn.predict(studentToTest);
            System.out.println("  KNN (k=5) Prediction:         " + getLabelString(knnPrediction));

        } else {
            System.out.println("Testing set is empty, cannot perform prediction test.");
        }
    }
}