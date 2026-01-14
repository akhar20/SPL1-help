import data.DataLoader;
import data.DataPoint;
import data.Preprocessor;
import models.decisionTree.DecisionTree;
import models.logisticRegression.LogisticRegression; // <-- ADD THIS IMPORT

import java.util.List;

public class Main {

    private static String getLabelString(int label) {
        if (label == 0) return "Low Stress";
        if (label == 1) return "Moderate Stress";
        if (label == 2) return "High Perceived Stress";
        return "Unknown";
    }

    public static void main(String[] args) {
        System.out.println("--- Mental Health Stress Prediction System ---");

        // --- 1. Load Data ---
        System.out.println("Step 1: Loading data...");
        DataLoader loader = new DataLoader();
        String filePath = "Processed.csv";
        List<DataPoint> allData = loader.loadData(filePath);
        System.out.println("=> Loaded " + allData.size() + " data points.");

        // --- 2. Split Data ---
        System.out.println("\nStep 2: Splitting data...");
        List<List<DataPoint>> splitData = Preprocessor.splitData(allData, 0.8);
        List<DataPoint> trainingSet = splitData.get(0);
        List<DataPoint> testingSet = splitData.get(1);
        System.out.println("=> Training set size: " + trainingSet.size());
        System.out.println("=> Testing set size: " + testingSet.size());

        // --- 3. Get Data Dimensions ---
        // We need these values to initialize our models.
        int numFeatures = trainingSet.get(0).getFeatureCount();
        int numClasses = 3; // Low, Moderate, High

        // --- 4. Train and Test Decision Tree ---
        System.out.println("\n--- Training Decision Tree model ---");
        DecisionTree tree = new DecisionTree(10, 2);
        tree.train(trainingSet);
        System.out.println("=> Decision Tree training complete.");

        // --- 5. Train and Test Logistic Regression ---
        // (ADD THIS NEW SECTION)
        System.out.println("\n--- Training Logistic Regression model ---");
        // Hyperparameters: learningRate=0.01, epochs=100
        LogisticRegression logReg = new LogisticRegression(numFeatures, numClasses, 0.01, 100);
        logReg.train(trainingSet);
        System.out.println("=> Logistic Regression training complete.");


        // --- 6. Perform Prediction Smoke Test ---
        System.out.println("\n--- Performing a single prediction test ---");
        if (!testingSet.isEmpty()) {
            DataPoint studentToTest = testingSet.get(54);

            System.out.println("  Student Features: " + java.util.Arrays.toString(studentToTest.getFeatures()));
            System.out.println("  Actual Label:       " + getLabelString(studentToTest.getLabel()));

            // Get prediction from Decision Tree
            int dtPrediction = tree.predict(studentToTest);
            System.out.println("  Decision Tree Prediction: " + getLabelString(dtPrediction));

            // Get prediction from Logistic Regression (ADD THIS PART)
            int lrPrediction = logReg.predict(studentToTest);
            System.out.println("  Logistic Regression Prediction: " + getLabelString(lrPrediction));

        } else {
            System.out.println("Testing set is empty, cannot perform prediction test.");
        }
    }
}