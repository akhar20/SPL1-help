import data.DataLoader;
import data.DataPoint;
import data.Preprocessor;
import models.decisionTree.DecisionTree;

import java.util.List;

public class Main {

    /**
     * A helper method to convert integer labels back to human-readable strings.
     * @param label The integer label (0, 1, or 2).
     * @return The string representation of the label.
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

        // --- 1. Load Data ---
        System.out.println("Step 1: Loading data from CSV...");
        DataLoader loader = new DataLoader();
        String filePath = "Processed.csv";
        List<DataPoint> allData = loader.loadData(filePath);
        System.out.println("=> Successfully loaded " + allData.size() + " data points.");

        // --- 2. Split Data ---
        System.out.println("\nStep 2: Splitting data into training (80%) and testing (20%) sets...");
        List<List<DataPoint>> splitData = Preprocessor.splitData(allData, 0.8);
        List<DataPoint> trainingSet = splitData.get(0);
        List<DataPoint> testingSet = splitData.get(1);
        System.out.println("=> Training set size: " + trainingSet.size());
        System.out.println("=> Testing set size: " + testingSet.size());

        // --- 3. Instantiate and Train Model ---
        System.out.println("\nStep 3: Training Decision Tree model...");
        DecisionTree tree = new DecisionTree(10, 2);
        tree.train(trainingSet);
        System.out.println("=> Model training complete.");

        // --- 4. Perform a "Smoke Test" Prediction ---
        System.out.println("\n--- Performing a single prediction test ---");
        if (!testingSet.isEmpty()) {
            DataPoint studentToTest = testingSet.get(117);

            int prediction = tree.predict(studentToTest);
            int actualLabel = studentToTest.getLabel();

            // Use our new helper method to get the string labels.
            String predictedLabelString = getLabelString(prediction);
            String actualLabelString = getLabelString(actualLabel);

            System.out.println("  Student Features: " + java.util.Arrays.toString(studentToTest.getFeatures()));
            System.out.println("  Model's Prediction: " + predictedLabelString);
            System.out.println("  Actual Label:       " + actualLabelString);
        } else {
            System.out.println("Testing set is empty, cannot perform prediction test.");
        }
    }
}