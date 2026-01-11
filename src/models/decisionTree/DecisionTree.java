package models.decisionTree;

import data.DataPoint;

import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * The main class that encapsulates the entire Decision Tree algorithm.
 * It is responsible for building the tree (training) and using it to make predictions.
 */
public class DecisionTree {

    // The 'root' is the very first node of the tree. It's the entry point for all predictions.
    // It is null until the train() method is called.
    private Node root;

    // A hyperparameter to control the maximum depth of the tree.
    // This is a crucial setting to prevent the tree from becoming too complex and "overfitting" the data.
    private final int maxDepth;

    // A hyperparameter that sets a minimum number of data points required to attempt a split.
    // This prevents the tree from creating branches for very small, insignificant groups of data.
    private final int minSamplesSplit;

    /**
     * Constructor for the DecisionTree. This is where we set the "rules" for our tree builder.
     * @param maxDepth The maximum number of questions in a single path from the root to a leaf.
     * @param minSamplesSplit The minimum number of students required in a group to consider splitting it further.
     */
    public DecisionTree(int maxDepth, int minSamplesSplit) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
    }

    /**
     * The main public method to start the training process.
     * It takes the training data and begins the recursive tree-building process.
     * @param trainingData A List of DataPoint objects to build the tree from.
     */
    public void train(List<DataPoint> trainingData) {
        // The entire, fully-assembled tree structure is returned by buildTree and stored in our 'root' field.
        this.root = buildTree(trainingData, 0);
    }

    /**
     * The main recursive method that builds the tree.
     * This will be implemented in the next step. It's the core of the "assembly line."
     */
    // Replace the empty buildTree method in DecisionTree.java with this.

    private Node buildTree(List<DataPoint> data, int currentDepth) {
        // === BASE CASES: These are the stopping conditions for the recursion ===

        // Condition 1: Have we reached the maximum allowed depth?
        // Condition 2: Is the group of data too small to be worth splitting?
        // Condition 3: Is the group of data already perfectly pure (all the same class)?
        if (currentDepth >= maxDepth || data.size() < minSamplesSplit || isPure(data)) {
            // If any stopping condition is met, we create a Leaf Node.
            // The prediction for this leaf is the most common class in the current data.
            int leafPrediction = majorityVote(data);
            return new Node(leafPrediction);
        }

        // === RECURSIVE STEP: Find the best split and continue building ===

        // Use our "master tool" to find the best possible question for the current dataset.
        BestSplitResult bestSplit = findBestSplit(data);

        // Condition 4: Another stopping condition. If findBestSplit couldn't find a split that
        // provided any information gain (gain=0), it's not worth splitting further.
        if (bestSplit.getGain() <= 0) {
            int leafPrediction = majorityVote(data);
            return new Node(leafPrediction);
        }

        // If we've made it this far, we are creating a Decision Node.

        // Recursively call this function to build the "yes" (left) branch.
        // We pass the left subset of data and increment the depth.
        Node leftChild = buildTree(bestSplit.getLeftData(), currentDepth + 1);

        // Recursively call this function to build the "no" (right) branch.
        Node rightChild = buildTree(bestSplit.getRightData(), currentDepth + 1);

        // Create a new Decision Node that holds the best question and the two sub-trees we just built.
        // This node is then returned up the chain to the function that called it.
        return new Node(bestSplit.getCondition(), leftChild, rightChild);
    }

    // In DecisionTree.java, replace the empty predict method and add the new traverseTree method.

    /**
     * The main public method to make a prediction on a new, unseen data point.
     * It simply starts the recursive traversal from the root of the tree.
     * @param dataPoint The student's data to classify.
     * @return The predicted class label (0, 1, or 2).
     */
    public int predict(DataPoint dataPoint) {
        // Start the recursive walk down the tree, beginning at the root.
        return traverseTree(dataPoint.getFeatures(), this.root);
    }

    /**
     * The private recursive method that walks down the tree to find a prediction.
     * @param features The features of the student we are classifying.
     * @param node The current node we are looking at.
     * @return The prediction from the leaf node that is eventually reached.
     */
    private int traverseTree(double[] features, Node node) {
        // --- BASE CASE ---
        // If the current node is a leaf, we have reached our destination.
        // Return the prediction stored in this leaf.
        if (node.isLeaf()) {
            return node.getPrediction();
        }

        // --- RECURSIVE STEP ---
        // If it's not a leaf, it's a decision node. We must ask its question.
        // Use the node's SplitCondition to check the student's features.
        if (node.getSplitCondition().matches(features)) {
            // If the condition is true (it matches), we continue traversing down the LEFT child.
            return traverseTree(features, node.getLeftChild());
        } else {
            // If the condition is false, we continue traversing down the RIGHT child.
            return traverseTree(features, node.getRightChild());
        }
    }


    // =================================================================
    // TOOLBOX OF HELPER METHODS
    // =================================================================

    /**
     * TOOL #1: The "Purity Scanner" (Array-based implementation).
     * Calculates the Gini Impurity for a given list of data points.
     * @param data A list of DataPoints.
     * @return A double between 0 (perfectly pure) and a max value.
     */
    private double calculateGini(List<DataPoint> data) {
        if (data.isEmpty()) {
            return 0.0;
        }

        // Create an array to hold the counts. Index 0 for class 0, Index 1 for class 1, etc.
        // We assume a maximum of 3 classes (0, 1, 2), so an array of size 3 is sufficient.
        int[] classCounts = new int[3];

        // Step 1: Count the occurrences of each class label.
        for (DataPoint dp : data) {
            int label = dp.getLabel();
            if (label >= 0 && label < classCounts.length) {
                classCounts[label]++; // Increment the count at the index corresponding to the label.
            }
        }

        // Step 2 & 3: Calculate the sum of squared proportions.
        double sumOfSquares = 0.0;
        int totalSamples = data.size();
        for (int count : classCounts) {
            if (count > 0) { // Only calculate for classes that are actually present.
                double proportion = (double) count / totalSamples;
                sumOfSquares += proportion * proportion;
            }
        }

        // Step 4: The final Gini formula.
        return 1.0 - sumOfSquares;
    }
    /**
     * TOOL #2: The "Final Answer Determiner" (Array-based implementation).
     * Finds the most frequent class label in a list of data.
     * @param data A list of DataPoints.
     * @return The integer label of the most frequent class.
     */
    private int majorityVote(List<DataPoint> data) {
        if (data.isEmpty()) {
            return -1; // Return an invalid label if the list is empty.
        }

        // Use an array to store the counts of each label.
        int[] classCounts = new int[3];
        for (DataPoint dp : data) {
            int label = dp.getLabel();
            if (label >= 0 && label < classCounts.length) {
                classCounts[label]++;
            }
        }

        // Find the index in the array that has the highest count.
        int majorityLabel = -1;
        int maxCount = -1;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > maxCount) {
                maxCount = classCounts[i];
                majorityLabel = i; // The index is the label.
            }
        }

        return majorityLabel;
    }

    /**
     * TOOL #3: The "Master Machine".
     * This is the core of the algorithm. It iterates through every possible question (every feature and every unique value)
     * to find the one that results in the highest Information Gain (the biggest reduction in Gini impurity).
     * @param data The current list of data points to be split.
     * @return A BestSplitResult object containing the best question and the resulting data subgroups.
     */
    private BestSplitResult findBestSplit(List<DataPoint> data) {
        double bestGain = 0.0;
        SplitCondition bestCondition = null;
        List<DataPoint> bestLeftData = null;
        List<DataPoint> bestRightData = null;

        // First, calculate the impurity of the current group before any splits.
        double parentGini = calculateGini(data);
        if (data.isEmpty()) {
            return new BestSplitResult(null, 0, null, null);
        }
        int numFeatures = data.get(0).getFeatureCount();

        // Loop 1: Go through each feature (e.g., Age, CGPA, etc.).
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            // Find all unique values for this feature to use as potential split points.
            Set<Double> uniqueValues = new HashSet<>();
            for (DataPoint dp : data) {
                uniqueValues.add(dp.getFeature(featureIndex));
            }

            // Loop 2: Go through each unique value to create a question.
            for (double value : uniqueValues) {
                SplitCondition condition = new SplitCondition(featureIndex, value);

                // Partition the data based on the current question.
                List<DataPoint> leftData = new ArrayList<>();
                List<DataPoint> rightData = new ArrayList<>();
                for (DataPoint dp : data) {
                    if (condition.matches(dp.getFeatures())) {
                        leftData.add(dp);
                    } else {
                        rightData.add(dp);
                    }
                }

                // Don't consider splits that don't actually divide the data.
                if (leftData.isEmpty() || rightData.isEmpty()) {
                    continue;
                }

                // Calculate the weighted average impurity of the two new groups.
                double pLeft = (double) leftData.size() / data.size();
                double weightedGini = pLeft * calculateGini(leftData) + (1.0 - pLeft) * calculateGini(rightData);

                // Information Gain is the reduction in impurity.
                double informationGain = parentGini - weightedGini;

                // If this split is the best one we've seen so far, save it.
                if (informationGain > bestGain) {
                    bestGain = informationGain;
                    bestCondition = condition;
                    bestLeftData = leftData;
                    bestRightData = rightData;
                }
            }
        }
        // Return a "tote box" containing all the details of the best split found.
        return new BestSplitResult(bestCondition, bestGain, bestLeftData, bestRightData);
    }
    private boolean isPure(List<DataPoint> data) {
        if (data.size() <= 1) {
            return true;
        }
        int firstLabel = data.get(0).getLabel();
        for (int i = 1; i < data.size(); i++) {
            if (data.get(i).getLabel() != firstLabel) {
                return false;
            }
        }
        return true;
    }
}