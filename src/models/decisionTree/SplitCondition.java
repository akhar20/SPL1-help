//Concept: A Decision Tree is built on questions. We need a way to represent
// a single question like "Is the student's CGPA less than or equal to 3.0?".
//Components: Every question needs two pieces of information:
//Which feature to look at (CGPA).
//What value to compare against (3.0).
//Outcome: The question always has a binary (yes/no) answer.


package models.decisionTree;

/**
 * Represents a question used to partition a dataset, e.g., "Is feature 2 (CGPA) <= 3.2?".
 * This is the condition that will be stored in each Decision Node.
 */
public class SplitCondition {
    // We use the feature's index to be general, so we don't have to hardcode "age", "cgpa", etc.
    private final int featureIndex;

    // The value to compare the feature against.
    private final double value;

    public SplitCondition(int featureIndex, double value) {
        this.featureIndex = featureIndex;
        this.value = value;
    }

    /**
     * Checks if a given student's features match this condition.
     * @param features The features of a single student (e.g., from a DataPoint object).
     * @return true if the condition is met, false otherwise.
     */
    public boolean matches(double[] features) {
        // Our rule is: if the feature value is less than or equal to the split value, it's a "match" (go left).
        return features[this.featureIndex] <= this.value;
    }

    // Getter methods are useful for debugging and building the tree.
    public int getFeatureIndex() {
        return featureIndex;
    }

    public double getValue() {
        return value;
    }
}