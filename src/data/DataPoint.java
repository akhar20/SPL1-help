package data;
/**
 * Represents a single row of the dataset after preprocessing.
 * It contains the numerical features and the integer-encoded class label.
 */
public class DataPoint {

    // An array of all the feature values for one student.
    // The order is important and must be consistent. For example:
    // [age, gender, academicYear, cgpa, scholarship, anxietyValue, depressionValue]
    private final double[] features;

    // The integer code for the stress level.
    // e.g., 0 for "Low", 1 for "Moderate", 2 for "High"
    private final int label;

    /**
     * Constructor for a new DataPoint.
     * @param features An array of numerical feature values.
     * @param label The integer-encoded class label.
     */
    public DataPoint(double[] features, int label) {
        this.features = features;
        this.label = label;
    }

    /**
     * Gets the features for this data point.
     * @return The array of feature values.
     */
    public double[] getFeatures() {
        return features;
    }

    /**
     * Gets the label for this data point.
     * @return The integer class label.
     */
    public int getLabel() {
        return label;
    }

    /**
     * A helper method to get a specific feature by its index.
     * @param index The index of the feature in the array.
     * @return The value of the feature at that index.
     */
    public double getFeature(int index) {
        if (index >= 0 && index < this.features.length) {
            return this.features[index];
        }
        // Throw an exception for invalid index to catch errors early.
        throw new IndexOutOfBoundsException("Feature index " + index + " is out of bounds.");
    }

    /**
     * Gets the total number of features for this data point.
     * @return The length of the features array.
     */
    public int getFeatureCount() {
        return this.features.length;
    }
}