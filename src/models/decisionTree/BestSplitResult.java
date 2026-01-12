package models.decisionTree;

import data.DataPoint;
import java.util.List;

/**
 * A simple data-holding class (a "struct" or "record") used to return multiple values
 * from the findBestSplit method. This is cleaner than returning a generic array or map.
 */
public class BestSplitResult {

    // The question that resulted in the best split (e.g., "Is CGPA <= 3.0?").
    private final SplitCondition condition;

    // The Information Gain score achieved by this split. Storing this is useful for debugging.
    private final double gain;

    // The subset of data that matched the condition (the "yes" group).
    private final List<DataPoint> leftData;

    // The subset of data that did not match the condition (the "no" group).
    private final List<DataPoint> rightData;

    public BestSplitResult(SplitCondition condition, double gain, List<DataPoint> leftData, List<DataPoint> rightData) {
        this.condition = condition;
        this.gain = gain;
        this.leftData = leftData;
        this.rightData = rightData;
    }

    // --- Standard "Getter" methods to access the stored data ---
    public SplitCondition getCondition() { return condition; }
    public double getGain() { return gain; }
    public List<DataPoint> getLeftData() { return leftData; }
    public List<DataPoint> getRightData() { return rightData; }
}