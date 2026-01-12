package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Preprocessor {

    /**
     * Splits the full dataset into a training set and a testing set.
     * @param data The full list of DataPoints.
     * @param trainSplitRatio The proportion of data for the training set (e.g., 0.8 for 80%).
     * @return A List containing two lists: the training set at index 0, and the testing set at index 1.
     */
    public static List<List<DataPoint>> splitData(List<DataPoint> data, double trainSplitRatio) {
        // Creating a mutable copy of the data to avoid modifying the original list.
        List<DataPoint> shuffledData = new ArrayList<>(data);

        // Randomly shuffle the data. This is crucial to ensure that the train and test sets
        // are representative of the overall data and not biased by any original ordering.
        Collections.shuffle(shuffledData);

        // Calculate the index where we will split the data.
        int splitIndex = (int) (shuffledData.size() * trainSplitRatio);

        // Create the training set as a sublist from the beginning to the split index.
        List<DataPoint> trainingSet = new ArrayList<>(shuffledData.subList(0, splitIndex));

        // Create the testing set as a sublist from the split index to the end.
        List<DataPoint> testingSet = new ArrayList<>(shuffledData.subList(splitIndex, shuffledData.size()));

        // Return both sets in a container list.
        List<List<DataPoint>> result = new ArrayList<>();
        result.add(trainingSet);
        result.add(testingSet);
        return result;
    }
}
