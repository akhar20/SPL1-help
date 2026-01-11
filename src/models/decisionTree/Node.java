package models.decisionTree;

public class Node {

    // --- Fields for a Decision Node ---
    private final SplitCondition splitCondition; // The question to ask
    private final Node leftChild;              // The "Yes" branch
    private final Node rightChild;             // The "No" branch

    // --- Field for a Leaf Node ---
    private final int prediction;              // The final answer

    /**
     * Constructor for creating a Decision Node (an internal question box).
     */
    public Node(SplitCondition splitCondition, Node leftChild, Node rightChild) {
        this.splitCondition = splitCondition;
        this.leftChild = leftChild;
        this.rightChild = rightChild;
        this.prediction = -1; // Use a special value like -1 to show this is not a leaf
    }

    /**
     * Constructor for creating a Leaf Node (a final answer box).
     */
    public Node(int prediction) {
        this.prediction = prediction;
        this.splitCondition = null; // A leaf has no question
        this.leftChild = null;      // A leaf has no children
        this.rightChild = null;
    }

    /**
     * Helper method to easily check if this node is a leaf.
     * @return true if this node is a leaf, false if it's a decision node.
     */
    public boolean isLeaf() {
        // A node is a leaf if it was created with the leaf constructor,
        // which means its children are null.
        return this.leftChild == null && this.rightChild == null;
    }

    // --- Getter methods to access the node's properties ---
    public SplitCondition getSplitCondition() { return splitCondition; }
    public Node getLeftChild() { return leftChild; }
    public Node getRightChild() { return rightChild; }
    public int getPrediction() { return prediction; }
}
