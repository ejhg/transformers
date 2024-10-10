/**
 * https://www.geeksforgeeks.org/fasttext-working-and-implementation/
 */
public class FastTextHierarchicalClassifier
{
    private int embeddingSize;
    private Dictionary<string, float[]> wordEmbeddings;
    private Node root;

    public FastTextHierarchicalClassifier (int embeddingSize = 100) {
        this.embeddingSize = embeddingSize;
        wordEmbeddings = new Dictionary<string, float[]> ();
    }

    // Train the model with input data: list of sentences and their labels
    public void Train (List<(string[] words, string label)> data) {
        BuildVocabulary (data);
        BuildHuffmanTree (data.Select (d => d.label).ToList ());
        foreach (var (words, label) in data) {
            var inputVector = GetSentenceVector (words);
            var path = GetPathToLabel (label);
            foreach (var node in path) {
                var score = Sigmoid (DotProduct (inputVector, node.Vector));
                var gradient = (node.Code - score);
                // Update node vector
                for (int i = 0; i < embeddingSize; i++) {
                    node.Vector[i] += (float)(0.1 * gradient * inputVector[i]); // learning rate = 0.1
                }

                // Update word embeddings
                foreach (var word in words) {
                    var emb = wordEmbeddings[word];
                    for (int i = 0; i < embeddingSize; i++) {
                        emb[i] += (float)(0.1 * gradient * node.Vector[i]);
                    }
                }
            }
        }
    }

    // Predict the label for a new sentence
    public string Predict (string[] words) {
        var inputVector = GetSentenceVector (words);
        var node = root;
        while (node.Left != null && node.Right != null) {
            var score = Sigmoid (DotProduct (inputVector, node.Vector));
            node = score > 0.5 ? node.Right : node.Left;
        }

        return node.Label;
    }

    // Helper methods below...

    private void BuildVocabulary (List<(string[] words, string label)> data) {
        foreach (var (words, _) in data) {
            foreach (var word in words) {
                if (!wordEmbeddings.ContainsKey (word)) {
                    wordEmbeddings[word] = InitializeVector ();
                }
            }
        }
    }

    private void BuildHuffmanTree (List<string> labels) {
        var freq = labels.GroupBy (l => l).ToDictionary (g => g.Key, g => g.Count ());
        var nodes = new List<Node> ();
        foreach (var label in freq.Keys) {
            nodes.Add (new Node {
                Label = label,
                Frequency = freq[label]
            });
        }

        while (nodes.Count > 1) {
            nodes = nodes.OrderBy (n => n.Frequency).ToList ();
            var left = nodes[0];
            var right = nodes[1];
            var parent = new Node {
                Left = left,
                Right = right,
                Frequency = left.Frequency + right.Frequency,
                Vector = InitializeVector ()
            };
            left.Parent = parent;
            left.Code = 0;
            right.Parent = parent;
            right.Code = 1;
            nodes.RemoveRange (0, 2);
            nodes.Add (parent);
        }

        root = nodes[0];
    }

    private List<Node> GetPathToLabel (string label) {
        var path = new List<Node> ();
        var node = FindNode (root, label);
        while (node.Parent != null) {
            path.Add (node.Parent);
            node = node.Parent;
        }

        path.Reverse ();
        return path;
    }

    private Node FindNode (Node node, string label) {
        if (node == null) return null;
        if (node.Label == label) return node;
        var left = FindNode (node.Left, label);
        if (left != null) return left;
        return FindNode (node.Right, label);
    }

    private float[] GetSentenceVector (string[] words) {
        var vector = new float[embeddingSize];
        foreach (var word in words) {
            if (wordEmbeddings.ContainsKey (word)) {
                var emb = wordEmbeddings[word];
                for (int i = 0; i < embeddingSize; i++) {
                    vector[i] += emb[i];
                }
            }
        }

        return vector;
    }

    private float[] InitializeVector () {
        var vector = new float[embeddingSize];
        var rand = new Random ();
        for (int i = 0; i < embeddingSize; i++) {
            vector[i] = (float)(rand.NextDouble () - 0.5) / embeddingSize;
        }

        return vector;
    }

    private double DotProduct (float[] vec1, float[] vec2) {
        double sum = 0.0;
        for (int i = 0; i < vec1.Length; i++) {
            sum += vec1[i] * vec2[i];
        }

        return sum;
    }

    private double Sigmoid (double x) {
        return 1.0 / (1.0 + Math.Exp (-x));
    }

    // Node class for Huffman tree
    private class Node
    {
        public string Label;
        public int Frequency;
        public Node Left;
        public Node Right;
        public Node Parent;
        public int Code;
        public float[] Vector;
    }

    void test () {
        // Prepare training data
        var data = new List<(string[], string)>
        {
            (new[] { "this", "is", "a", "cat" }, "animal"),
            (new[] { "this", "is", "a", "dog" }, "animal"),
            (new[] { "this", "is", "a", "car" }, "vehicle"),
            (new[] { "this", "is", "a", "bus" }, "vehicle"),
        };

        // Initialize and train the model
        var classifier = new FastTextHierarchicalClassifier();
        classifier.Train(data);

        // Predict a label
        var label = classifier.Predict(new[] { "a", "fast", "car" });
        Console.WriteLine($"Predicted label: {label}");
    }
}
