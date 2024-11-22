class FastText
{
    Dictionary<string, float[]> wordVectors = new Dictionary<string, float[]> ();

    Dictionary<string, float[]> labelVectors = new Dictionary<string, float[]> ();

    int vectorSize = 100;

    Random rand = new Random ();

    static void run () {
        var trainingData = new List<(string Text, string Label)> {
            ("hello world", "greeting"),
            ("goodbye world", "farewell"),
            ("hi there", "greeting"),
            ("see you later", "farewell")
        };

        var ft = new FastText ();
        ft.Train (trainingData);

        var prediction = ft.Predict ("hello there");
        Console.WriteLine ($"Prediction: {prediction}");
    }

    public void Train (List<(string Text, string Label)> data, int epochs = 5) {
        foreach (var (text, label) in data) {
            var words = GetNGrams (text);
            foreach (var word in words) {
                if (!wordVectors.ContainsKey (word))
                    wordVectors[word] = RandomVector ();
            }

            if (!labelVectors.ContainsKey (label))
                labelVectors[label] = RandomVector ();
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            foreach (var (text, label) in data) {
                var words = GetNGrams (text);
                var inputVec = AverageVectors (words.Select (w => wordVectors[w]));
                var outputVec = labelVectors[label];

                // Gradient descent step (simplified)
                for (int i = 0; i < vectorSize; i++) {
                    float error = inputVec[i] - outputVec[i];
                    foreach (var word in words)
                        wordVectors[word][i] -= 0.01f * error;
                    labelVectors[label][i] += 0.01f * error;
                }
            }
        }
    }

    public string Predict (string text) {
        var words = GetNGrams (text);
        var inputVec = AverageVectors (words.Where (w => wordVectors.ContainsKey (w)).Select (w => wordVectors[w]));

        var bestLabel = labelVectors.Keys
            .OrderByDescending (label => CosineSimilarity (inputVec, labelVectors[label]))
            .FirstOrDefault ();

        return bestLabel;
    }

    List<string> GetNGrams (string text, int minN = 3, int maxN = 6) {
        var ngrams = new List<string> ();
        string cleanText = $"<${text.Replace (" ", "")}$>";
        for (int n = minN; n <= maxN; n++) {
            for (int i = 0; i <= cleanText.Length - n; i++)
                ngrams.Add (cleanText.Substring (i, n));
        }

        return ngrams;
    }

    float[] RandomVector () {
        return Enumerable.Range (0, vectorSize).Select (_ => (float)(rand.NextDouble () - 0.5)).ToArray ();
    }

    float[] AverageVectors (IEnumerable<float[]> vectors) {
        var avg = new float[vectorSize];
        int count = 0;
        foreach (var vec in vectors) {
            for (int i = 0; i < vectorSize; i++)
                avg[i] += vec[i];
            count++;
        }

        if (count > 0)
            for (int i = 0; i < vectorSize; i++)
                avg[i] /= count;
        return avg;
    }

    float CosineSimilarity (float[] vecA, float[] vecB) {
        float dot = 0f, magA = 0f, magB = 0f;
        for (int i = 0; i < vectorSize; i++) {
            dot += vecA[i] * vecB[i];
            magA += vecA[i] * vecA[i];
            magB += vecB[i] * vecB[i];
        }

        return dot / ((float)Math.Sqrt (magA) * (float)Math.Sqrt (magB) + 1e-10f);
    }
}
