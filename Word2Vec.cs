class Word2Vec
{
    public static void run () {
        string text = "The quick brown fox jumps over the lazy dog";
        int windowSize = 2, embeddingSize = 10, epochs = 5000;
        double learningRate = 0.01;

        var tokens = text.ToLower ().Split (' ');
        var vocab = tokens.Distinct ().ToList ();
        int vocabSize = vocab.Count;
        var word2Index = vocab.Select ((w, i) => new {
            w,
            i
        }).ToDictionary (x => x.w, x => x.i);

        var trainingData = new List<(int, int)> ();
        for (int i = 0; i < tokens.Length; i++) {
            int target = word2Index[tokens[i]];
            for (int j = Math.Max (0, i - windowSize); j <= Math.Min (tokens.Length - 1, i + windowSize); j++) {
                if (i != j)
                    trainingData.Add ((target, word2Index[tokens[j]]));
            }
        }

        Random rnd = new Random ();
        double[,] W1 = new double[vocabSize, embeddingSize];
        double[,] W2 = new double[embeddingSize, vocabSize];
        for (int i = 0; i < vocabSize; i++)
        for (int j = 0; j < embeddingSize; j++)
            W1[i, j] = rnd.NextDouble () - 0.5;
        for (int i = 0; i < embeddingSize; i++)
        for (int j = 0; j < vocabSize; j++)
            W2[i, j] = rnd.NextDouble () - 0.5;

        for (int epoch = 0; epoch < epochs; epoch++) {
            foreach (var (targetIndex, contextIndex) in trainingData) {
                double[] h = new double[embeddingSize];
                for (int i = 0; i < embeddingSize; i++)
                    h[i] = W1[targetIndex, i];

                double[] u = new double[vocabSize];
                for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < embeddingSize; j++)
                    u[i] += h[j] * W2[j, i];

                double[] y_pred = Softmax (u);

                double[] e = new double[vocabSize];
                for (int i = 0; i < vocabSize; i++)
                    e[i] = y_pred[i] - (i == contextIndex ? 1 : 0);

                for (int i = 0; i < embeddingSize; i++)
                for (int j = 0; j < vocabSize; j++)
                    W2[i, j] -= learningRate * e[j] * h[i];

                for (int i = 0; i < embeddingSize; i++) {
                    double delta = 0;
                    for (int j = 0; j < vocabSize; j++)
                        delta += e[j] * W2[i, j];
                    W1[targetIndex, i] -= learningRate * delta;
                }
            }
        }

        for (int i = 0; i < vocabSize; i++)
            Console.WriteLine ($"{vocab[i]}: [{string.Join (", ", Enumerable.Range (0, embeddingSize).Select (j => W1[i, j].ToString ("F4")))}]");
    }

    static double[] Softmax (double[] x) {
        double max = x.Max ();
        double sum = x.Sum (v => Math.Exp (v - max));
        return x.Select (v => Math.Exp (v - max) / sum).ToArray ();
    }
}
