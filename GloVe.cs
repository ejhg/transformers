class GloVe
{
    static void run () {
        // Read corpus and build vocabulary
        var corpus = File.ReadAllText ("corpus.txt").Split ();
        var vocab = corpus.Distinct ().ToList ();
        var word2id = vocab.Select ((w, i) => new {
            w,
            i
        }).ToDictionary (x => x.w, x => x.i);
        int vocabSize = vocab.Count;

        // Build co-occurrence matrix
        var cooc = new Dictionary<(int, int), double> ();
        int windowSize = 5;
        for (int i = 0; i < corpus.Length; i++) {
            int wi = word2id[corpus[i]];
            int start = Math.Max (i - windowSize, 0);
            int end = Math.Min (i + windowSize + 1, corpus.Length);
            for (int j = start; j < end; j++) {
                if (j == i) continue;
                int wj = word2id[corpus[j]];
                double distance = Math.Abs (j - i);
                double weight = 1.0 / distance;
                var key = (wi, wj);
                cooc[key] = cooc.ContainsKey (key) ? cooc[key] + weight : weight;
            }
        }

        // Initialize word vectors and biases
        int vectorSize = 50;
        var W = new double[vocabSize][];
        var biases = new double[vocabSize];
        var gradW = new double[vocabSize][];
        var gradBiases = new double[vocabSize];
        var rand = new Random ();
        for (int i = 0; i < vocabSize; i++) {
            W[i] = new double[vectorSize];
            gradW[i] = new double[vectorSize];
            for (int j = 0; j < vectorSize; j++)
                W[i][j] = (rand.NextDouble () - 0.5) / vectorSize;
            biases[i] = 0;
            gradBiases[i] = 1.0;
        }

        // Training parameters
        double xmax = 100, alpha = 0.75, learningRate = 0.05;
        int epochs = 50;

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            foreach (var pair in cooc) {
                int i = pair.Key.Item1, j = pair.Key.Item2;
                double Xij = pair.Value;
                double weight = (Xij < xmax) ? Math.Pow (Xij / xmax, alpha) : 1.0;
                double dot = 0;
                for (int k = 0; k < vectorSize; k++)
                    dot += W[i][k] * W[j][k];
                double diff = dot + biases[i] + biases[j] - Math.Log (Xij);
                double fdiff = weight * diff;

                for (int k = 0; k < vectorSize; k++) {
                    double temp = fdiff * W[j][k];
                    W[i][k] -= learningRate * temp / Math.Sqrt (gradW[i][k]);
                    gradW[i][k] += temp * temp;
                    temp = fdiff * W[i][k];
                    W[j][k] -= learningRate * temp / Math.Sqrt (gradW[j][k]);
                    gradW[j][k] += temp * temp;
                }

                biases[i] -= learningRate * fdiff / Math.Sqrt (gradBiases[i]);
                gradBiases[i] += fdiff * fdiff;
                biases[j] -= learningRate * fdiff / Math.Sqrt (gradBiases[j]);
                gradBiases[j] += fdiff * fdiff;
            }

            Console.WriteLine ("Epoch {0} complete", epoch + 1);
        }

        // Output word vectors
        using (var writer = new StreamWriter ("vectors.txt")) {
            for (int i = 0; i < vocabSize; i++) {
                writer.Write (vocab[i]);
                for (int j = 0; j < vectorSize; j++)
                    writer.Write (" " + W[i][j]);
                writer.WriteLine ();
            }
        }
    }
}
