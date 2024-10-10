class Word2Vec2
{
    public static void run () {
        var text = File.ReadAllText ("resources/oxford-english-dictionary.txt");//.Substring (0, 1000);

        int windowSize = 2, embeddingSize = 10, epochs = 100;
        var learningRate = 0.01;

        var tokens = text.ToLower ().Split (' ');
        var vocab = tokens.Distinct ().ToArray ();
        var vocabSize = vocab.Length;
        var word2Index = vocab.Select ((w, i) => new {
            w,
            i
        }).ToDictionary (x => x.w, x => x.i);

        Console.WriteLine ("text length: " + text.Length);
        Console.WriteLine ("tokens: " + tokens.Length);
        Console.WriteLine ("vocabulary: " + vocab.Length);

        var trainingData = new List<(int, int)> ();
        for (var i = 0; i < tokens.Length; i++) {
            var target = word2Index[tokens[i]];
            for (var j = Math.Max (0, i - windowSize); j <= Math.Min (tokens.Length - 1, i + windowSize); j++) {
                if (i != j)
                    trainingData.Add ((target, word2Index[tokens[j]]));
            }
        }

        var rnd = new Random ();

        var W1 = new double[vocabSize, embeddingSize];
        var W2 = new double[embeddingSize, vocabSize];

        for (var i = 0; i < vocabSize; i++) {
            for (var j = 0; j < embeddingSize; j++) {
                W1[i, j] = rnd.NextDouble () - 0.5;
            }
        }

        for (var i = 0; i < embeddingSize; i++) {
            for (var j = 0; j < vocabSize; j++) {
                W2[i, j] = rnd.NextDouble () - 0.5;
            }
        }

        var h = new double[embeddingSize];
        var e = new double[vocabSize];
        var u = new double[vocabSize];
        var softmaxBuffer = new double[vocabSize];

        for (var epoch = 0; epoch < epochs; epoch++) {
            Console.Write ($"epoch {epoch}");

            var curr = 0;

            foreach (var (targetIndex, contextIndex) in trainingData) {
                for (var i = 0; i < embeddingSize; i++) {
                    h[i] = W1[targetIndex, i];
                }

                for (var i = 0; i < vocabSize; i++) {
                    u[i] = 0;
                    for (var j = 0; j < embeddingSize; j++) {
                        u[i] += h[j] * W2[j, i];
                    }
                }

                var y_pred = softmax (u, ref softmaxBuffer);

                for (var i = 0; i < vocabSize; i++) {
                    e[i] = y_pred[i] - (i == contextIndex ? 1 : 0);
                }

                for (var i = 0; i < embeddingSize; i++) {
                    for (var j = 0; j < vocabSize; j++) {
                        W2[i, j] -= learningRate * e[j] * h[i];
                    }
                }

                for (var i = 0; i < embeddingSize; i++) {
                    double delta = 0;
                    for (var j = 0; j < vocabSize; j++) {
                        delta += e[j] * W2[i, j];
                    }

                    W1[targetIndex, i] -= learningRate * delta;
                }

                curr++;

                if (curr % 10 == 0) {
                    // Console.WriteLine (curr);
                }
            }
        }

        for (var i = 0; i < vocabSize; i++) {
            Console.WriteLine ($"{vocab[i]}: [{string.Join (", ", Enumerable.Range (0, embeddingSize).Select (j => W1[i, j].ToString ("F4")))}]");
        }
    }

    static double[] softmax (double[] x, ref double[] buffer) {
        var max = double.MinValue;

        foreach (var v in x) {
            if (v > max) {
                max = v;
            }
        }

        var sum = 0.0;

        foreach (var v in x) {
            sum += Math.Exp (v - max);
        }

        for (var i = 0; i < buffer.Length; i++) {
            buffer[i] = Math.Exp (x[i] - max) / sum;
        }

        return buffer;
    }
}
