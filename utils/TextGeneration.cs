namespace transformers.utils;

static class TextGeneration
{
    public static string predict (
        Func<int[], double[]> Forward,
        char[] vocabulary,
        string prompt,
        int generatedLength,
        double temperature = 1.0,
        int topK = 10,
        bool argmax = false
    ) {
        var generatedTokens = new List<int> (DataLoader.encode (prompt, vocabulary));

        for (int i = 0; i < generatedLength - prompt.Length; i++) {
            var nextTokenId = nextToken (
                Forward,
                generatedTokens.ToArray (),
                temperature,
                topK,
                argmax);

            generatedTokens.Add (nextTokenId);
        }

        return DataLoader.decode (generatedTokens.ToArray (), vocabulary);
    }

    static int nextToken (
        Func<int[], double[]> Forward,
        int[] inputIds,
        double temperature = 1.0,
        int topK = 10,
        bool argmax = false
    ) {
        var logits = Forward (inputIds);
        var logitsOverTemp = new double[logits.Length];

        for (int i = 0; i < logits.Length; i++)
            logitsOverTemp[i] = logits[i] / temperature;

        if (topK > 0) {
            logitsOverTemp = TopKFilter (logitsOverTemp, topK);
        }

        // Apply softmax to convert logits to probabilities
        var probabilities = Softmax (logitsOverTemp);

        // Sample the next token based on probabilities
        return argmax
            ? sampling.ArgMax (probabilities)
            : sampling.SampleFromDistribution (probabilities);
    }

    static double[] TopKFilter (double[] logits, int k) {
        var filteredLogits = new double[logits.Length];
        Array.Copy (logits, filteredLogits, logits.Length);

        var indices = new int[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            indices[i] = i;

        Array.Sort (logits, indices);
        Array.Reverse (logits);
        Array.Reverse (indices);

        for (int i = k; i < logits.Length; i++)
            filteredLogits[indices[i]] = double.NegativeInfinity;

        return filteredLogits;
    }

    static double[] Softmax (double[] logits) {
        double maxLogit = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > maxLogit)
                maxLogit = logits[i];

        double sumExp = 0.0;
        var expLogits = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++) {
            expLogits[i] = Math.Exp (logits[i] - maxLogit);
            sumExp += expLogits[i];
        }

        var probabilities = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            probabilities[i] = expLogits[i] / sumExp;

        return probabilities;
    }
}
