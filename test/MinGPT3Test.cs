namespace mingpt3;

public class MinGPT3Test
{
    public static void run () {
        int embeddingSize = 96;
        int numHeads = 6;
        int numLayers = 4;
        int maxSeqLen = 8;
        int batchSize = 1;

        var data = LoadData (maxSeqLen, out var vocabSize, out var vocabulary);

        Console.WriteLine("vocabulary size: " + vocabSize);

        var model = new GPTModel (vocabSize, embeddingSize, numHeads, numLayers, maxSeqLen);
        var optimizer = new Optimizer (learningRate: 0.0005);

        for (int epoch = 0; epoch < 1000; epoch++) {

            var batchInputIds = Enumerable
                .Range (0, batchSize)
                .Select (_ => data ())
                .ToArray ();

            var logitsBatch = model.Forward (batchInputIds);

            double loss = ComputeLoss (logitsBatch, batchInputIds, out Matrix[] dLogitsBatch);

            model.Backward (dLogitsBatch, batchInputIds);
            optimizer.Step (model);

            Console.WriteLine ($"Epoch {epoch}, Loss: {loss:F4}");

            var tokens =  Enumerable
                .Range (0, 5)
                .Select (i => model.PredictNextToken (encode ("The", vocabulary), temperature: 0.7, topK: 40))
                .ToArray ();
            Console.Write (decode (tokens, vocabulary));
        }

        // After training, generate text
        int[] seedInput = GetSeedInput (); // Your initial sequence of tokens
        int generatedLength = 20; // Number of tokens to generate

        var generatedTokens = new List<int> (seedInput);

        for (int i = 0; i < generatedLength; i++) {
            int[] currentInput = generatedTokens.ToArray ();
            int nextTokenId = model.PredictNextToken (currentInput);
            generatedTokens.Add (nextTokenId);
        }

        // Convert token IDs to actual tokens (depends on your tokenizer)
        string generatedText = DecodeTokens (generatedTokens.ToArray ());
        Console.WriteLine ("Generated Text:");
        Console.WriteLine (generatedText);
    }

    static int[] GetSeedInput () {
        // Provide your own seed input sequence
        // For demonstration, we'll generate a random sequence
        var rand = new Random ();
        int seqLen = 5;
        var inputIds = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
            inputIds[i] = rand.Next (0, 1000);
        return inputIds;
    }

    static string DecodeTokens (int[] tokenIds) {
        // Convert token IDs back to text
        // This depends on your tokenizer/vocabulary mapping
        // For demonstration, we'll just return token IDs as strings
        return string.Join (" ", tokenIds);
    }

    static int[] encode (string source, char[] vocabulary) {
        var charToIndex = vocabulary
            .Select ((c, index) => (c, index))
            .ToDictionary ();
        return source.Select (c => charToIndex[c]).ToArray ();
    }

    static string decode (int[] tokens, char[] vocabulary) {
        return string.Join ("", tokens.Select (c => c >= vocabulary.Length ? '?' : vocabulary[c]).ToArray ());
    }

    static Func<int[]> LoadData (int sequenceLength, out int vocabularySize, out char[] vocabulary) {
        var text = File.ReadAllText ("resources/tinyshakespeare.txt");
        var sourceCharacters = text.ToArray ();

        vocabulary = sourceCharacters
            .Distinct ()
            .Order()
            .ToArray ();
        vocabularySize = vocabulary.Length;

        Console.WriteLine($"vocabulary: {string.Join ("",vocabulary)}");

        var charToIndex = vocabulary
            .Select ((c, index) => (c, index))
            .ToDictionary ();

        var data = sourceCharacters.Select (c => charToIndex[c]).ToArray ();

        var rnd = new Random ();

        return () => data
            .Skip (rnd.Next(0, data.Length - sequenceLength))
            .Take (sequenceLength)
            .ToArray ();
    }

    static double ComputeLoss (Matrix[] logitsBatch, int[][] batchTargetIds, out Matrix[] dLogitsBatch) {
        int batchSize = logitsBatch.Length;
        double totalLoss = 0.0;
        dLogitsBatch = new Matrix[batchSize];

        for (int b = 0; b < batchSize; b++) {
            Matrix logits = logitsBatch[b];
            int[] targetIds = batchTargetIds[b];
            int N = targetIds.Length;
            int V = logits.Cols;
            var dLogits = new Matrix (N, V);
            double loss = 0.0;

            for (int i = 0; i < N; i++) {
                double maxLogit = double.NegativeInfinity;
                for (int j = 0; j < V; j++)
                    if (logits.Data[i, j] > maxLogit)
                        maxLogit = logits.Data[i, j];

                double sumExp = 0.0;
                for (int j = 0; j < V; j++)
                    sumExp += Math.Exp (logits.Data[i, j] - maxLogit);
                double logSumExp = maxLogit + Math.Log (sumExp);

                double logProb = logits.Data[i, targetIds[i]] - logSumExp;
                loss -= logProb;

                // Compute gradient
                for (int j = 0; j < V; j++) {
                    double softmax = Math.Exp (logits.Data[i, j] - logSumExp) / sumExp;
                    dLogits.Data[i, j] = softmax;
                }

                dLogits.Data[i, targetIds[i]] -= 1.0;
            }

            loss /= N;
            totalLoss += loss;

            // Normalize gradient
            for (int i = 0; i < N; i++)
            for (int j = 0; j < V; j++)
                dLogits.Data[i, j] /= N;

            dLogitsBatch[b] = dLogits;
        }

        totalLoss /= batchSize;
        return totalLoss;
    }
}
