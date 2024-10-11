using transformers.utils;

namespace mingpt3;

public class MinGPT3Test
{
    public static void run () {
        int embeddingSize = 96;
        int numHeads = 6;
        int numLayers = 4;
        int maxSeqLen = 8;
        int batchSize = 1;

        var data = DataLoader.LoadData (maxSeqLen, out var vocabSize, out var vocabulary);

        var model = new GPTModel (vocabSize, embeddingSize, numHeads, numLayers, maxSeqLen);
        var optimizer = new Optimizer (learningRate: 0.0005);

        for (int epoch = 0; epoch < 1000; epoch++) {
            var batchInputIds = Enumerable
                .Range (0, batchSize)
                .Select (_ => data ().Item1)
                .ToArray ();

            var logitsBatch = model.Forward (batchInputIds);

            double loss = ComputeLoss (logitsBatch, batchInputIds, out Matrix[] dLogitsBatch);

            model.Backward (dLogitsBatch, batchInputIds);
            optimizer.Step (model);

            Console.WriteLine ($"Epoch {epoch}, Loss: {loss:F4}, {predict (model, vocabulary, "The ", maxSeqLen)}");
        }
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

    static string predict (GPTModel model, char[] vocabulary, string prompt, int generatedLength) {
        int[] seedInput = DataLoader.encode (prompt, vocabulary);

        var generatedTokens = new List<int> (seedInput);

        for (int i = 0; i < generatedLength - prompt.Length; i++) {
            int nextTokenId = model.PredictNextToken (generatedTokens.ToArray ());
            generatedTokens.Add (nextTokenId);
        }

        return DataLoader.decode (generatedTokens.ToArray (), vocabulary);
    }
}
