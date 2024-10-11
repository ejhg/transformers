using mingpt3;
using transformers.utils;

namespace mingpt1;

public class MinGPT1Test
{
    public static void run () {
        int embeddingSize = 64;
        int numHeads = 4;
        int numLayers = 2;
        int maxSeqLen = 128;

        var data = DataLoader.LoadData (maxSeqLen, out var vocabSize, out var dictionary);

        var model = new MinGPT1 (vocabSize, embeddingSize, numHeads, numLayers, maxSeqLen);
        var optimizer = new Optimizer (learningRate: 0.001);

        for (int epoch = 0; epoch < 10000; epoch++) {
            var (inputIds, _) = data ();

            var logits = model.Forward (inputIds);

            double loss = ComputeLoss (logits, inputIds, out Matrix dLogits);

            model.Backward (dLogits);
            optimizer.Step (model);

            Console.WriteLine ($"Epoch {epoch}, Loss: {loss}");
        }
    }

    static double ComputeLoss (Matrix logits, int[] targetIds, out Matrix dLogits) {
        int N = targetIds.Length;
        int V = logits.Cols;
        dLogits = new Matrix (N, V);
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
                double softmax = Math.Exp (logits.Data[i, j] - logSumExp);
                softmax /= sumExp;
                dLogits.Data[i, j] = softmax;
            }

            dLogits.Data[i, targetIds[i]] -= 1.0;
        }

        loss /= N;

        // Normalize gradient
        for (int i = 0; i < N; i++)
        for (int j = 0; j < V; j++)
            dLogits.Data[i, j] /= N;

        return loss;
    }
}
