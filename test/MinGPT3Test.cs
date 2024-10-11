using transformers.utils;

namespace mingpt3;

public class MinGPT3Test
{
    public static void run () {
        int embeddingSize = 96;
        int numHeads = 6;
        int numLayers = 4;
        int maxSeqLen = 8;
        int batchSize = 8;

        var data = DataLoader.LoadData (maxSeqLen, out var vocabSize, out var vocabulary);

        var model = new Model (vocabSize, embeddingSize, numHeads, numLayers, maxSeqLen);
        var optimizer = new Optimizer (learningRate: 0.0005);

        for (int epoch = 0; epoch < 1000; epoch++) {
            var batchInputIds = Enumerable
                .Range (0, batchSize)
                .Select (_ => data ().Item1)
                .ToArray ();

            var totalLoss = 0.0;

            foreach (var input in batchInputIds) {
                var logitsBatch = model.Forward (input);

                totalLoss += ComputeLoss (logitsBatch, input, out var dLogitsBatch) / batchSize;

                model.Backward (dLogitsBatch, input);
            }

            optimizer.Step (model);

            var a = predict (model, vocabulary, "The ", maxSeqLen, topK: 10);
            var b = predict (model, vocabulary, "The ", maxSeqLen, topK: 0);
            var c = predict (model, vocabulary, "The ", maxSeqLen, topK: 0, argmax: true);
            Console.WriteLine ($"Epoch {epoch}, Loss: {totalLoss:F4}, {a}, {b}, {c}");
        }
    }

    static double ComputeLoss (Matrix logits, int[] targetIds, out Matrix dLogitsBatch) {
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

            loss -= logits.Data[i, targetIds[i]] - logSumExp;

            // Compute gradient
            for (int j = 0; j < V; j++) {
                double softmax = Math.Exp (logits.Data[i, j] - logSumExp) / sumExp;
                dLogits.Data[i, j] = softmax;
            }

            dLogits.Data[i, targetIds[i]] -= 1.0;
        }

        {
            // Normalize gradient
            for (int i = 0; i < N; i++)
            for (int j = 0; j < V; j++)
                dLogits.Data[i, j] /= N;
        }

        dLogitsBatch = dLogits;

        return loss / N;
    }

    static string predict (Model model, char[] vocabulary, string prompt, int generatedLength, double temperature = 1.0, int topK = 10,
        bool argmax = false) {
        int[] seedInput = DataLoader.encode (prompt, vocabulary);

        var generatedTokens = new List<int> (seedInput);

        for (int i = 0; i < generatedLength - prompt.Length; i++) {
            int nextTokenId = model.PredictNextToken (
                generatedTokens.ToArray (),
                temperature,
                topK,
                argmax);
            generatedTokens.Add (nextTokenId);
        }

        return DataLoader.decode (generatedTokens.ToArray (), vocabulary);
    }
}
