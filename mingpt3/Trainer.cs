using transformers.utils;

namespace mingpt3;

static class Trainer
{
    public static void train (Model model, Optimizer optimizer, int batchSize, Func<(int[], int)> data, char[]vocabulary) {
        for (int epoch = 0; epoch < 1000; epoch++) {
            var batchInputIds = Enumerable
                .Range (0, batchSize)
                .Select (_ => data ().Item1)
                .ToArray ();

            var totalLoss = 0.0;

            foreach (var input in batchInputIds) {
                var logitsBatch = model.Forward (input);

                var dLogitsBatch = CrossEntropyLoss.ComputeLoss (logitsBatch, input, out var loss);

                totalLoss += loss / batchSize;

                model.Backward (dLogitsBatch, input);
            }

            optimizer.Step (model);

            var a = predict (model, vocabulary, "The ", model.MaxSeqLen, topK: 10);
            var b = predict (model, vocabulary, "The ", model.MaxSeqLen, topK: 0);
            var c = predict (model, vocabulary, "The ", model.MaxSeqLen, topK: 0, argmax: true);
            Console.WriteLine ($"Epoch {epoch}, Loss: {totalLoss:F4}, {a}, {b}, {c}");
        }
    }

    static string predict (Model model, char[] vocabulary, string prompt, int generatedLength, double temperature = 1.0, int topK = 10,
        bool argmax = false) {
        var generatedTokens = new List<int> (DataLoader.encode (prompt, vocabulary));

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
