using transformers.utils;

namespace mingpt.cs;

static class Trainer
{
    public static void train (Model model, Optimizer optimizer, int batchSize, Func<(int[] tokens, int[] targets)> data, char[] vocabulary) {
        for (int epoch = 0; epoch < 1000; epoch++) {
            var documents = Enumerable
                .Range (0, batchSize)
                .Select (_ => data ())
                .ToArray ();

            var totalLoss = 0.0;

            foreach (var document in documents) {
                var logitsBatch = model.Forward (document.tokens);

                var dLogitsBatch = CrossEntropyLoss.ComputeLoss (logitsBatch, document.targets, out var loss);

                totalLoss += loss;

                model.Backward (dLogitsBatch, document.tokens);
            }

            totalLoss /= batchSize;
            optimizer.Step (model);


            double[] logits (Matrix x) {
                var logits = new double[x.Data.GetLength (1)];
                var seqLength = x.Data.GetLength (0);

                for (var i = 0; i < logits.Length; i++) {
                    logits[i] = x.Data[seqLength - 1, i];
                }

                return logits;
            }

            var a = TextGeneration.predict (_ => logits (model.Forward (_)), vocabulary, "The ", model.MaxSeqLen, topK: 10);
            var b = TextGeneration.predict (_ => logits (model.Forward (_)), vocabulary, "The ", model.MaxSeqLen, topK: 0);
            var c = TextGeneration.predict (_ => logits (model.Forward (_)), vocabulary, "The ", model.MaxSeqLen, topK: 0, argmax: true);
            Console.WriteLine ($"Epoch {epoch}, Loss: {totalLoss:F4}, {a}, {b}, {c}");
        }
    }
}
