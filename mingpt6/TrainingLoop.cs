namespace mingpt6;

public class TrainingLoop
{
    private Transformer model;
    private int vocabSize;

    public TrainingLoop (Transformer model, int vocabSize) {
        this.model = model;
        this.vocabSize = vocabSize;
    }

    public void Train (Func<(int[], int)> data, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            int totalTokens = 0;

            var (inputIds, targetId) = data ();

            double[] probabilities = model.Forward (inputIds);

            // Compute loss (negative log-likelihood)
            double loss = -Math.Log (probabilities[targetId] + 1e-10);
            totalLoss += loss;
            totalTokens++;

            // Compute gradient w.r.t logits
            double[] gradOutput = new double[vocabSize];
            gradOutput[targetId] = -1 / (probabilities[targetId] + 1e-10);

            // Backward pass
            model.Backward (inputIds, gradOutput);

            // Update parameters
            // For simplicity, parameter updates are assumed to be handled in model.Backward

            double perplexity = Math.Exp (totalLoss / totalTokens);
            Console.WriteLine ($"Epoch {epoch + 1}: Perplexity = {perplexity}");
        }
    }
}
