using transformers.utils;

namespace mingpt5;

public class Trainer
{
    public TransformerModel Model;
    public double LearningRate;

    public Trainer (TransformerModel model, double learningRate) {
        Model = model;
        LearningRate = learningRate;
    }

    public void Train (Func<(int[], int)> getSample, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            int totalTokens = 0;

            var (inputSequence, targetToken) = getSample ();

            // Forward pass
            Vector logits = Model.Forward (inputSequence);
            Vector probs = new Vector (math.Softmax (logits.Data));
            double loss = math.CrossEntropyLoss (probs.Data, targetToken);
            totalLoss += loss;
            totalTokens++;

            // Backward pass
            Vector dLoss = Model.ComputeLossGradient (probs, targetToken);
            Model.Backward (dLoss, inputSequence);

            // Update parameters
            Model.Embedding.UpdateParameters (LearningRate);
            // Other parameters are updated within their respective backward methods

            double perplexity = Math.Exp (totalLoss / totalTokens);
            Console.WriteLine ($"Epoch {epoch + 1}, Loss: {totalLoss / totalTokens}, Perplexity: {perplexity}");
        }
    }
}
