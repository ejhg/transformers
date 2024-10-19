namespace llama;

public class Trainer
{
    public static void train (LlamaForCausalLM model, AdamOptimizer optimizer, Func<(int[], int[])> data, int epochs, int epochSize,
        Action callback) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            for (int i = 0; i < epochSize; i++) {
                var (inputTokens, targetTokens) = data ();

                // Forward pass
                List<double[]> logitsList = model.Forward (inputTokens);

                // Compute loss and gradient for each time step
                List<double[]> dLogitsList = new List<double[]> ();
                double loss = 0.0;

                for (int t = 0; t < targetTokens.Length; t++) {
                    double[] dLogits;
                    loss += LossFunctions.CrossEntropyLoss (logitsList[t], targetTokens[t], out dLogits);
                    dLogitsList.Add (dLogits);
                }

                totalLoss += loss / targetTokens.Length;

                // Backward pass
                model.Backward (dLogitsList);

                // Update parameters
                optimizer.Step (model);
            }

            Console.WriteLine ($"Epoch {epoch + 1}/{epochs}, Loss: {totalLoss / epochSize}");

            callback?.Invoke ();
        }
    }
}
