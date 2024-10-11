using mingpt5;

namespace mingpt4;

class MinGPT4Test
{
    static void Main (string[] args) {
        // Hyperparameters
        int vocabSize = 10000;
        int maxSeqLength = 20;
        int embeddingDim = 128;
        int numHeads = 8;
        int numLayers = 2;
        int hiddenDim = 256;
        double learningRate = 0.001;

        TransformerModel model = new TransformerModel (vocabSize, maxSeqLength, embeddingDim, numHeads, numLayers, hiddenDim);

        // Dummy data for demonstration
        int[] inputTokens = new int[maxSeqLength];
        int[] targetTokens = new int[maxSeqLength];

        Random rand = new Random ();
        for (int i = 0; i < maxSeqLength; i++) {
            inputTokens[i] = rand.Next (vocabSize);
            targetTokens[i] = rand.Next (vocabSize);
        }

        // Training loop
        for (int epoch = 0; epoch < 10; epoch++) {
            // Forward pass
            Matrix logits = model.Forward (inputTokens);

            mingpt3.CrossEntropyLoss.ComputeLoss (new mingpt3.Matrix (logits.Data), targetTokens, out var loss);

            // Update parameters
            model.UpdateParameters (learningRate);

            // Compute perplexity
            double perplexity = Math.Exp (loss);

            Console.WriteLine ($"Epoch {epoch + 1}, Loss: {loss}, Perplexity: {perplexity}");
        }

        // Next token prediction
        int[] testInput = new int[maxSeqLength];
        for (int i = 0; i < maxSeqLength; i++)
            testInput[i] = rand.Next (vocabSize);

        Matrix testLogits = model.Forward (testInput);
        int predictedToken = ArgMax (testLogits.Data[maxSeqLength - 1]);

        Console.WriteLine ($"Predicted next token: {predictedToken}");
    }

    static int ArgMax (double[] array) {
        int index = 0;
        double max = array[0];
        for (int i = 1; i < array.Length; i++)
            if (array[i] > max) {
                max = array[i];
                index = i;
            }

        return index;
    }
}
