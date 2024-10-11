namespace mingpt7;

public class Trainer
{
    public Transformer model;
    public double learningRate;

    public Trainer (Transformer model, double learningRate) {
        this.model = model;
        this.learningRate = learningRate;
    }

    public void Train (Func<(int[], int)> data, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {

            var (tokenIndices, targetIndex) = data ();

            double[][] outputs = model.Forward (tokenIndices);

            if (epoch % 10 == 0) {
                double totalLoss = 0.0;
                int T = tokenIndices.Length;
                for (int t = 0; t < T - 1; t++) {
                    double[] probs = model.Predict (outputs[t]);
                    double loss = MathUtils.CrossEntropyLoss (probs, targetIndex);
                    totalLoss += loss;
                }

                double avgLoss = totalLoss / (T - 1);
                double perplexity = Math.Exp (avgLoss);
                Console.WriteLine ($"Epoch {epoch + 1}, Loss: {avgLoss:F4}, Perplexity: {perplexity:F4}");
            }

            model.Backward (tokenIndices, outputs);
            model.UpdateParameters (learningRate);
        }
    }

    public int PredictNextToken (int[] tokenIndices) {
        double[][] outputs = model.Forward (tokenIndices);
        double[] probs = model.Predict (outputs[tokenIndices.Length - 1]);
        int nextToken = SampleFromDistribution (probs);
        return nextToken;
    }

    int SampleFromDistribution (double[] probs) {
        Random rand = new Random ();
        double r = rand.NextDouble ();
        double cumulative = 0.0;
        for (int i = 0; i < probs.Length; i++) {
            cumulative += probs[i];
            if (r < cumulative)
                return i;
        }

        return probs.Length - 1;
    }
}
