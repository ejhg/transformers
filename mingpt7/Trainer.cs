using transformers.utils;

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

            model.Backward (tokenIndices, outputs);
            model.UpdateParameters (learningRate);

            if (epoch % 10 == 0) {
                double totalLoss = 0.0;
                int T = tokenIndices.Length;
                for (int t = 0; t < T - 1; t++) {
                    double[] probs = model.Predict (outputs[t]);
                    double loss = math.CrossEntropyLoss (probs, targetIndex);
                    totalLoss += loss;
                }

                double avgLoss = totalLoss / (T - 1);
                double perplexity = Math.Exp (avgLoss);
                Console.WriteLine ($"Epoch {epoch + 1}, Loss: {avgLoss:F4}, Perplexity: {perplexity:F4}");
            }
        }
    }

    public int PredictNextToken (int[] tokenIndices) {
        double[][] outputs = model.Forward (tokenIndices);
        double[] probs = model.Predict (outputs[tokenIndices.Length - 1]);
        int nextToken = sampling.SampleFromDistribution (probs);
        return nextToken;
    }


}
