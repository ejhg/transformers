using mingpt3;
using transformers.utils;

namespace mingpt7;

public class Transformer
{
    public int vocabSize;
    public int embeddingDim;
    public int numHeads;
    public int numLayers;
    public int hiddenDim;
    public EmbeddingLayer embedding;
    public TransformerBlock[] layers;
    public double[,] linearWeight;
    public double[] linearBias;
    public double[,] dLinearWeight;
    public double[] dLinearBias;

    public Transformer (int vocabSize, int embeddingDim, int numHeads, int numLayers, int hiddenDim) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.hiddenDim = hiddenDim;

        embedding = new EmbeddingLayer (vocabSize, embeddingDim);
        layers = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++)
            layers[i] = new TransformerBlock (embeddingDim, numHeads, hiddenDim);

        linearWeight = new double[vocabSize, embeddingDim];
        linearBias = new double[vocabSize];
        dLinearWeight = new double[vocabSize, embeddingDim];
        dLinearBias = new double[vocabSize];

        Random rand = new Random ();
        InitializeMatrix (linearWeight, rand);
        InitializeVector (linearBias, rand);
    }

    void InitializeMatrix (double[,] matrix, Random rand) {
        int rows = matrix.GetLength (0);
        int cols = matrix.GetLength (1);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble () * 0.01;
    }

    void InitializeVector (double[] vector, Random rand) {
        for (int i = 0; i < vector.Length; i++)
            vector[i] = 0.0;
    }

    public double[][] Forward (int[] tokenIndices) {
        // TODO double[][] x = embedding.Forward (tokenIndices);
        embedding.Forward (tokenIndices);
        double[][] x = null;

        for (int i = 0; i < numLayers; i++)
            x = layers[i].Forward (x);
        return x;
    }

    public double[] Predict (double[] x_t) {
        double[] logits = math.MatVecMul (linearWeight, x_t);
        logits = math.Add (logits, linearBias);
        double[] probs = math.Softmax (logits);
        return probs;
    }

    public void Backward (int[] tokenIndices, double[][] outputs) {
        int T = tokenIndices.Length;
        double[][] grad = new double[T][];
        for (int t = T - 1; t >= 0; t--) {
            grad[t] = new double[embeddingDim];
            if (t < T - 1) {
                double[] probs = Predict (outputs[t]);
                int target = tokenIndices[t + 1];
                probs[target] -= 1.0;
                for (int i = 0; i < vocabSize; i++) {
                    for (int j = 0; j < embeddingDim; j++)
                        dLinearWeight[i, j] += probs[i] * outputs[t][j];
                    dLinearBias[i] += probs[i];
                    for (int j = 0; j < embeddingDim; j++)
                        grad[t][j] += linearWeight[i, j] * probs[i];
                }
            }
        }

        for (int i = numLayers - 1; i >= 0; i--)
            grad = layers[i].Backward (grad);

        // todo
        // embedding.Backward (tokenIndices, grad);
    }

    public void UpdateParameters (double learningRate) {
        for (int i = 0; i < vocabSize; i++)
        for (int j = 0; j < embeddingDim; j++) {
            linearWeight[i, j] -= learningRate * dLinearWeight[i, j];
            dLinearWeight[i, j] = 0.0;
        }

        for (int i = 0; i < vocabSize; i++) {
            linearBias[i] -= learningRate * dLinearBias[i];
            dLinearBias[i] = 0.0;
        }

        embedding.UpdateParameters (learningRate);
        for (int i = 0; i < numLayers; i++)
            layers[i].UpdateParameters (learningRate);
    }
}
