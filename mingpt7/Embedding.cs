namespace mingpt7;

public class Embedding
{
    public int vocabSize;
    public int embeddingDim;
    public double[,] weights;
    public double[,] gradWeights;

    public Embedding (int vocabSize, int embeddingDim) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        weights = new double[vocabSize, embeddingDim];
        gradWeights = new double[vocabSize, embeddingDim];
        Random rand = new Random ();
        for (int i = 0; i < vocabSize; i++)
        for (int j = 0; j < embeddingDim; j++)
            weights[i, j] = rand.NextDouble () * 0.01;
    }

    public double[][] Forward (int[] tokenIndices) {
        int T = tokenIndices.Length;
        double[][] embeddings = new double[T][];
        for (int t = 0; t < T; t++) {
            embeddings[t] = new double[embeddingDim];
            int tokenIndex = tokenIndices[t];
            for (int i = 0; i < embeddingDim; i++)
                embeddings[t][i] = weights[tokenIndex, i];
        }

        return embeddings;
    }

    public void Backward (int[] tokenIndices, double[][] grad) {
        for (int t = 0; t < tokenIndices.Length; t++) {
            int tokenIndex = tokenIndices[t];
            for (int i = 0; i < embeddingDim; i++)
                gradWeights[tokenIndex, i] += grad[t][i];
        }
    }

    public void UpdateParameters (double learningRate) {
        for (int i = 0; i < vocabSize; i++)
        for (int j = 0; j < embeddingDim; j++) {
            weights[i, j] -= learningRate * gradWeights[i, j];
            gradWeights[i, j] = 0.0;
        }
    }
}
