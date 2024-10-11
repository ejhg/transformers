namespace mingpt5;

public class EmbeddingLayer
{
    public int vocabSize;
    public int embeddingDim;
    public Matrix weights;
    public Dictionary<int, Vector> gradWeights;

    public EmbeddingLayer (int vocabSize, int embeddingDim) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        weights = new Matrix (this.vocabSize, this.embeddingDim);
        gradWeights = new Dictionary<int, Vector> ();
        Random rand = new Random ();
        for (int i = 0; i < this.vocabSize; i++)
        for (int j = 0; j < this.embeddingDim; j++)
            weights.Data[i][j] = (rand.NextDouble () - 0.5) / this.embeddingDim;
    }

    public Vector GetEmbedding (int tokenIndex) {
        Vector embedding = new Vector (embeddingDim);
        for (int i = 0; i < embeddingDim; i++) {
            embedding.Data[i] = weights.Data[tokenIndex][i];
        }

        return embedding;
    }

    public void Backward (int tokenIndex, Vector grad) {
        if (!gradWeights.ContainsKey (tokenIndex))
            gradWeights[tokenIndex] = new Vector (embeddingDim);

        for (int i = 0; i < embeddingDim; i++) {
            gradWeights[tokenIndex].Data[i] += grad.Data[i];
        }
    }

    public void UpdateParameters (double learningRate) {
        foreach (var kvp in gradWeights) {
            int tokenIndex = kvp.Key;
            Vector grad = kvp.Value;
            for (int i = 0; i < embeddingDim; i++) {
                weights.Data[tokenIndex][i] -= learningRate * grad.Data[i];
            }
        }

        gradWeights.Clear ();
    }
}
