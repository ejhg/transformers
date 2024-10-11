namespace mingpt5;

public class EmbeddingLayer
{
    public int VocabSize;
    public int EmbeddingDim;
    public Matrix EmbeddingMatrix;
    public Dictionary<int, Vector> Gradients;

    public EmbeddingLayer (int vocabSize, int embeddingDim) {
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        EmbeddingMatrix = new Matrix (VocabSize, EmbeddingDim);
        Gradients = new Dictionary<int, Vector> ();
        InitializeWeights ();
    }

    private void InitializeWeights () {
        Random rand = new Random ();
        for (int i = 0; i < VocabSize; i++)
        for (int j = 0; j < EmbeddingDim; j++)
            EmbeddingMatrix.Data[i][j] = (rand.NextDouble () - 0.5) / EmbeddingDim;
    }

    public Vector GetEmbedding (int tokenIndex) {
        Vector embedding = new Vector (EmbeddingDim);
        for (int i = 0; i < EmbeddingDim; i++) {
            embedding.Data[i] = EmbeddingMatrix.Data[tokenIndex][i];
        }

        return embedding;
    }

    public void Backward (int tokenIndex, Vector grad) {
        if (!Gradients.ContainsKey (tokenIndex))
            Gradients[tokenIndex] = new Vector (EmbeddingDim);

        for (int i = 0; i < EmbeddingDim; i++) {
            Gradients[tokenIndex].Data[i] += grad.Data[i];
        }
    }

    public void UpdateParameters (double learningRate) {
        foreach (var kvp in Gradients) {
            int tokenIndex = kvp.Key;
            Vector grad = kvp.Value;
            for (int i = 0; i < EmbeddingDim; i++) {
                EmbeddingMatrix.Data[tokenIndex][i] -= learningRate * grad.Data[i];
            }
        }

        Gradients.Clear ();
    }
}
