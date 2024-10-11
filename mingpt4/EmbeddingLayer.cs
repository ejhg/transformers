namespace mingpt4;

class EmbeddingLayer
{
    public int VocabSize { get; set; }
    public int EmbeddingDim { get; set; }
    public Matrix EmbeddingMatrix { get; set; }
    public Matrix EmbeddingGradients { get; set; }

    public EmbeddingLayer (int vocabSize, int embeddingDim) {
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        EmbeddingMatrix = new Matrix (vocabSize, embeddingDim);
        EmbeddingGradients = new Matrix (vocabSize, embeddingDim);

        Random rand = new Random ();
        // Initialize embeddings randomly
        for (int i = 0; i < VocabSize; i++)
        for (int j = 0; j < EmbeddingDim; j++)
            EmbeddingMatrix.Data[i][j] = rand.NextDouble () * 0.01;
    }

    // Forward pass
    public Matrix Forward (int[] inputTokens) {
        int seqLength = inputTokens.Length;
        Matrix embeddings = new Matrix (seqLength, EmbeddingDim);
        for (int i = 0; i < seqLength; i++) {
            int tokenId = inputTokens[i];
            for (int j = 0; j < EmbeddingDim; j++)
                embeddings.Data[i][j] = EmbeddingMatrix.Data[tokenId][j];
        }

        return embeddings;
    }

    // Backward pass
    public void Backward (Matrix gradOutput, int[] inputTokens) {
        for (int i = 0; i < inputTokens.Length; i++) {
            int tokenId = inputTokens[i];
            for (int j = 0; j < EmbeddingDim; j++) {
                EmbeddingGradients.Data[tokenId][j] += gradOutput.Data[i][j];
            }
        }
    }

    // Parameter update
    public void UpdateParameters (double learningRate) {
        for (int i = 0; i < VocabSize; i++) {
            for (int j = 0; j < EmbeddingDim; j++) {
                EmbeddingMatrix.Data[i][j] -= learningRate * EmbeddingGradients.Data[i][j];
                EmbeddingGradients.Data[i][j] = 0; // Reset gradients
            }
        }
    }
}
