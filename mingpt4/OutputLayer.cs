namespace mingpt4;

class OutputLayer
{
    public int EmbeddingDim { get; set; }
    public int VocabSize { get; set; }
    public Matrix W_out { get; set; }
    public Matrix dW_out { get; set; }

    public OutputLayer (int embeddingDim, int vocabSize) {
        EmbeddingDim = embeddingDim;
        VocabSize = vocabSize;
        W_out = new Matrix (EmbeddingDim, VocabSize);
        dW_out = new Matrix (EmbeddingDim, VocabSize);

        Random rand = new Random ();
        for (int i = 0; i < EmbeddingDim; i++)
        for (int j = 0; j < VocabSize; j++)
            W_out.Data[i][j] = rand.NextDouble () * 0.01;
    }

    public Matrix Forward (Matrix X) {
        Matrix logits = Matrix.Multiply (X, W_out);
        return logits;
    }

    // Backward pass (omitted for brevity)

    public void UpdateParameters (double learningRate) {
        for (int i = 0; i < W_out.Rows; i++)
        for (int j = 0; j < W_out.Cols; j++) {
            W_out.Data[i][j] -= learningRate * dW_out.Data[i][j];
            dW_out.Data[i][j] = 0;
        }
    }
}
