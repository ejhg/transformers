namespace mingpt5;

public class TransformerModel
{
    public int VocabSize;
    public int EmbeddingDim;
    public int NumHeads;
    public int HiddenDim;
    public int NumLayers;
    public EmbeddingLayer Embedding;
    public PositionalEncoding PosEncoding;
    public TransformerBlock[] Layers;
    public Matrix ClassificationLayer;
    public Vector[] Embeddings;

    public TransformerModel (int vocabSize, int embeddingDim, int numHeads, int hiddenDim, int numLayers) {
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        NumHeads = numHeads;
        HiddenDim = hiddenDim;
        NumLayers = numLayers;

        Embedding = new EmbeddingLayer (VocabSize, EmbeddingDim);
        PosEncoding = new PositionalEncoding (EmbeddingDim);
        Layers = new TransformerBlock[NumLayers];
        for (int i = 0; i < NumLayers; i++) {
            Layers[i] = new TransformerBlock (EmbeddingDim, NumHeads, HiddenDim);
        }

        ClassificationLayer = new Matrix (VocabSize, EmbeddingDim);
        Random rand = new Random ();
        for (int i1 = 0; i1 < ClassificationLayer.Rows; i1++)
        for (int j = 0; j < ClassificationLayer.Cols; j++)
            ClassificationLayer.Data[i1][j] = (rand.NextDouble () - 0.5) / ClassificationLayer.Cols;
    }

    private void InitializeMatrix (Matrix m) {
        Random rand = new Random ();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = (rand.NextDouble () - 0.5) / m.Cols;
    }

    public Vector Forward (int[] inputTokens) {
        int seqLength = inputTokens.Length;
        Embeddings = new Vector[seqLength];
        for (int i = 0; i < seqLength; i++) {
            Vector embedding = Embedding.GetEmbedding (inputTokens[i]);
            embedding = PosEncoding.ApplyRoPE (embedding, i);
            Embeddings[i] = embedding;
        }

        for (int l = 0; l < NumLayers; l++) {
            for (int i = 0; i < seqLength; i++) {
                Embeddings[i] = Layers[l].Forward (Embeddings, i);
            }
        }

        // Use the last token's embedding for classification
        Vector logits = ClassificationLayer.Multiply (Embeddings[seqLength - 1]);

        return logits;
    }

    public Vector Backward (Vector dLogits, int[] inputTokens) {
        // Backprop through classification layer
        Vector dLastEmbedding = ClassificationLayer.Transpose ().Multiply (dLogits);

        // Gradient for classification layer
        Matrix dClassificationLayer = Matrix.OuterProduct (dLogits, Embeddings[inputTokens.Length - 1]);

        // Update classification layer
        for (int r = 0; r < ClassificationLayer.Rows; r++)
        for (int c = 0; c < ClassificationLayer.Cols; c++) {
            ClassificationLayer.Data[r][c] -= dClassificationLayer.Data[r][c];
        }

        // Backprop through transformer layers
        Vector[] dEmbeddings = new Vector[inputTokens.Length];
        for (int i = 0; i < inputTokens.Length; i++) {
            dEmbeddings[i] = new Vector (EmbeddingDim);
        }

        dEmbeddings[inputTokens.Length - 1] = dLastEmbedding;

        for (int l = NumLayers - 1; l >= 0; l--) {
            for (int i = inputTokens.Length - 1; i >= 0; i--) {
                dEmbeddings[i] = Layers[l].Backward (dEmbeddings[i]);
            }
        }

        // Backprop through embedding layer
        for (int i = 0; i < inputTokens.Length; i++) {
            Embedding.Backward (inputTokens[i], dEmbeddings[i]);
        }

        return null; // No need to return anything
    }

    public Vector ComputeLossGradient (Vector probs, int targetIndex) {
        Vector grad = probs.Clone ();
        grad.Data[targetIndex] -= 1.0;
        return grad;
    }
}
