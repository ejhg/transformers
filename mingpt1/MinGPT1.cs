using mingpt3;

namespace mingpt1;

/// <summary>
/// Uses matrices, no batching.
/// </summary>
public class MinGPT1
{
    public EmbeddingLayer TokenEmbedding;
    public EmbeddingLayer PositionalEmbedding;
    public TransformerBlock[] Layers;
    public LinearLayer FinalLayer;

    public MinGPT1 (int vocabSize, int embeddingSize, int numHeads, int numLayers, int maxSeqLen) {
        TokenEmbedding = new EmbeddingLayer (vocabSize, embeddingSize);
        PositionalEmbedding = new EmbeddingLayer (maxSeqLen, embeddingSize);
        Layers = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++)
            Layers[i] = new TransformerBlock (embeddingSize, numHeads);
        FinalLayer = new LinearLayer (embeddingSize, vocabSize);
    }

    public Matrix Forward (int[] inputIds) {
        // Token and positional embeddings
        var x = TokenEmbedding.Forward (inputIds);
        var positions = new int[inputIds.Length];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;
        var posEmb = PositionalEmbedding.Forward (positions);
        x = x + posEmb;

        // Transformer layers
        foreach (var layer in Layers)
            x = layer.Forward (x);

        // Final linear layer
        var logits = FinalLayer.Forward (x);
        return logits;
    }

    public void Backward (Matrix dLogits) {
        // Backward through final linear layer
        var dX = FinalLayer.Backward (dLogits);

        // Backward through transformer layers
        for (int i = Layers.Length - 1; i >= 0; i--)
            dX = Layers[i].Backward (dX);

        // Backward through embeddings
        var dPosEmb = dX;
        TokenEmbedding.Backward (dX);
        PositionalEmbedding.Backward (dPosEmb);
    }
}
