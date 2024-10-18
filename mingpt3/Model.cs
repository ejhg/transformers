namespace mingpt3;

/**
 * https://chatgpt.com/share/67084674-24ec-8009-b7e1-6e45002c5f96
 */
public class Model
{
    public int VocabSize, EmbeddingSize, NumHeads, NumLayers, MaxSeqLen;
    public EmbeddingLayer TokenEmbedding;
    public EmbeddingLayer PositionalEmbedding;
    public TransformerBlock[] Layers;
    public LinearLayer FinalLayer;

    public Model (int vocabSize, int embeddingSize, int numHeads, int numLayers, int maxSeqLen) {
        VocabSize = vocabSize;
        EmbeddingSize = embeddingSize;
        NumHeads = numHeads;
        NumLayers = numLayers;
        MaxSeqLen = maxSeqLen;

        TokenEmbedding = new EmbeddingLayer (vocabSize, embeddingSize);
        PositionalEmbedding = new EmbeddingLayer (maxSeqLen, embeddingSize);
        Layers = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++)
            Layers[i] = new TransformerBlock (embeddingSize, numHeads);
        FinalLayer = new LinearLayer (embeddingSize, vocabSize);
    }

    public Matrix Forward (int[] batchInputIds) {
        var tokenEmb = TokenEmbedding.Forward (batchInputIds);
        var posEmb = PositionalEmbedding.Forward (Enumerable
            .Range (0, batchInputIds.Length)
            .Select (i => i)
            .ToArray ());

        var x = tokenEmb + posEmb;

        foreach (var layer in Layers) {
            x = layer.Forward (x);
        }

        x = FinalLayer.Forward (x);

        return x; // Returns logits
    }

    public void Backward (Matrix dLogits, int[] batchInputIds) {
        // Backward through final linear layer
        var dX = FinalLayer.Backward (dLogits);

        // Backward through transformer layers
        for (int i = Layers.Length - 1; i >= 0; i--) {
            dX = Layers[i].Backward (dX);
        }

        // Backward through embeddings
        TokenEmbedding.Backward (dX, batchInputIds);
        PositionalEmbedding.Backward (dX, GetPositions (batchInputIds.Length));
    }

    private int[] GetPositions (int length) {
        var positions = new int[length];
        for (int i = 0; i < length; i++)
            positions[i] = i;
        return positions;
    }
}
