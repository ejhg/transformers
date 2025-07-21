namespace mingpt.cs;

/**
 * https://chatgpt.com/share/67084674-24ec-8009-b7e1-6e45002c5f96
 */
public class Model
{
    public int VocabSize, EmbeddingSize, NumHeads, NumLayers, MaxSeqLen;
    public EmbeddingLayer TokenEmbedding;
    public PositionalEncoding PositionalEmbedding;
    public TransformerBlock[] Layers;
    public LayerNorm FinalLayerNorm;
    public LinearLayer FinalLayer;
    public bool Training { get; set; } = true;
    public double DropoutRate { get; set; } = 0.1;

    public Model (int vocabSize, int embeddingSize, int numHeads, int numLayers, int maxSeqLen, PositionalEncodingType positionType = PositionalEncodingType.Learnable) {
        VocabSize = vocabSize;
        EmbeddingSize = embeddingSize;
        NumHeads = numHeads;
        NumLayers = numLayers;
        MaxSeqLen = maxSeqLen;

        TokenEmbedding = new EmbeddingLayer (vocabSize, embeddingSize);
        PositionalEmbedding = new PositionalEncoding (maxSeqLen, embeddingSize, positionType);
        Layers = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++)
            Layers[i] = new TransformerBlock (embeddingSize, numHeads);
        FinalLayerNorm = new LayerNorm(embeddingSize);
        FinalLayer = new LinearLayer (embeddingSize, vocabSize, useBias: false); // GPT-2 style: no bias in final layer
    }

    public Matrix Forward (int[] batchInputIds) {
        // Update training mode and dropout rate for all layers
        foreach (var layer in Layers) {
            layer.SelfAttention.Training = Training;
            layer.SelfAttention.DropoutRate = DropoutRate;
            layer.FFN.Training = Training;
            layer.FFN.DropoutRate = DropoutRate;
        }

        var tokenEmb = TokenEmbedding.Forward (batchInputIds);
        var posEmb = PositionalEmbedding.Forward (Enumerable
            .Range (0, batchInputIds.Length)
            .Select (i => i)
            .ToArray ());

        var x = tokenEmb + posEmb;

        foreach (var layer in Layers) {
            x = layer.Forward (x);
        }

        x = FinalLayerNorm.Forward(x);
        x = FinalLayer.Forward (x);

        return x; // Returns logits
    }

    public void Backward (Matrix dLogits, int[] batchInputIds) {
        // Backward through final linear layer
        var dX = FinalLayer.Backward (dLogits);
        
        // Backward through final layer norm
        dX = FinalLayerNorm.Backward(dX);

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
