using transformers.utils;

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

    private double[] Softmax (double[] logits) {
        double maxLogit = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > maxLogit)
                maxLogit = logits[i];

        double sumExp = 0.0;
        var expLogits = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++) {
            expLogits[i] = Math.Exp (logits[i] - maxLogit);
            sumExp += expLogits[i];
        }

        var probabilities = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            probabilities[i] = expLogits[i] / sumExp;

        return probabilities;
    }

    public int PredictNextToken (int[] inputIds, double temperature = 1.0, int topK = 10, bool argmax = false) {
        var logits = Forward (inputIds);
        var lastLogits = new double[VocabSize];
        for (int i = 0; i < VocabSize; i++)
            lastLogits[i] = logits.Data[inputIds.Length - 1, i] / temperature;

        if (topK > 0) {
            lastLogits = TopKFilter (lastLogits, topK);
        }

        // Apply softmax to convert logits to probabilities
        var probabilities = Softmax (lastLogits);

        // Sample the next token based on probabilities
        return argmax
            ? sampling.ArgMax (probabilities)
            : sampling.SampleFromDistribution (probabilities);
    }

    private double[] TopKFilter (double[] logits, int k) {
        var filteredLogits = new double[logits.Length];
        Array.Copy (logits, filteredLogits, logits.Length);

        var indices = new int[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            indices[i] = i;

        Array.Sort (logits, indices);
        Array.Reverse (logits);
        Array.Reverse (indices);

        for (int i = k; i < logits.Length; i++)
            filteredLogits[indices[i]] = double.NegativeInfinity;

        return filteredLogits;
    }
}
