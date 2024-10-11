namespace mingpt4;

class TransformerModel
{
    public EmbeddingLayer EmbeddingLayer { get; set; }
    public PositionalEncoding PositionalEncoding { get; set; }
    public TransformerEncoder Encoder { get; set; }
    public OutputLayer OutputLayer { get; set; }
    public int MaxSeqLength { get; set; }
    public int VocabSize { get; set; }
    public int EmbeddingDim { get; set; }

    public TransformerModel (int vocabSize, int maxSeqLength, int embeddingDim, int numHeads, int numLayers, int hiddenDim) {
        VocabSize = vocabSize;
        MaxSeqLength = maxSeqLength;
        EmbeddingDim = embeddingDim;

        EmbeddingLayer = new EmbeddingLayer (vocabSize, embeddingDim);
        PositionalEncoding = new PositionalEncoding (maxSeqLength, embeddingDim);
        Encoder = new TransformerEncoder (numLayers, embeddingDim, numHeads, hiddenDim);
        OutputLayer = new OutputLayer (embeddingDim, vocabSize);
    }

    public Matrix Forward (int[] inputTokens) {
        Matrix embeddings = EmbeddingLayer.Forward (inputTokens);
        Matrix positionEncoded = PositionalEncoding.AddPositionalEncoding (embeddings);
        Matrix encoderOutput = Encoder.Forward (positionEncoded);
        Matrix logits = OutputLayer.Forward (encoderOutput);
        return logits;
    }

    // Backward pass (omitted for brevity)

    public void UpdateParameters (double learningRate) {
        EmbeddingLayer.UpdateParameters (learningRate);
        Encoder.UpdateParameters (learningRate);
        OutputLayer.UpdateParameters (learningRate);
    }
}
