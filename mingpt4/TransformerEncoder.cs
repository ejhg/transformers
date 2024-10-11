namespace mingpt4;

class TransformerEncoder
{
    public int NumLayers { get; set; }
    public EncoderLayer[] EncoderLayers { get; set; }

    public TransformerEncoder (int numLayers, int embeddingDim, int numHeads, int hiddenDim) {
        NumLayers = numLayers;
        EncoderLayers = new EncoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++)
            EncoderLayers[i] = new EncoderLayer (embeddingDim, numHeads, hiddenDim);
    }

    public Matrix Forward (Matrix X) {
        Matrix output = X;
        for (int i = 0; i < NumLayers; i++)
            output = EncoderLayers[i].Forward (output);
        return output;
    }

    // Backward pass (omitted for brevity)

    public void UpdateParameters (double learningRate) {
        for (int i = 0; i < NumLayers; i++)
            EncoderLayers[i].UpdateParameters (learningRate);
    }
}
