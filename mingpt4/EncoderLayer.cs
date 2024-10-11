using mingpt5;

namespace mingpt4;

class EncoderLayer
{
    public MultiHeadAttention MultiHeadAttention { get; set; }
    public LayerNormalization LayerNorm1 { get; set; }
    public FeedForwardNetwork FeedForward { get; set; }
    public LayerNormalization LayerNorm2 { get; set; }

    public EncoderLayer (int embeddingDim, int numHeads, int hiddenDim) {
        MultiHeadAttention = new MultiHeadAttention (embeddingDim, numHeads);
        LayerNorm1 = new LayerNormalization (embeddingDim);
        FeedForward = new FeedForwardNetwork (embeddingDim, hiddenDim);
        LayerNorm2 = new LayerNormalization (embeddingDim);
    }

    public Matrix Forward (Matrix X) {
        // Multi-head attention
        Matrix attnOutput = MultiHeadAttention.Forward (X);

        // Add & Norm
        Matrix attnOutputNorm = LayerNorm1.Forward (Matrix.Add (X, attnOutput));

        // Feedforward
        Matrix ffOutput = FeedForward.Forward (attnOutputNorm);

        // Add & Norm
        Matrix output = LayerNorm2.Forward (Matrix.Add (attnOutputNorm, ffOutput));

        return output;
    }

    // Backward pass (omitted for brevity)

    public void UpdateParameters (double learningRate) {
        MultiHeadAttention.UpdateParameters (learningRate);
        FeedForward.UpdateParameters (learningRate);
        LayerNorm1.UpdateParameters (learningRate);
        LayerNorm2.UpdateParameters (learningRate);
    }
}
