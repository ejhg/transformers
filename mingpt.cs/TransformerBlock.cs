namespace mingpt.cs;

public class TransformerBlock
{
    public MultiHeadSelfAttention SelfAttention;
    public LayerNorm LayerNorm1;
    public FeedForward FFN;
    public LayerNorm LayerNorm2;

    public TransformerBlock (int embeddingSize, int numHeads) {
        SelfAttention = new MultiHeadSelfAttention (embeddingSize, numHeads);
        LayerNorm1 = new LayerNorm (embeddingSize);
        FFN = new FeedForward (embeddingSize);
        LayerNorm2 = new LayerNorm (embeddingSize);
    }

    public Matrix Forward (Matrix x) {
        // Pre-norm: LayerNorm before attention, then residual connection
        x += SelfAttention.Forward(LayerNorm1.Forward(x));

        // Pre-norm: LayerNorm before FFN, then residual connection
        x += FFN.Forward(LayerNorm2.Forward(x));

        return x;
    }

    public Matrix Backward (Matrix dOutput) {
        // Backward through second residual connection (x + FFN(LayerNorm2(x)))
        var dResidual2 = dOutput; // gradient to x
        var dFFN = FFN.Backward(dOutput); // gradient through FFN
        var dLayerNorm2 = LayerNorm2.Backward(dFFN); // gradient through LayerNorm2
        dResidual2 += dLayerNorm2; // combine gradients

        // Backward through first residual connection (x + SelfAttention(LayerNorm1(x)))
        var dResidual1 = dResidual2; // gradient to x
        var dAttn = SelfAttention.Backward(dResidual2); // gradient through attention
        var dLayerNorm1 = LayerNorm1.Backward(dAttn); // gradient through LayerNorm1
        dResidual1 += dLayerNorm1; // combine gradients

        return dResidual1;
    }
}
