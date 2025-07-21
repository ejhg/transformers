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
        var attnOutput = SelfAttention.Forward (x);
        x += attnOutput;
        x = LayerNorm1.Forward (x);

        var ffnOutput = FFN.Forward (x);
        x += ffnOutput;
        x = LayerNorm2.Forward (x);
        return x;
    }

    public Matrix Backward (Matrix dOutput) {
        dOutput = LayerNorm2.Backward (dOutput);
        var dFFN = dOutput;
        var dResidual2 = dOutput;

        dFFN = FFN.Backward (dFFN);
        dResidual2 += dFFN;

        dOutput = dResidual2;
        dOutput = LayerNorm1.Backward (dOutput);
        var dAttn = dOutput;
        var dResidual1 = dOutput;

        dAttn = SelfAttention.Backward (dAttn);
        dResidual1 += dAttn;

        return dResidual1;
    }
}
