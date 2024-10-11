using mingpt3;

namespace mingpt1;

public class TransformerBlock
{
    public int EmbeddingSize, NumHeads;
    public MultiHeadSelfAttention SelfAttention;
    public LayerNorm LayerNorm1;
    public FeedForward FFN;
    public LayerNorm LayerNorm2;
    private Matrix Residual1;
    private Matrix Residual2;

    public TransformerBlock (int embeddingSize, int numHeads) {
        EmbeddingSize = embeddingSize;
        NumHeads = numHeads;
        SelfAttention = new MultiHeadSelfAttention (embeddingSize, numHeads);
        LayerNorm1 = new LayerNorm (embeddingSize);
        FFN = new FeedForward (embeddingSize);
        LayerNorm2 = new LayerNorm (embeddingSize);
    }

    public Matrix Forward (Matrix x) {
        Residual1 = x;
        var attnOutput = SelfAttention.Forward (x);
        x = x + attnOutput;
        x = LayerNorm1.Forward (x);

        Residual2 = x;
        var ffnOutput = FFN.Forward (x);
        x = x + ffnOutput;
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
