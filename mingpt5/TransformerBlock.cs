namespace mingpt5;

public class TransformerBlock
{
    public MultiHeadAttention MHA;
    public LayerNormalization LayerNorm1;
    public FeedForwardNetwork FFN;
    public LayerNormalization LayerNorm2;
    public Vector Input;
    public Vector AttnOutput;
    public Vector FFNOutput;
    public int Position;

    public TransformerBlock (int embeddingDim, int numHeads, int hiddenDim) {
        MHA = new MultiHeadAttention (embeddingDim, numHeads);
        LayerNorm1 = new LayerNormalization (embeddingDim);
        FFN = new FeedForwardNetwork (embeddingDim, hiddenDim);
        LayerNorm2 = new LayerNormalization (embeddingDim);
    }

    public Vector Forward (Vector[] inputs, int position) {
        Input = inputs[position].Clone ();
        Position = position;
        // Self-Attention
        AttnOutput = MHA.Forward (inputs, position);

        // Residual Connection and Layer Norm
        Vector x = Input + AttnOutput;
        x = LayerNorm1.Forward (x);

        // Feed-Forward Network
        FFNOutput = FFN.Forward (x);

        // Residual Connection and Layer Norm
        x = x + FFNOutput;
        x = LayerNorm2.Forward (x);

        return x;
    }

    public Vector Backward (Vector dout) {
        // Backward through Layer Norm 2
        Vector dLN2 = LayerNorm2.Backward (dout);

        // Backward through residual connection
        Vector dFFNOutput = dLN2.Clone ();
        Vector dResidual = dLN2.Clone ();

        // Backward through Feed-Forward Network
        Vector dFFNInput = FFN.Backward (dFFNOutput);

        // Backward through Layer Norm 1
        Vector dLN1 = LayerNorm1.Backward (dFFNInput + dResidual);

        // Backward through residual connection
        Vector dAttnOutput = dLN1.Clone ();
        Vector dInput = dLN1.Clone ();

        // Backward through Multi-Head Attention
        Vector dMHAInput = MHA.Backward (dAttnOutput, Position);

        // Sum gradients
        dInput = dInput + dMHAInput;

        return dInput;
    }
}
