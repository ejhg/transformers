using transformers.utils;

namespace mingpt3;

public class FeedForward
{
    public LinearLayer Linear1;
    public LinearLayer Linear2;
    private Matrix Hidden;

    public FeedForward (int embeddingSize) {
        Linear1 = new LinearLayer (embeddingSize, embeddingSize * 4);
        Linear2 = new LinearLayer (embeddingSize * 4, embeddingSize);
    }

    public Matrix Forward (Matrix x) {
        Hidden = Linear1.Forward (x);
        Hidden = new Matrix (math.Relu (Hidden.Data));
        return Linear2.Forward (Hidden);
    }

    public Matrix Backward (Matrix dOutput) {
        var dHidden = Linear2.Backward (dOutput);
        dHidden = new Matrix (math.ReluBackward (Hidden.Data, dHidden));
        return Linear1.Backward (dHidden);
    }
}
