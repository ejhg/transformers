using mingpt3;

namespace mingpt1;

public class FeedForward
{
    public int EmbeddingSize;
    public LinearLayer Linear1;
    public LinearLayer Linear2;
    private Matrix Input;
    private Matrix Hidden;

    public FeedForward (int embeddingSize) {
        EmbeddingSize = embeddingSize;
        Linear1 = new LinearLayer (embeddingSize, embeddingSize * 4);
        Linear2 = new LinearLayer (embeddingSize * 4, embeddingSize);
    }

    public Matrix Forward (Matrix x) {
        Input = x;
        Hidden = Linear1.Forward (x);
        Hidden = Relu (Hidden);
        var output = Linear2.Forward (Hidden);
        return output;
    }

    public Matrix Backward (Matrix dOutput) {
        var dHidden = Linear2.Backward (dOutput);
        dHidden = ReluBackward (Hidden, dHidden);
        var dInput = Linear1.Backward (dHidden);
        return dInput;
    }

    private Matrix Relu (Matrix x) {
        var result = new Matrix (x.Rows, x.Cols);
        for (int i = 0; i < x.Rows; i++)
        for (int j = 0; j < x.Cols; j++)
            result.Data[i, j] = Math.Max (0, x.Data[i, j]);
        return result;
    }

    private Matrix ReluBackward (Matrix x, Matrix dOutput) {
        var result = new Matrix (dOutput.Rows, dOutput.Cols);
        for (int i = 0; i < x.Rows; i++)
        for (int j = 0; j < x.Cols; j++)
            result.Data[i, j] = x.Data[i, j] > 0 ? dOutput.Data[i, j] : 0;
        return result;
    }
}
