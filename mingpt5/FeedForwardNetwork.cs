namespace mingpt5;

public class FeedForwardNetwork
{
    public int EmbeddingDim;
    public int HiddenDim;
    public Matrix W1;
    public Vector b1;
    public Matrix W2;
    public Vector b2;
    public Vector Hidden;
    public Vector Input;

    public FeedForwardNetwork (int embeddingDim, int hiddenDim) {
        EmbeddingDim = embeddingDim;
        HiddenDim = hiddenDim;
        W1 = new Matrix (HiddenDim, EmbeddingDim);
        b1 = new Vector (HiddenDim);
        W2 = new Matrix (EmbeddingDim, HiddenDim);
        b2 = new Vector (EmbeddingDim);

        InitializeMatrix (W1);
        InitializeMatrix (W2);
        InitializeVector (b1);
        InitializeVector (b2);
    }

    private void InitializeMatrix (Matrix m) {
        Random rand = new Random ();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = (rand.NextDouble () - 0.5) / m.Cols;
    }

    private void InitializeVector (Vector v) {
        for (int i = 0; i < v.Size; i++)
            v.Data[i] = 0.0;
    }

    public Vector Forward (Vector input) {
        Input = input.Clone ();
        Hidden = W1.Multiply (Input);
        Hidden = Hidden + b1;
        // ReLU activation
        for (int i = 0; i < Hidden.Size; i++) {
            Hidden.Data[i] = Math.Max (0, Hidden.Data[i]);
        }

        Vector output = W2.Multiply (Hidden);
        output = output + b2;
        return output;
    }

    public Vector Backward (Vector dout) {
        Vector dHidden = W2.Transpose ().Multiply (dout);
        // Gradients for W2 and b2
        Matrix dW2 = Matrix.OuterProduct (dout, Hidden);
        Vector db2 = dout.Clone ();

        // Backprop through ReLU
        for (int i = 0; i < dHidden.Size; i++) {
            if (Hidden.Data[i] <= 0) {
                dHidden.Data[i] = 0;
            }
        }

        // Gradients for W1 and b1
        Matrix dW1 = Matrix.OuterProduct (dHidden, Input);
        Vector db1 = dHidden.Clone ();
        Vector dx = W1.Transpose ().Multiply (dHidden);

        // Update parameters
        for (int r = 0; r < W1.Rows; r++)
        for (int c = 0; c < W1.Cols; c++) {
            W1.Data[r][c] -= dW1.Data[r][c];
        }

        for (int i = 0; i < b1.Size; i++) {
            b1.Data[i] -= db1.Data[i];
        }

        for (int r = 0; r < W2.Rows; r++)
        for (int c = 0; c < W2.Cols; c++) {
            W2.Data[r][c] -= dW2.Data[r][c];
        }

        for (int i = 0; i < b2.Size; i++) {
            b2.Data[i] -= db2.Data[i];
        }

        return dx;
    }
}
