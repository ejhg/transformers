using mingpt5;

namespace mingpt4;

class FeedForwardNetwork
{
    public int EmbeddingDim { get; set; }
    public int HiddenDim { get; set; }
    public Matrix W1 { get; set; }
    public Vector b1 { get; set; }
    public Matrix W2 { get; set; }
    public Vector b2 { get; set; }

    public Matrix dW1 { get; set; }
    public Vector db1 { get; set; }
    public Matrix dW2 { get; set; }
    public Vector db2 { get; set; }

    public FeedForwardNetwork (int embeddingDim, int hiddenDim) {
        EmbeddingDim = embeddingDim;
        HiddenDim = hiddenDim;

        W1 = new Matrix (EmbeddingDim, HiddenDim);
        b1 = new Vector (HiddenDim);
        W2 = new Matrix (HiddenDim, EmbeddingDim);
        b2 = new Vector (EmbeddingDim);

        dW1 = new Matrix (EmbeddingDim, HiddenDim);
        db1 = new Vector (HiddenDim);
        dW2 = new Matrix (HiddenDim, EmbeddingDim);
        db2 = new Vector (EmbeddingDim);

        Random rand = new Random ();

        // Initialize weights
        for (int i = 0; i < EmbeddingDim; i++)
        for (int j = 0; j < HiddenDim; j++)
            W1.Data[i][j] = rand.NextDouble () * 0.01;

        for (int i = 0; i < HiddenDim; i++)
        for (int j = 0; j < EmbeddingDim; j++)
            W2.Data[i][j] = rand.NextDouble () * 0.01;
    }

    public Matrix Forward (Matrix X) {
        int seqLength = X.Rows;
        Matrix hidden = new Matrix (seqLength, HiddenDim);
        Matrix output = new Matrix (seqLength, EmbeddingDim);

        // First linear layer + ReLU activation
        for (int i = 0; i < seqLength; i++) {
            for (int j = 0; j < HiddenDim; j++) {
                double sum = 0;
                for (int k = 0; k < EmbeddingDim; k++)
                    sum += X.Data[i][k] * W1.Data[k][j];
                sum += b1.Data[j];
                hidden.Data[i][j] = Math.Max (0, sum); // ReLU activation
            }
        }

        // Second linear layer
        for (int i = 0; i < seqLength; i++) {
            for (int j = 0; j < EmbeddingDim; j++) {
                double sum = 0;
                for (int k = 0; k < HiddenDim; k++)
                    sum += hidden.Data[i][k] * W2.Data[k][j];
                sum += b2.Data[j];
                output.Data[i][j] = sum;
            }
        }

        return output;
    }

    // Backward pass (omitted for brevity)
    // Compute gradients w.r.t weights and biases

    public void UpdateParameters (double learningRate) {
        for (int i = 0; i < W1.Rows; i++)
        for (int j = 0; j < W1.Cols; j++) {
            W1.Data[i][j] -= learningRate * dW1.Data[i][j];
            dW1.Data[i][j] = 0;
        }

        for (int i = 0; i < b1.Size; i++) {
            b1.Data[i] -= learningRate * db1.Data[i];
            db1.Data[i] = 0;
        }

        for (int i = 0; i < W2.Rows; i++)
        for (int j = 0; j < W2.Cols; j++) {
            W2.Data[i][j] -= learningRate * dW2.Data[i][j];
            dW2.Data[i][j] = 0;
        }

        for (int i = 0; i < b2.Size; i++) {
            b2.Data[i] -= learningRate * db2.Data[i];
            db2.Data[i] = 0;
        }
    }
}
