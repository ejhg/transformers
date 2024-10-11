using mingpt5;

namespace mingpt4;

class MultiHeadAttention
{
    public int NumHeads { get; set; }
    public int EmbeddingDim { get; set; }
    public int Dk { get; set; }

    public Matrix[] W_Q { get; set; }
    public Matrix[] W_K { get; set; }
    public Matrix[] W_V { get; set; }
    public Matrix W_O { get; set; }

    public Matrix[] dW_Q { get; set; }
    public Matrix[] dW_K { get; set; }
    public Matrix[] dW_V { get; set; }
    public Matrix dW_O { get; set; }

    public MultiHeadAttention (int embeddingDim, int numHeads) {
        EmbeddingDim = embeddingDim;
        NumHeads = numHeads;
        Dk = embeddingDim / numHeads;

        W_Q = new Matrix[NumHeads];
        W_K = new Matrix[NumHeads];
        W_V = new Matrix[NumHeads];

        dW_Q = new Matrix[NumHeads];
        dW_K = new Matrix[NumHeads];
        dW_V = new Matrix[NumHeads];

        Random rand = new Random ();

        for (int h = 0; h < NumHeads; h++) {
            W_Q[h] = new Matrix (EmbeddingDim, Dk);
            W_K[h] = new Matrix (EmbeddingDim, Dk);
            W_V[h] = new Matrix (EmbeddingDim, Dk);

            dW_Q[h] = new Matrix (EmbeddingDim, Dk);
            dW_K[h] = new Matrix (EmbeddingDim, Dk);
            dW_V[h] = new Matrix (EmbeddingDim, Dk);

            // Initialize weights
            for (int i = 0; i < EmbeddingDim; i++) {
                for (int j = 0; j < Dk; j++) {
                    W_Q[h].Data[i][j] = rand.NextDouble () * 0.01;
                    W_K[h].Data[i][j] = rand.NextDouble () * 0.01;
                    W_V[h].Data[i][j] = rand.NextDouble () * 0.01;
                }
            }
        }

        W_O = new Matrix (NumHeads * Dk, EmbeddingDim);
        dW_O = new Matrix (NumHeads * Dk, EmbeddingDim);

        // Initialize W_O
        for (int i = 0; i < W_O.Rows; i++)
        for (int j = 0; j < W_O.Cols; j++)
            W_O.Data[i][j] = rand.NextDouble () * 0.01;
    }

    public Matrix Forward (Matrix X) {
        int seqLength = X.Rows;
        Matrix[] heads = new Matrix[NumHeads];

        for (int h = 0; h < NumHeads; h++) {
            // Compute Q, K, V
            Matrix Q = Matrix.Multiply (X, W_Q[h]);
            Matrix K = Matrix.Multiply (X, W_K[h]);
            Matrix V = Matrix.Multiply (X, W_V[h]);

            // Compute attention
            Matrix head = ScaledDotProductAttention.ComputeAttention (Q, K, V);
            heads[h] = head;
        }

        // Concatenate heads
        Matrix concatenatedHeads = ConcatenateHeads (heads);

        // Final linear layer
        Matrix output = Matrix.Multiply (concatenatedHeads, W_O);

        return output;
    }

    private Matrix ConcatenateHeads (Matrix[] heads) {
        int seqLength = heads[0].Rows;
        int totalDim = NumHeads * Dk;
        Matrix result = new Matrix (seqLength, totalDim);

        for (int i = 0; i < seqLength; i++) {
            int colIndex = 0;
            for (int h = 0; h < NumHeads; h++) {
                for (int j = 0; j < Dk; j++) {
                    result.Data[i][colIndex] = heads[h].Data[i][j];
                    colIndex++;
                }
            }
        }

        return result;
    }

    // Backward pass (omitted for brevity)
    // Compute gradients w.r.t weights

    public void UpdateParameters (double learningRate) {
        for (int h = 0; h < NumHeads; h++) {
            for (int i = 0; i < W_Q[h].Rows; i++)
            for (int j = 0; j < W_Q[h].Cols; j++) {
                W_Q[h].Data[i][j] -= learningRate * dW_Q[h].Data[i][j];
                W_K[h].Data[i][j] -= learningRate * dW_K[h].Data[i][j];
                W_V[h].Data[i][j] -= learningRate * dW_V[h].Data[i][j];

                dW_Q[h].Data[i][j] = 0;
                dW_K[h].Data[i][j] = 0;
                dW_V[h].Data[i][j] = 0;
            }
        }

        for (int i = 0; i < W_O.Rows; i++)
        for (int j = 0; j < W_O.Cols; j++) {
            W_O.Data[i][j] -= learningRate * dW_O.Data[i][j];
            dW_O.Data[i][j] = 0;
        }
    }
}
