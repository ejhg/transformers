namespace mingpt5;

public class MultiHeadAttention
{
    public int EmbeddingDim;
    public int NumHeads;
    public int HeadDim;
    public Matrix[] Wq;
    public Matrix[] Wk;
    public Matrix[] Wv;

    public Matrix Wo;

    // Cache for backward pass
    public Vector[] Q;
    public Vector[][] K;
    public Vector[][] V;
    public double[][] AttentionWeights;
    public Vector[] Inputs;

    public MultiHeadAttention (int embeddingDim, int numHeads) {
        EmbeddingDim = embeddingDim;
        NumHeads = numHeads;
        HeadDim = EmbeddingDim / NumHeads;

        Wq = new Matrix[NumHeads];
        Wk = new Matrix[NumHeads];
        Wv = new Matrix[NumHeads];

        for (int i = 0; i < NumHeads; i++) {
            Wq[i] = new Matrix (HeadDim, EmbeddingDim);
            Wk[i] = new Matrix (HeadDim, EmbeddingDim);
            Wv[i] = new Matrix (HeadDim, EmbeddingDim);
            InitializeMatrix (Wq[i]);
            InitializeMatrix (Wk[i]);
            InitializeMatrix (Wv[i]);
        }

        Wo = new Matrix (EmbeddingDim, EmbeddingDim);
        InitializeMatrix (Wo);
    }

    private void InitializeMatrix (Matrix m) {
        Random rand = new Random ();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = (rand.NextDouble () - 0.5) / m.Cols;
    }

    public Vector Forward (Vector[] inputs, int position) {
        Inputs = inputs;
        int seqLength = inputs.Length;
        Q = new Vector[NumHeads];
        K = new Vector[NumHeads][];
        V = new Vector[NumHeads][];
        AttentionWeights = new double[NumHeads][];
        Vector concatHeads = new Vector (EmbeddingDim);

        for (int h = 0; h < NumHeads; h++) {
            // Compute query, key, value for this head
            Q[h] = Wq[h].Multiply (inputs[position]);
            K[h] = new Vector[position + 1];
            V[h] = new Vector[position + 1];
            AttentionWeights[h] = new double[position + 1];
            for (int t = 0; t <= position; t++) // Autoregressive mask
            {
                K[h][t] = Wk[h].Multiply (inputs[t]);
                V[h][t] = Wv[h].Multiply (inputs[t]);
            }

            // Compute attention scores
            double[] scores = new double[position + 1];
            for (int t = 0; t <= position; t++) {
                scores[t] = Q[h].Dot (K[h][t]) / Math.Sqrt (HeadDim);
            }

            // Apply softmax
            double maxScore = double.MinValue;
            for (int t = 0; t <= position; t++) {
                if (scores[t] > maxScore) maxScore = scores[t];
            }

            double sumExp = 0.0;
            for (int t = 0; t <= position; t++) {
                scores[t] = Math.Exp (scores[t] - maxScore);
                sumExp += scores[t];
            }

            for (int t = 0; t <= position; t++) {
                scores[t] /= sumExp;
            }

            AttentionWeights[h] = scores;

            // Compute weighted sum of values
            Vector headOutput = new Vector (HeadDim);
            for (int t = 0; t <= position; t++) {
                for (int i = 0; i < HeadDim; i++) {
                    headOutput.Data[i] += scores[t] * V[h][t].Data[i];
                }
            }

            // Concatenate head outputs
            for (int i = 0; i < HeadDim; i++) {
                concatHeads.Data[h * HeadDim + i] = headOutput.Data[i];
            }
        }

        // Apply Wo
        Vector output = Wo.Multiply (concatHeads);
        return output;
    }

    public Vector Backward (Vector dout, int position) {
        Vector dConcatHeads = Wo.Transpose ().Multiply (dout);
        // Gradients for Wo
        Matrix dWo = Matrix.OuterProduct (dout, dConcatHeads);

        // Initialize gradients
        Vector[] dInputs = new Vector[Inputs.Length];
        for (int i = 0; i < Inputs.Length; i++) {
            dInputs[i] = new Vector (EmbeddingDim);
        }

        // Backpropagate through heads
        for (int h = 0; h < NumHeads; h++) {
            Vector dHeadOutput = new Vector (HeadDim);
            for (int i = 0; i < HeadDim; i++) {
                dHeadOutput.Data[i] = dConcatHeads.Data[h * HeadDim + i];
            }

            // Backprop through weighted sum of values
            Vector[] dV = new Vector[position + 1];
            double[] dScores = new double[position + 1];
            for (int t = 0; t <= position; t++) {
                dV[t] = new Vector (HeadDim);
            }

            for (int t = 0; t <= position; t++) {
                double attnWeight = AttentionWeights[h][t];
                for (int i = 0; i < HeadDim; i++) {
                    dV[t].Data[i] += dHeadOutput.Data[i] * attnWeight;
                }
            }

            // Backprop through attention weights
            for (int t = 0; t <= position; t++) {
                double attnWeight = AttentionWeights[h][t];
                dScores[t] = 0.0;
                for (int i = 0; i < HeadDim; i++) {
                    dScores[t] += dHeadOutput.Data[i] * V[h][t].Data[i];
                }

                dScores[t] *= attnWeight;
            }

            // Backprop through scores to Q, K
            Vector dQ = new Vector (HeadDim);
            Vector[] dK = new Vector[position + 1];
            for (int t = 0; t <= position; t++) {
                dK[t] = new Vector (HeadDim);
            }

            for (int t = 0; t <= position; t++) {
                double scale = 1.0 / Math.Sqrt (HeadDim);
                for (int i = 0; i < HeadDim; i++) {
                    double temp = dScores[t] * scale;
                    dQ.Data[i] += temp * K[h][t].Data[i];
                    dK[t].Data[i] += temp * Q[h].Data[i];
                }
            }

            // Backprop through linear layers Wq, Wk, Wv
            Vector dInputQ = Wq[h].Transpose ().Multiply (dQ);
            Matrix dWq = Matrix.OuterProduct (dQ, Inputs[position]);

            for (int t = 0; t <= position; t++) {
                Vector dInputK = Wk[h].Transpose ().Multiply (dK[t]);
                Vector dInputV = Wv[h].Transpose ().Multiply (dV[t]);
                Matrix dWk = Matrix.OuterProduct (dK[t], Inputs[t]);
                Matrix dWv = Matrix.OuterProduct (dV[t], Inputs[t]);

                // Accumulate gradients
                dInputs[t] = dInputs[t] + dInputK + dInputV;
                // Update Wk[h] and Wv[h]
                for (int r = 0; r < Wk[h].Rows; r++)
                for (int c = 0; c < Wk[h].Cols; c++) {
                    Wk[h].Data[r][c] -= dWk.Data[r][c];
                    Wv[h].Data[r][c] -= dWv.Data[r][c];
                }
            }

            dInputs[position] = dInputs[position] + dInputQ;
            // Update Wq[h]
            for (int r = 0; r < Wq[h].Rows; r++)
            for (int c = 0; c < Wq[h].Cols; c++) {
                Wq[h].Data[r][c] -= dWq.Data[r][c];
            }
        }

        // Update Wo
        for (int r = 0; r < Wo.Rows; r++)
        for (int c = 0; c < Wo.Cols; c++) {
            Wo.Data[r][c] -= dWo.Data[r][c];
        }

        return dInputs[position];
    }
}
