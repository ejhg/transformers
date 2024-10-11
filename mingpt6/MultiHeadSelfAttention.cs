namespace mingpt6;

public class MultiHeadSelfAttention
{
    private int numHeads;
    private int hiddenSize;
    private int headDim;

    // Weight matrices
    private Matrix Wq;
    private Matrix Wk;
    private Matrix Wv;
    private Matrix Wo;

    private RoPE rope;

    public MultiHeadSelfAttention (int numHeads, int hiddenSize, int maxPosition) {
        this.numHeads = numHeads;
        this.hiddenSize = hiddenSize;
        this.headDim = hiddenSize / numHeads;

        Wq = new Matrix (hiddenSize, hiddenSize);
        Wk = new Matrix (hiddenSize, hiddenSize);
        Wv = new Matrix (hiddenSize, hiddenSize);
        Wo = new Matrix (hiddenSize, hiddenSize);

        InitializeMatrix (Wq);
        InitializeMatrix (Wk);
        InitializeMatrix (Wv);
        InitializeMatrix (Wo);

        rope = new RoPE (hiddenSize, maxPosition);
    }

    private void InitializeMatrix (Matrix m) {
        Random rand = new Random ();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = rand.NextDouble () * 0.02 - 0.01; // Small random values
    }

    public double[] Forward (double[][] inputs, int position) {
        int seqLength = inputs.Length;
        double[][] Q = new double[seqLength][];
        double[][] K = new double[seqLength][];
        double[][] V = new double[seqLength][];

        // Compute Q, K, V
        for (int t = 0; t < seqLength; t++) {
            Q[t] = MultiplyMatrixVector (Wq, inputs[t]);
            K[t] = MultiplyMatrixVector (Wk, inputs[t]);
            V[t] = MultiplyMatrixVector (Wv, inputs[t]);

            // Apply RoPE to Q and K
            Q[t] = rope.ApplyRoPE (Q[t], t);
            K[t] = rope.ApplyRoPE (K[t], t);
        }

        // Split Q, K, V into heads
        double[][][] Q_heads = SplitHeads (Q);
        double[][][] K_heads = SplitHeads (K);
        double[][][] V_heads = SplitHeads (V);

        double[][][] attentionOutputs = new double[numHeads][][];
        for (int h = 0; h < numHeads; h++) {
            // Scaled dot-product attention
            double[][] attentionScores = new double[seqLength][];
            for (int t = 0; t < seqLength; t++) {
                attentionScores[t] = new double[seqLength];
                for (int s = 0; s <= t; s++) // Causal mask
                {
                    double score = Vector.Dot (new Vector (Q_heads[h][t]), new Vector (K_heads[h][s])) / Math.Sqrt (headDim);
                    attentionScores[t][s] = score;
                }
            }

            // Apply softmax
            double[][] attentionWeights = new double[seqLength][];
            for (int t = 0; t < seqLength; t++) {
                double[] scores = attentionScores[t];
                double maxScore = double.MinValue;
                for (int s = 0; s <= t; s++)
                    if (scores[s] > maxScore)
                        maxScore = scores[s];
                double sumExp = 0;
                for (int s = 0; s <= t; s++) {
                    scores[s] = Math.Exp (scores[s] - maxScore);
                    sumExp += scores[s];
                }

                for (int s = 0; s <= t; s++)
                    scores[s] /= sumExp;
                attentionWeights[t] = scores;
            }

            // Compute attention output
            double[][] context = new double[seqLength][];
            for (int t = 0; t < seqLength; t++) {
                context[t] = new double[headDim];
                for (int s = 0; s <= t; s++) {
                    double weight = attentionWeights[t][s];
                    for (int i = 0; i < headDim; i++)
                        context[t][i] += weight * V_heads[h][s][i];
                }
            }

            attentionOutputs[h] = context;
        }

        // Concatenate heads
        double[][] concatenated = new double[seqLength][];
        for (int t = 0; t < seqLength; t++) {
            concatenated[t] = new double[hiddenSize];
            for (int h = 0; h < numHeads; h++)
                Array.Copy (attentionOutputs[h][t], 0, concatenated[t], h * headDim, headDim);
        }

        // Apply final linear projection
        double[][] outputs = new double[seqLength][];
        for (int t = 0; t < seqLength; t++)
            outputs[t] = MultiplyMatrixVector (Wo, concatenated[t]);

        // For simplicity, we return the last output
        return outputs[seqLength - 1];
    }

    private double[] MultiplyMatrixVector (Matrix m, double[] v) {
        double[] result = new double[m.Rows];
        for (int i = 0; i < m.Rows; i++) {
            result[i] = 0;
            for (int j = 0; j < m.Cols; j++)
                result[i] += m.Data[i][j] * v[j];
        }

        return result;
    }

    private double[][][] SplitHeads (double[][] x) {
        int seqLength = x.Length;
        double[][][] result = new double[numHeads][][];
        for (int h = 0; h < numHeads; h++)
            result[h] = new double[seqLength][];

        for (int t = 0; t < seqLength; t++) {
            for (int h = 0; h < numHeads; h++) {
                result[h][t] = new double[headDim];
                Array.Copy (x[t], h * headDim, result[h][t], 0, headDim);
            }
        }

        return result;
    }
}
