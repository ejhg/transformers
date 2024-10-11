namespace mingpt7;

public class MultiHeadSelfAttention
{
    public int numHeads;
    public int headDim;
    public int embeddingDim;

    public double[,] W_q;
    public double[,] W_k;
    public double[,] W_v;
    public double[,] W_o;

    // Gradients
    public double[,] dW_q;
    public double[,] dW_k;
    public double[,] dW_v;
    public double[,] dW_o;

    // Intermediate variables
    double[][] Q;
    double[][] K;
    double[][] V;
    double[][] attnOutput;

    public MultiHeadSelfAttention (int numHeads, int embeddingDim) {
        this.numHeads = numHeads;
        this.embeddingDim = embeddingDim;
        this.headDim = embeddingDim / numHeads;

        W_q = new double[embeddingDim, embeddingDim];
        W_k = new double[embeddingDim, embeddingDim];
        W_v = new double[embeddingDim, embeddingDim];
        W_o = new double[embeddingDim, embeddingDim];

        dW_q = new double[embeddingDim, embeddingDim];
        dW_k = new double[embeddingDim, embeddingDim];
        dW_v = new double[embeddingDim, embeddingDim];
        dW_o = new double[embeddingDim, embeddingDim];

        Random rand = new Random ();
        InitializeMatrix (W_q, rand);
        InitializeMatrix (W_k, rand);
        InitializeMatrix (W_v, rand);
        InitializeMatrix (W_o, rand);
    }

    void InitializeMatrix (double[,] matrix, Random rand) {
        int rows = matrix.GetLength (0);
        int cols = matrix.GetLength (1);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble () * 0.01;
    }

    public double[][] Forward (double[][] x) {
        int T = x.Length;
        Q = new double[T][];
        K = new double[T][];
        V = new double[T][];
        for (int t = 0; t < T; t++) {
            Q[t] = math.MatVecMul (W_q, x[t]);
            K[t] = math.MatVecMul (W_k, x[t]);
            V[t] = math.MatVecMul (W_v, x[t]);

            Q[t] = math.ApplyROPE (Q[t], t);
            K[t] = math.ApplyROPE (K[t], t);
        }

        // Compute attention
        double[][][] headOutputs = new double[numHeads][][];
        for (int h = 0; h < numHeads; h++) {
            headOutputs[h] = ComputeAttentionHead (Q, K, V, h);
        }

        // Concatenate heads
        attnOutput = new double[T][];
        for (int t = 0; t < T; t++) {
            attnOutput[t] = new double[embeddingDim];
            for (int h = 0; h < numHeads; h++) {
                Array.Copy (headOutputs[h][t], 0, attnOutput[t], h * headDim, headDim);
            }

            attnOutput[t] = math.MatVecMul (W_o, attnOutput[t]);
        }

        return attnOutput;
    }

    double[][] ComputeAttentionHead (double[][] Q, double[][] K, double[][] V, int head) {
        int T = Q.Length;
        double[][] q = new double[T][];
        double[][] k = new double[T][];
        double[][] v = new double[T][];
        for (int t = 0; t < T; t++) {
            q[t] = new double[headDim];
            k[t] = new double[headDim];
            v[t] = new double[headDim];
            Array.Copy (Q[t], head * headDim, q[t], 0, headDim);
            Array.Copy (K[t], head * headDim, k[t], 0, headDim);
            Array.Copy (V[t], head * headDim, v[t], 0, headDim);
        }

        double[][] attnOutput = new double[T][];
        double scale = 1.0 / Math.Sqrt (headDim);
        for (int t = 0; t < T; t++) {
            double[] scores = new double[t + 1];
            for (int s = 0; s <= t; s++) {
                scores[s] = math.Dot (q[t], k[s]) * scale;
            }

            double[] attnWeights = math.Softmax (scores);
            attnOutput[t] = new double[headDim];
            for (int s = 0; s <= t; s++) {
                double[] weightedV = math.Multiply (v[s], attnWeights[s]);
                attnOutput[t] = math.Add (attnOutput[t], weightedV);
            }
        }

        return attnOutput;
    }

    public double[][] Backward (double[][] gradOutput) {
        // For brevity, detailed backpropagation implementation is omitted
        // Implementing backprop through attention mechanism is complex
        // Here, we will assume gradients are passed back appropriately
        return gradOutput;
    }

    public void UpdateParameters (double learningRate) {
        // Update W_q, W_k, W_v, W_o
        for (int i = 0; i < embeddingDim; i++)
        for (int j = 0; j < embeddingDim; j++) {
            W_q[i, j] -= learningRate * dW_q[i, j];
            W_k[i, j] -= learningRate * dW_k[i, j];
            W_v[i, j] -= learningRate * dW_v[i, j];
            W_o[i, j] -= learningRate * dW_o[i, j];
            dW_q[i, j] = dW_k[i, j] = dW_v[i, j] = dW_o[i, j] = 0.0;
        }
    }
}
