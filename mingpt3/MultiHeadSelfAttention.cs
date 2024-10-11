namespace mingpt3;

public class MultiHeadSelfAttention
{
    public int EmbeddingSize, NumHeads, HeadSize;
    public Matrix Wq, Wk, Wv, Wo;
    public Matrix GradWq, GradWk, GradWv, GradWo;

    private Matrix Q, K, V, AttentionOutput;
    private Matrix[] Q_heads, K_heads, V_heads, AttnWeights, HeadOutputs;

    public MultiHeadSelfAttention (int embeddingSize, int numHeads) {
        EmbeddingSize = embeddingSize;
        NumHeads = numHeads;
        HeadSize = embeddingSize / numHeads;

        Wq = Matrix.Random (embeddingSize, embeddingSize);
        Wk = Matrix.Random (embeddingSize, embeddingSize);
        Wv = Matrix.Random (embeddingSize, embeddingSize);
        Wo = Matrix.Random (embeddingSize, embeddingSize);

        GradWq = new Matrix (embeddingSize, embeddingSize);
        GradWk = new Matrix (embeddingSize, embeddingSize);
        GradWv = new Matrix (embeddingSize, embeddingSize);
        GradWo = new Matrix (embeddingSize, embeddingSize);
    }

    public Matrix Forward (Matrix x) {
        Q = x * Wq;
        K = x * Wk;
        V = x * Wv;

        Q_heads = SplitHeads (Q);
        K_heads = SplitHeads (K);
        V_heads = SplitHeads (V);

        HeadOutputs = new Matrix[NumHeads];
        AttnWeights = new Matrix[NumHeads];

        for (int i = 0; i < NumHeads; i++) {
            var scores = (Q_heads[i] * K_heads[i].Transpose ()) / Math.Sqrt (HeadSize);
            var attn_weights = Softmax (scores);
            var attn_output = attn_weights * V_heads[i];

            AttnWeights[i] = attn_weights;
            HeadOutputs[i] = attn_output;
        }

        var concat = ConcatenateHeads (HeadOutputs);
        AttentionOutput = concat;
        var output = concat * Wo;
        return output;
    }

    public Matrix Backward (Matrix dOutput) {
        // Backprop through Wo
        var dConcat = dOutput * Wo.Transpose ();
        GradWo += AttentionOutput.Transpose () * dOutput;

        // Split gradients for heads
        var dHeadOutputs = SplitHeads (dConcat);

        var dQ = new Matrix (Q.Rows, Q.Cols);
        var dK = new Matrix (K.Rows, K.Cols);
        var dV = new Matrix (V.Rows, V.Cols);

        for (int i = 0; i < NumHeads; i++) {
            var dAttnOutput = dHeadOutputs[i];
            var dAttnWeights = dAttnOutput * V_heads[i].Transpose ();
            var dV_head = AttnWeights[i].Transpose () * dAttnOutput;

            // Backprop through Softmax
            var dScores = SoftmaxBackward (AttnWeights[i], dAttnWeights);

            var dQ_head = dScores * K_heads[i];
            var dK_head = dScores.Transpose () * Q_heads[i];

            // Aggregate gradients
            AddToMatrix (dV, dV_head, i);
            AddToMatrix (dQ, dQ_head, i);
            AddToMatrix (dK, dK_head, i);
        }

        // Backprop through Wq, Wk, Wv
        GradWq += Q.Transpose () * dQ;
        GradWk += K.Transpose () * dK;
        GradWv += V.Transpose () * dV;

        var dInput = dQ * Wq.Transpose () + dK * Wk.Transpose () + dV * Wv.Transpose ();

        return dInput;
    }

    private void AddToMatrix (Matrix fullMatrix, Matrix dHeadMatrix, int headIndex) {
        int offset = headIndex * HeadSize;
        for (int i = 0; i < fullMatrix.Rows; i++)
        for (int j = 0; j < HeadSize; j++)
            fullMatrix.Data[i, offset + j] += dHeadMatrix.Data[i, j];
    }

    private Matrix[] SplitHeads (Matrix x) {
        var heads = new Matrix[NumHeads];
        for (int i = 0; i < NumHeads; i++) {
            heads[i] = new Matrix (x.Rows, HeadSize);
            for (int j = 0; j < x.Rows; j++)
            for (int k = 0; k < HeadSize; k++)
                heads[i].Data[j, k] = x.Data[j, i * HeadSize + k];
        }

        return heads;
    }

    private Matrix ConcatenateHeads (Matrix[] heads) {
        var seq_len = heads[0].Rows;
        var concat = new Matrix (seq_len, EmbeddingSize);
        for (int i = 0; i < NumHeads; i++)
        for (int j = 0; j < seq_len; j++)
        for (int k = 0; k < HeadSize; k++)
            concat.Data[j, i * HeadSize + k] = heads[i].Data[j, k];
        return concat;
    }

    private Matrix Softmax (Matrix x) {
        var result = new Matrix (x.Rows, x.Cols);
        for (int i = 0; i < x.Rows; i++) {
            double max = double.NegativeInfinity;
            for (int j = 0; j < x.Cols; j++)
                if (x.Data[i, j] > max)
                    max = x.Data[i, j];
            double sum = 0.0;
            for (int j = 0; j < x.Cols; j++) {
                result.Data[i, j] = Math.Exp (x.Data[i, j] - max);
                sum += result.Data[i, j];
            }

            for (int j = 0; j < x.Cols; j++)
                result.Data[i, j] /= sum;
        }

        return result;
    }

    private Matrix SoftmaxBackward (Matrix softmaxOutput, Matrix dOutput) {
        var result = new Matrix (softmaxOutput.Rows, softmaxOutput.Cols);
        for (int i = 0; i < softmaxOutput.Rows; i++) {
            for (int j = 0; j < softmaxOutput.Cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < softmaxOutput.Cols; k++) {
                    double delta = (j == k) ? 1.0 : 0.0;
                    sum += dOutput.Data[i, k] * softmaxOutput.Data[i, k] * (delta - softmaxOutput.Data[i, j]);
                }

                result.Data[i, j] = sum;
            }
        }

        return result;
    }
}
