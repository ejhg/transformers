using System;
using System.Collections.Generic;
using System.Linq;

namespace mingpt4;

public class Embedding
{
    public double[,] Weight;
    public double[,] Grad;

    public Embedding(int vocabSize, int embeddingDim)
    {
        Weight = new double[vocabSize, embeddingDim];
        Grad = new double[vocabSize, embeddingDim];
        Random rand = new Random();
        for (int i = 0; i < vocabSize; i++)
            for (int j = 0; j < embeddingDim; j++)
                Weight[i, j] = rand.NextDouble() * 0.01;
    }

    public double[,] Forward(int[] indices)
    {
        int N = indices.Length;
        int D = Weight.GetLength(1);
        double[,] output = new double[N, D];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                output[i, j] = Weight[indices[i], j];
        return output;
    }

    public void Backward(int[] indices, double[,] gradOutput)
    {
        int N = indices.Length;
        int D = Weight.GetLength(1);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                Grad[indices[i], j] += gradOutput[i, j];
    }
}

public class PositionalEncoding
{
    public double[,] Weight;

    public PositionalEncoding(int maxSeqLen, int embeddingDim)
    {
        Weight = new double[maxSeqLen, embeddingDim];
        for (int pos = 0; pos < maxSeqLen; pos++)
            for (int i = 0; i < embeddingDim; i++)
            {
                double angle = pos / Math.Pow(10000, 2 * (i / 2) / (double)embeddingDim);
                Weight[pos, i] = i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
            }
    }

    public double[,] Forward(int seqLen)
    {
        double[,] output = new double[seqLen, Weight.GetLength(1)];
        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < Weight.GetLength(1); j++)
                output[i, j] = Weight[i, j];
        return output;
    }
}

public class MultiHeadSelfAttention
{
    int D, H, Dk;
    public double[,] Wq, Wk, Wv, Wo;
    public double[,] GradWq, GradWk, GradWv, GradWo;

    public MultiHeadSelfAttention(int embeddingDim, int numHeads)
    {
        D = embeddingDim;
        H = numHeads;
        Dk = D / H;
        Wq = InitWeight(D, D);
        Wk = InitWeight(D, D);
        Wv = InitWeight(D, D);
        Wo = InitWeight(D, D);
        GradWq = new double[D, D];
        GradWk = new double[D, D];
        GradWv = new double[D, D];
        GradWo = new double[D, D];
    }

    double[,] InitWeight(int rows, int cols)
    {
        double[,] W = new double[rows, cols];
        Random rand = new Random();
        double std = Math.Sqrt(2.0 / (rows + cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                W[i, j] = rand.NextDouble() * std;
        return W;
    }

    public double[,] Forward(double[,] x, out object cache)
    {
        int N = x.GetLength(0);

        double[,] Q = MatrixOperations.MatMul(x, Wq);
        double[,] K = MatrixOperations.MatMul(x, Wk);
        double[,] V = MatrixOperations.MatMul(x, Wv);

        double[,,] Q_heads = ReshapeToHeads(Q, N);
        double[,,] K_heads = ReshapeToHeads(K, N);
        double[,,] V_heads = ReshapeToHeads(V, N);

        double[,,] AttentionOutputs = new double[N, H, Dk];
        double[,,] AttentionWeights = new double[N, H, N];

        for (int h = 0; h < H; h++)
        {
            double[,] Q_h = GetHead(Q_heads, N, h);
            double[,] K_h = GetHead(K_heads, N, h);
            double[,] V_h = GetHead(V_heads, N, h);

            double[,] Scores = MatrixOperations.MatMul(Q_h, MatrixOperations.Transpose(K_h));
            double scale = 1.0 / Math.Sqrt(Dk);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    Scores[i, j] *= scale;

            double[,] AttnWeights = MatrixOperations.Softmax(Scores);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    AttentionWeights[i, h, j] = AttnWeights[i, j];

            double[,] AttnOutput = MatrixOperations.MatMul(AttnWeights, V_h);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < Dk; j++)
                    AttentionOutputs[i, h, j] = AttnOutput[i, j];
        }

        double[,] ConcatOutput = ConcatenateHeads(AttentionOutputs, N);
        double[,] Output = MatrixOperations.MatMul(ConcatOutput, Wo);

        cache = new
        {
            x,
            Q,
            K,
            V,
            Q_heads,
            K_heads,
            V_heads,
            AttentionWeights,
            ConcatOutput
        };
        return Output;
    }

    public double[,] Backward(double[,] dout, object cache)
    {
        dynamic c = cache;
        int N = c.x.GetLength(0);

        double[,] dConcatOutput = MatrixOperations.MatMul(dout, MatrixOperations.Transpose(Wo));
        double[,] dWo = MatrixOperations.MatMul(MatrixOperations.Transpose(c.ConcatOutput), dout);
        for (int i = 0; i < Wo.GetLength(0); i++)
            for (int j = 0; j < Wo.GetLength(1); j++)
                GradWo[i, j] += dWo[i, j];

        double[,,] dAttentionOutputs = SplitHeads(dConcatOutput, N);

        double[,] dQ = new double[N, D];
        double[,] dK = new double[N, D];
        double[,] dV = new double[N, D];

        for (int h = 0; h < H; h++)
        {
            double[,] dAttnOutput_h = GetHead(dAttentionOutputs, N, h);
            double[,] AttnWeights_h = GetHead(c.AttentionWeights, N, h);
            double[,] V_h = GetHead(c.V_heads, N, h);

            double[,] dAttnWeights = MatrixOperations.MatMul(dAttnOutput_h, MatrixOperations.Transpose(V_h));
            double[,] dV_h = MatrixOperations.MatMul(MatrixOperations.Transpose(AttnWeights_h), dAttnOutput_h);

            double[,] dScores = MatrixOperations.SoftmaxBackward(dAttnWeights, AttnWeights_h);

            double[,] Q_h = GetHead(c.Q_heads, N, h);
            double[,] K_h = GetHead(c.K_heads, N, h);

            double scale = 1.0 / Math.Sqrt(Dk);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    dScores[i, j] *= scale;

            double[,] dQ_h = MatrixOperations.MatMul(dScores, K_h);
            double[,] dK_h = MatrixOperations.MatMul(MatrixOperations.Transpose(dScores), Q_h);

            AddToHead(dQ, dQ_h, N, h);
            AddToHead(dK, dK_h, N, h);
            AddToHead(dV, dV_h, N, h);
        }

        double[,] dWq = MatrixOperations.MatMul(MatrixOperations.Transpose(c.x), dQ);
        double[,] dWk = MatrixOperations.MatMul(MatrixOperations.Transpose(c.x), dK);
        double[,] dWv = MatrixOperations.MatMul(MatrixOperations.Transpose(c.x), dV);

        double[,] dx_q = MatrixOperations.MatMul(dQ, MatrixOperations.Transpose(Wq));
        double[,] dx_k = MatrixOperations.MatMul(dK, MatrixOperations.Transpose(Wk));
        double[,] dx_v = MatrixOperations.MatMul(dV, MatrixOperations.Transpose(Wv));

        double[,] dx = MatrixOperations.AddMat(MatrixOperations.AddMat(dx_q, dx_k), dx_v);

        for (int i = 0; i < Wq.GetLength(0); i++)
            for (int j = 0; j < Wq.GetLength(1); j++)
                GradWq[i, j] += dWq[i, j];

        for (int i = 0; i < Wk.GetLength(0); i++)
            for (int j = 0; j < Wk.GetLength(1); j++)
                GradWk[i, j] += dWk[i, j];

        for (int i = 0; i < Wv.GetLength(0); i++)
            for (int j = 0; j < Wv.GetLength(1); j++)
                GradWv[i, j] += dWv[i, j];

        return dx;
    }

    double[,,] ReshapeToHeads(double[,] x, int N)
    {
        double[,,] x_heads = new double[N, H, Dk];
        for (int i = 0; i < N; i++)
            for (int h = 0; h < H; h++)
                for (int k = 0; k < Dk; k++)
                    x_heads[i, h, k] = x[i, h * Dk + k];
        return x_heads;
    }

    double[,,] SplitHeads(double[,] x, int N)
    {
        double[,,] x_heads = new double[N, H, Dk];
        for (int i = 0; i < N; i++)
            for (int h = 0; h < H; h++)
                for (int k = 0; k < Dk; k++)
                    x_heads[i, h, k] = x[i, h * Dk + k];
        return x_heads;
    }

    double[,] GetHead(double[,,] x_heads, int N, int h)
    {
        double[,] x_h = new double[N, Dk];
        for (int i = 0; i < N; i++)
            for (int k = 0; k < Dk; k++)
                x_h[i, k] = x_heads[i, h, k];
        return x_h;
    }

    void AddToHead(double[,] x, double[,] x_h, int N, int h)
    {
        for (int i = 0; i < N; i++)
            for (int k = 0; k < Dk; k++)
                x[i, h * Dk + k] += x_h[i, k];
    }

    double[,] ConcatenateHeads(double[,,] AttentionOutputs, int N)
    {
        double[,] ConcatOutput = new double[N, D];
        for (int i = 0; i < N; i++)
            for (int h = 0; h < H; h++)
                for (int k = 0; k < Dk; k++)
                    ConcatOutput[i, h * Dk + k] = AttentionOutputs[i, h, k];
        return ConcatOutput;
    }
}

public class LayerNorm
{
    public int D;
    public double[] Gamma, Beta;
    public double[] GradGamma, GradBeta;

    public LayerNorm(int embeddingDim)
    {
        D = embeddingDim;
        Gamma = Enumerable.Repeat(1.0, D).ToArray();
        Beta = new double[D];
        GradGamma = new double[D];
        GradBeta = new double[D];
    }

    public double[,] Forward(double[,] x, out object cache)
    {
        int N = x.GetLength(0);
        double[,] y = new double[N, D];
        double[] mean = new double[N];
        double[] var = new double[N];
        double epsilon = 1e-5;

        for (int i = 0; i < N; i++)
        {
            mean[i] = 0.0;
            for (int j = 0; j < D; j++)
                mean[i] += x[i, j];
            mean[i] /= D;
        }

        for (int i = 0; i < N; i++)
        {
            var[i] = 0.0;
            for (int j = 0; j < D; j++)
                var[i] += Math.Pow(x[i, j] - mean[i], 2);
            var[i] /= D;
        }

        double[] std = var.Select(v => Math.Sqrt(v + epsilon)).ToArray();

        double[,] x_hat = new double[N, D];

        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                x_hat[i, j] = (x[i, j] - mean[i]) / std[i];

        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                y[i, j] = x_hat[i, j] * Gamma[j] + Beta[j];

        cache = new { x, mean, var, std, x_hat };
        return y;
    }

    public double[,] Backward(double[,] dout, object cache)
    {
        dynamic c = cache;
        int N = c.x.GetLength(0);
        int D = c.x.GetLength(1);
        double epsilon = 1e-5;

        double[,] dx = new double[N, D];

        // Gradients with respect to Gamma and Beta
        for (int j = 0; j < D; j++)
        {
            GradGamma[j] = 0.0;
            GradBeta[j] = 0.0;
            for (int i = 0; i < N; i++)
            {
                GradGamma[j] += dout[i, j] * c.x_hat[i, j];
                GradBeta[j] += dout[i, j];
            }
        }

        // Backprop through normalization
        for (int i = 0; i < N; i++)
        {
            double invStd = 1.0 / c.std[i];
            double dmean = 0.0;
            double dvar = 0.0;

            // Compute dx_hat
            double[] dx_hat = new double[D];
            for (int j = 0; j < D; j++)
                dx_hat[j] = dout[i, j] * Gamma[j];

            // Compute dvar
            for (int j = 0; j < D; j++)
                dvar += dx_hat[j] * (c.x[i, j] - c.mean[i]) * -0.5 * Math.Pow(c.std[i], -3);

            // Compute dmean
            for (int j = 0; j < D; j++)
                dmean += dx_hat[j] * -invStd;

            // Compute dx
            for (int j = 0; j < D; j++)
                dx[i, j] = dx_hat[j] * invStd + dvar * 2 * (c.x[i, j] - c.mean[i]) / D + dmean / D;
        }

        return dx;
    }
}

public class FeedForward
{
    int D, H;
    public double[,] W1, W2;
    public double[] b1, b2;
    public double[,] GradW1, GradW2;
    public double[] Gradb1, Gradb2;

    public FeedForward(int embeddingDim, int hiddenDim)
    {
        D = embeddingDim;
        H = hiddenDim;
        W1 = InitWeight(D, H);
        b1 = new double[H];
        W2 = InitWeight(H, D);
        b2 = new double[D];
        GradW1 = new double[D, H];
        Gradb1 = new double[H];
        GradW2 = new double[H, D];
        Gradb2 = new double[D];
    }

    double[,] InitWeight(int rows, int cols)
    {
        double[,] W = new double[rows, cols];
        Random rand = new Random();
        double std = Math.Sqrt(2.0 / (rows + cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                W[i, j] = rand.NextDouble() * std;
        return W;
    }

    public double[,] Forward(double[,] x, out object cache)
    {
        int N = x.GetLength(0);
        double[,] h = MatrixOperations.MatMul(x, W1);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < H; j++)
                h[i, j] += b1[j];

        double[,] h_relu = new double[N, H];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < H; j++)
                h_relu[i, j] = Math.Max(0.0, h[i, j]);

        double[,] outp = MatrixOperations.MatMul(h_relu, W2);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                outp[i, j] += b2[j];

        cache = new { x, h, h_relu };
        return outp;
    }

    public double[,] Backward(double[,] dout, object cache)
    {
        dynamic c = cache;
        int N = dout.GetLength(0);

        // Backprop through W2 and b2
        double[,] dW2 = MatrixOperations.MatMul(MatrixOperations.Transpose(c.h_relu), dout);
        double[] db2 = new double[D];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                db2[j] += dout[i, j];

        double[,] dh_relu = MatrixOperations.MatMul(dout, MatrixOperations.Transpose(W2));

        // Backprop through ReLU
        double[,] dh = new double[N, H];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < H; j++)
                dh[i, j] = c.h[i, j] > 0 ? dh_relu[i, j] : 0.0;

        // Backprop through W1 and b1
        double[,] dW1 = MatrixOperations.MatMul(MatrixOperations.Transpose(c.x), dh);
        double[] db1 = new double[H];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < H; j++)
                db1[j] += dh[i, j];

        double[,] dx = MatrixOperations.MatMul(dh, MatrixOperations.Transpose(W1));

        // Accumulate gradients
        for (int i = 0; i < W1.GetLength(0); i++)
            for (int j = 0; j < W1.GetLength(1); j++)
                GradW1[i, j] += dW1[i, j];

        for (int j = 0; j < b1.Length; j++)
            Gradb1[j] += db1[j];

        for (int i = 0; i < W2.GetLength(0); i++)
            for (int j = 0; j < W2.GetLength(1); j++)
                GradW2[i, j] += dW2[i, j];

        for (int j = 0; j < b2.Length; j++)
            Gradb2[j] += db2[j];

        return dx;
    }
}

public class TransformerBlock
{
    public MultiHeadSelfAttention SelfAttention;
    public LayerNorm LayerNorm1, LayerNorm2;
    public FeedForward FeedForward;

    public TransformerBlock(int embeddingDim, int numHeads, int hiddenDim)
    {
        SelfAttention = new MultiHeadSelfAttention(embeddingDim, numHeads);
        LayerNorm1 = new LayerNorm(embeddingDim);
        FeedForward = new FeedForward(embeddingDim, hiddenDim);
        LayerNorm2 = new LayerNorm(embeddingDim);
    }

    public double[,] Forward(double[,] x, out object cache)
    {
        object sa_cache, ln1_cache, ff_cache, ln2_cache;

        double[,] sa_out = SelfAttention.Forward(x, out sa_cache);
        double[,] sa_residual = MatrixOperations.AddMat(x, sa_out);
        double[,] ln_out1 = LayerNorm1.Forward(sa_residual, out ln1_cache);

        double[,] ff_out = FeedForward.Forward(ln_out1, out ff_cache);
        double[,] ff_residual = MatrixOperations.AddMat(ln_out1, ff_out);
        double[,] ln_out2 = LayerNorm2.Forward(ff_residual, out ln2_cache);

        cache = new { x, sa_cache, ln1_cache, ff_cache, ln2_cache, sa_residual, ln_out1, ff_residual };
        return ln_out2;
    }

    public double[,] Backward(double[,] dout, object cache)
    {
        dynamic c = cache;

        double[,] dln_out2 = dout;
        double[,] dff_residual = LayerNorm2.Backward(dln_out2, c.ln2_cache);

        double[,] dln_out1 = new double[dff_residual.GetLength(0), dff_residual.GetLength(1)];
        double[,] dff_out = new double[dff_residual.GetLength(0), dff_residual.GetLength(1)];

        for (int i = 0; i < dff_residual.GetLength(0); i++)
            for (int j = 0; j < dff_residual.GetLength(1); j++)
            {
                dln_out1[i, j] += dff_residual[i, j];
                dff_out[i, j] = dff_residual[i, j];
            }

        double[,] dff_in = FeedForward.Backward(dff_out, c.ff_cache);

        for (int i = 0; i < dln_out1.GetLength(0); i++)
            for (int j = 0; j < dln_out1.GetLength(1); j++)
                dln_out1[i, j] += dff_in[i, j];

        double[,] dsa_residual = LayerNorm1.Backward(dln_out1, c.ln1_cache);

        double[,] dx = new double[dsa_residual.GetLength(0), dsa_residual.GetLength(1)];
        double[,] dsa_out = new double[dsa_residual.GetLength(0), dsa_residual.GetLength(1)];

        for (int i = 0; i < dsa_residual.GetLength(0); i++)
            for (int j = 0; j < dsa_residual.GetLength(1); j++)
            {
                dx[i, j] += dsa_residual[i, j];
                dsa_out[i, j] = dsa_residual[i, j];
            }

        double[,] dsa_in = SelfAttention.Backward(dsa_out, c.sa_cache);

        for (int i = 0; i < dx.GetLength(0); i++)
            for (int j = 0; j < dx.GetLength(1); j++)
                dx[i, j] += dsa_in[i, j];

        return dx;
    }
}

public class GPT
{
    public Embedding TokenEmbedding;
    public PositionalEncoding PosEncoding;
    public List<TransformerBlock> Blocks;
    int VocabSize, D;

    public GPT(int vocabSize, int embeddingDim, int numHeads, int numLayers, int maxSeqLen, int hiddenDim)
    {
        VocabSize = vocabSize;
        D = embeddingDim;
        TokenEmbedding = new Embedding(vocabSize, embeddingDim);
        PosEncoding = new PositionalEncoding(maxSeqLen, embeddingDim);
        Blocks = new List<TransformerBlock>();
        for (int i = 0; i < numLayers; i++)
            Blocks.Add(new TransformerBlock(embeddingDim, numHeads, hiddenDim));
    }

    public double[,] Forward(int[] inputIds, out object cache)
    {
        double[,] x = TokenEmbedding.Forward(inputIds);
        double[,] pos = PosEncoding.Forward(inputIds.Length);
        x = MatrixOperations.AddMat(x, pos);

        List<object> blockCaches = new List<object>();

        foreach (var block in Blocks)
        {
            x = block.Forward(x, out object blockCache);
            blockCaches.Add(blockCache);
        }

        double[,] logits = MatrixOperations.MatMul(x, MatrixOperations.Transpose(TokenEmbedding.Weight));

        cache = new { inputIds, x, blockCaches };
        return logits;
    }

    public void Backward(double[,] dlogits, object cache)
    {
        dynamic c = cache;

        double[,] dx = MatrixOperations.MatMul(dlogits, TokenEmbedding.Weight);

        for (int i = Blocks.Count - 1; i >= 0; i--)
        {
            dx = Blocks[i].Backward(dx, c.blockCaches[i]);
        }

        TokenEmbedding.Backward(c.inputIds, dx);
    }

    public List<object> GetParameters()
    {
        List<object> parameters = new List<object>();
        parameters.Add(TokenEmbedding);
        foreach (var block in Blocks)
        {
            parameters.Add(block.SelfAttention);
            parameters.Add(block.LayerNorm1);
            parameters.Add(block.FeedForward);
            parameters.Add(block.LayerNorm2);
        }
        return parameters;
    }
}

public class SGD
{
    double LearningRate;

    public SGD(double learningRate)
    {
        LearningRate = learningRate;
    }

    public void Step(List<object> parameters)
    {
        foreach (var param in parameters)
        {
            if (param is Embedding e)
            {
                for (int i = 0; i < e.Weight.GetLength(0); i++)
                    for (int j = 0; j < e.Weight.GetLength(1); j++)
                    {
                        e.Weight[i, j] -= LearningRate * e.Grad[i, j];
                        e.Grad[i, j] = 0.0; // Reset gradient after update
                    }
            }
            else if (param is MultiHeadSelfAttention sa)
            {
                UpdateWeights(sa.Wq, sa.GradWq);
                UpdateWeights(sa.Wk, sa.GradWk);
                UpdateWeights(sa.Wv, sa.GradWv);
                UpdateWeights(sa.Wo, sa.GradWo);
            }
            else if (param is LayerNorm ln)
            {
                for (int j = 0; j < ln.D; j++)
                {
                    ln.Gamma[j] -= LearningRate * ln.GradGamma[j];
                    ln.Beta[j] -= LearningRate * ln.GradBeta[j];
                    ln.GradGamma[j] = 0.0;
                    ln.GradBeta[j] = 0.0;
                }
            }
            else if (param is FeedForward ff)
            {
                UpdateWeights(ff.W1, ff.GradW1);
                UpdateWeights(ff.W2, ff.GradW2);
                UpdateBias(ff.b1, ff.Gradb1);
                UpdateBias(ff.b2, ff.Gradb2);
            }
        }
    }

    void UpdateWeights(double[,] W, double[,] dW)
    {
        for (int i = 0; i < W.GetLength(0); i++)
            for (int j = 0; j < W.GetLength(1); j++)
            {
                W[i, j] -= LearningRate * dW[i, j];
                dW[i, j] = 0.0;
            }
    }

    void UpdateBias(double[] b, double[] db)
    {
        for (int j = 0; j < b.Length; j++)
        {
            b[j] -= LearningRate * db[j];
            db[j] = 0.0;
        }
    }
}

public class Trainer
{
    public void Train(List<int[]> data, GPT model, int epochs, int batchSize, double learningRate)
    {
        SGD optimizer = new SGD(learningRate);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0.0;
            int numBatches = 0;

            foreach (var batch in GetBatches(data, batchSize))
            {
                numBatches++;
                List<double[,]> logitsList = new List<double[,]>();
                List<int[]> targetsList = new List<int[]>();
                List<object> caches = new List<object>();

                for (int idx = 0; idx < batch.Count; idx++)
                {
                    int[] inputIds = batch[idx];
                    int[] inputs = inputIds.Take(inputIds.Length - 1).ToArray();
                    int[] targets = inputIds.Skip(1).ToArray();

                    double[,] logits = model.Forward(inputs, out object cache);
                    logitsList.Add(logits);
                    targetsList.Add(targets);
                    caches.Add(cache);
                }

                double loss = ComputeLoss(logitsList, targetsList, out List<double[,]> dlogitsList);
                totalLoss += loss;

                // Backward pass
                for (int idx = 0; idx < batch.Count; idx++)
                {
                    model.Backward(dlogitsList[idx], caches[idx]);
                }

                // Update parameters
                optimizer.Step(model.GetParameters());
            }

            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {totalLoss / numBatches}");
        }
    }

    List<List<int[]>> GetBatches(List<int[]> data, int batchSize)
    {
        List<List<int[]>> batches = new List<List<int[]>>();
        for (int i = 0; i < data.Count; i += batchSize)
            batches.Add(data.Skip(i).Take(batchSize).ToList());
        return batches;
    }

    double ComputeLoss(List<double[,]> logitsList, List<int[]> targetsList, out List<double[,]> dlogitsList)
    {
        double loss = 0.0;
        dlogitsList = new List<double[,]>();

        for (int idx = 0; idx < logitsList.Count; idx++)
        {
            double[,] logits = logitsList[idx];
            int[] targets = targetsList[idx];
            int N = logits.GetLength(0);
            int V = logits.GetLength(1);

            double[,] probs = MatrixOperations.Softmax(logits);
            double[,] dlogits = new double[N, V];

            for (int t = 0; t < N; t++)
            {
                int target = targets[t];
                double[] prob = new double[V];
                for (int v = 0; v < V; v++)
                    prob[v] = probs[t, v];

                loss -= Math.Log(prob[target]);

                for (int v = 0; v < V; v++)
                {
                    dlogits[t, v] = prob[v];
                }
                dlogits[t, target] -= 1.0;
            }

            dlogitsList.Add(dlogits);
        }

        loss /= logitsList.Count;
        return loss;
    }
}

public static class MatrixOperations
{
    public static double[,] MatMul(double[,] A, double[,] B)
    {
        int m = A.GetLength(0);
        int n = A.GetLength(1);
        int p = B.GetLength(1);

        if (n != B.GetLength(0))
            throw new Exception("Matrix dimensions are not compatible for multiplication.");

        double[,] C = new double[m, p];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < p; j++)
            {
                C[i, j] = 0.0;
                for (int k = 0; k < n; k++)
                    C[i, j] += A[i, k] * B[k, j];
            }
        return C;
    }

    public static double[,] Transpose(double[,] A)
    {
        int m = A.GetLength(0);
        int n = A.GetLength(1);
        double[,] At = new double[n, m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                At[j, i] = A[i, j];
        return At;
    }

    public static double[,] Softmax(double[,] logits)
    {
        int m = logits.GetLength(0);
        int n = logits.GetLength(1);
        double[,] probs = new double[m, n];
        for (int i = 0; i < m; i++)
        {
            double maxLogit = double.MinValue;
            for (int j = 0; j < n; j++)
                if (logits[i, j] > maxLogit)
                    maxLogit = logits[i, j];

            double sumExp = 0.0;
            for (int j = 0; j < n; j++)
            {
                probs[i, j] = Math.Exp(logits[i, j] - maxLogit);
                sumExp += probs[i, j];
            }

            for (int j = 0; j < n; j++)
                probs[i, j] /= sumExp;
        }
        return probs;
    }

    public static double[,] SoftmaxBackward(double[,] dprobs, double[,] probs)
    {
        int m = probs.GetLength(0);
        int n = probs.GetLength(1);
        double[,] dlogits = new double[m, n];

        for (int i = 0; i < m; i++)
        {
            double sum_dprobs = 0.0;
            for (int j = 0; j < n; j++)
                sum_dprobs += dprobs[i, j];

            for (int j = 0; j < n; j++)
                dlogits[i, j] = probs[i, j] * (dprobs[i, j] - sum_dprobs * probs[i, j]);
        }
        return dlogits;
    }

    public static double[,] AddMat(double[,] A, double[,] B)
    {
        int m = A.GetLength(0);
        int n = A.GetLength(1);
        double[,] C = new double[m, n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i, j] = A[i, j] + B[i, j];
        return C;
    }
}

// Example usage:

public class Program
{
    public static void Main()
    {
        int vocabSize = 1000;
        int embeddingDim = 32;
        int numHeads = 4;
        int numLayers = 2;
        int maxSeqLen = 20;
        int hiddenDim = 64;

        GPT model = new GPT(vocabSize, embeddingDim, numHeads, numLayers, maxSeqLen, hiddenDim);

        // Dummy data
        List<int[]> data = new List<int[]>();
        Random rand = new Random();

        // Generate dummy data: list of sequences of token IDs
        for (int i = 0; i < 100; i++)
        {
            int seqLen = rand.Next(5, maxSeqLen);
            int[] sequence = new int[seqLen];
            for (int j = 0; j < seqLen; j++)
                sequence[j] = rand.Next(0, vocabSize);
            data.Add(sequence);
        }

        Trainer trainer = new Trainer();
        trainer.Train(data, model, epochs: 5, batchSize: 16, learningRate: 0.001);

        Console.WriteLine("Training completed.");
    }
}
