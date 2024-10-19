using System;
using System.Collections.Generic;
using System.Linq;

namespace llama;

// Utility functions for vector and matrix operations
public static class MathOps
{
    public static double[] Add (double[] a, double[] b) {
        double[] result = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] + b[i];
        return result;
    }

    public static double Dot (double[] a, double[] b) {
        double result = 0;
        for (int i = 0; i < a.Length; i++)
            result += a[i] * b[i];
        return result;
    }

    public static double[] MatrixVectorProduct (double[,] matrix, double[] vector) {
        int rows = matrix.GetLength (0);
        int cols = matrix.GetLength (1);
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            result[i] = 0;
            for (int j = 0; j < cols; j++)
                result[i] += matrix[i, j] * vector[j];
        }

        return result;
    }

    public static double[] Softmax (double[] logits) {
        double maxLogit = logits.Max ();
        double[] expLogits = logits.Select (x => Math.Exp (x - maxLogit)).ToArray ();
        double sumExp = expLogits.Sum ();
        return expLogits.Select (x => x / sumExp).ToArray ();
    }

    public static double[] SoftmaxGradient (double[] softmax, double[] dLoss_dSoftmax) {
        int n = softmax.Length;
        double[] dScores = new double[n];

        double dot = 0.0;
        for (int i = 0; i < n; i++)
            dot += softmax[i] * dLoss_dSoftmax[i];

        for (int i = 0; i < n; i++)
            dScores[i] = softmax[i] * (dLoss_dSoftmax[i] - dot);

        return dScores;
    }
}

// Embedding layer with backward pass
public class Embedding
{
    public double[,] Weights;

    public double[,] Gradients;

    // For Adam optimizer
    public double[,] mWeights;
    public double[,] vWeights;

    private Random rand;

    public Embedding (int vocabSize, int embedSize, Random random) {
        rand = random;
        Weights = new double[vocabSize, embedSize];
        Gradients = new double[vocabSize, embedSize];
        mWeights = new double[vocabSize, embedSize];
        vWeights = new double[vocabSize, embedSize];

        double limit = Math.Sqrt (6.0 / (vocabSize + embedSize)); // Xavier Initialization
        for (int i = 0; i < vocabSize; i++)
        for (int j = 0; j < embedSize; j++)
            Weights[i, j] = rand.NextDouble () * 2 * limit - limit;
    }

    public double[] Forward (int token) {
        double[] embedding = new double[Weights.GetLength (1)];
        for (int i = 0; i < Weights.GetLength (1); i++)
            embedding[i] = Weights[token, i];
        return embedding;
    }

    public void Backward (int token, double[] grad) {
        for (int i = 0; i < Weights.GetLength (1); i++)
            Gradients[token, i] += grad[i];
    }
}

// Layer Normalization with backward pass
public class LayerNorm
{
    public double[] Gamma;
    public double[] Beta;
    public int Size;

    public double[] dGamma;
    public double[] dBeta;

    // For Adam optimizer
    public double[] mGamma;
    public double[] vGamma;
    public double[] mBeta;
    public double[] vBeta;

    public LayerNorm (int size) {
        Size = size;
        Gamma = new double[size];
        Beta = new double[size];
        dGamma = new double[size];
        dBeta = new double[size];

        mGamma = new double[size];
        vGamma = new double[size];
        mBeta = new double[size];
        vBeta = new double[size];

        for (int i = 0; i < size; i++) {
            Gamma[i] = 1.0;
            Beta[i] = 0.0;
        }
    }

    public (double[] output, LayerNormCache cache) Forward (double[] x) {
        double mean = x.Average ();
        double variance = x.Select (val => Math.Pow (val - mean, 2)).Average ();
        double[] normalized = x.Select (val => (val - mean) / Math.Sqrt (variance + 1e-5)).ToArray ();
        double[] output = new double[Size];
        for (int i = 0; i < Size; i++)
            output[i] = Gamma[i] * normalized[i] + Beta[i];

        var cache = new LayerNormCache {
            x_input = x,
            mean = mean,
            variance = variance,
            normalized = normalized
        };

        return (output, cache);
    }

    public double[] Backward (double[] gradOutput, LayerNormCache cache) {
        double[] dxhat = new double[Size];
        for (int i = 0; i < Size; i++) {
            dGamma[i] += gradOutput[i] * cache.normalized[i];
            dBeta[i] += gradOutput[i];
            dxhat[i] = gradOutput[i] * Gamma[i];
        }

        double stdInv = 1.0 / Math.Sqrt (cache.variance + 1e-5);
        double[] dx = new double[Size];
        double dvar = -0.5 * stdInv * stdInv * stdInv * dxhat.Select ((dxh, i) => (cache.x_input[i] - cache.mean) * dxh).Sum ();
        double dmean = -stdInv * dxhat.Sum () + dvar * (-2.0 / Size) * (cache.x_input.Sum () - Size * cache.mean);
        for (int i = 0; i < Size; i++)
            dx[i] = stdInv * dxhat[i] + dvar * 2.0 * (cache.x_input[i] - cache.mean) / Size + dmean / Size;
        return dx;
    }
}

public class LayerNormCache
{
    public double[] x_input;
    public double mean;
    public double variance;
    public double[] normalized;
}

// Self-Attention mechanism with RoPE and backward pass
public class SelfAttention
{
    public double[,] Wq, Wk, Wv, Wo;

    public double[,] dWq, dWk, dWv, dWo;

    // For Adam optimizer
    public double[,] mWq, vWq;
    public double[,] mWk, vWk;
    public double[,] mWv, vWv;
    public double[,] mWo, vWo;

    public int EmbedSize, HeadSize;

    private Random rand;

    public SelfAttention (int embedSize, int headSize, Random random) {
        rand = random;
        EmbedSize = embedSize;
        HeadSize = headSize;
        Wq = InitializeMatrix (HeadSize, EmbedSize);
        Wk = InitializeMatrix (HeadSize, EmbedSize);
        Wv = InitializeMatrix (HeadSize, EmbedSize);
        Wo = InitializeMatrix (EmbedSize, HeadSize);

        dWq = new double[HeadSize, EmbedSize];
        dWk = new double[HeadSize, EmbedSize];
        dWv = new double[HeadSize, EmbedSize];
        dWo = new double[EmbedSize, HeadSize];

        mWq = new double[HeadSize, EmbedSize];
        vWq = new double[HeadSize, EmbedSize];
        mWk = new double[HeadSize, EmbedSize];
        vWk = new double[HeadSize, EmbedSize];
        mWv = new double[HeadSize, EmbedSize];
        vWv = new double[HeadSize, EmbedSize];
        mWo = new double[EmbedSize, HeadSize];
        vWo = new double[EmbedSize, HeadSize];
    }

    private double[,] InitializeMatrix (int rows, int cols) {
        double[,] matrix = new double[rows, cols];
        double limit = Math.Sqrt (6.0 / (rows + cols)); // Xavier Initialization
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble () * 2 * limit - limit;
        return matrix;
    }

    public (List<double[]> outputs, SelfAttentionCache cache) Forward (List<double[]> inputs, int startPosition) {
        int seqLen = inputs.Count;

        var cache = new SelfAttentionCache {
            Inputs = inputs,
            Qs = new List<double[]> (),
            Ks = new List<double[]> (),
            Vs = new List<double[]> (),
            Zs = new List<double[]> (),
            Softmaxes = new List<double[]> (),
            StartPosition = startPosition
        };

        // Compute Qs, Ks, Vs
        for (int i = 0; i < seqLen; i++) {
            var x = inputs[i];
            var q = MathOps.MatrixVectorProduct (Wq, x);
            var k = MathOps.MatrixVectorProduct (Wk, x);
            var v = MathOps.MatrixVectorProduct (Wv, x);

            // Apply RoPE embeddings to q and k
            q = ApplyRoPE (q, i + startPosition);
            k = ApplyRoPE (k, i + startPosition);

            cache.Qs.Add (q);
            cache.Ks.Add (k);
            cache.Vs.Add (v);
        }

        // Compute attention outputs
        for (int i = 0; i < seqLen; i++) {
            // Compute attention scores up to position i
            double[] scores = new double[i + 1];
            for (int j = 0; j <= i; j++)
                scores[j] = MathOps.Dot (cache.Qs[i], cache.Ks[j]) / Math.Sqrt (HeadSize);

            // Apply softmax to get attention weights
            double[] softmax = MathOps.Softmax (scores);
            cache.Softmaxes.Add (softmax);

            // Compute weighted sum of Vs
            double[] z = new double[HeadSize];
            for (int j = 0; j <= i; j++)
            for (int k = 0; k < HeadSize; k++)
                z[k] += softmax[j] * cache.Vs[j][k];
            cache.Zs.Add (z);
        }

        // Output projection
        List<double[]> outputs = new List<double[]> ();
        for (int i = 0; i < seqLen; i++) {
            double[] output = MathOps.MatrixVectorProduct (Wo, cache.Zs[i]);
            outputs.Add (output);
        }

        return (outputs, cache);
    }

    public List<double[]> Backward (List<double[]> gradOutputs, SelfAttentionCache cache) {
        int seqLen = cache.Inputs.Count;
        List<double[]> dInputs = new List<double[]> ();
        for (int i = 0; i < seqLen; i++)
            dInputs.Add (new double[EmbedSize]);

        // Initialize gradients for Qs, Ks, Vs
        double[][] dQs = new double[seqLen][];
        double[][] dKs = new double[seqLen][];
        double[][] dVs = new double[seqLen][];
        for (int i = 0; i < seqLen; i++) {
            dQs[i] = new double[HeadSize];
            dKs[i] = new double[HeadSize];
            dVs[i] = new double[HeadSize];
        }

        // Gradients w.r.t. output projection Wo
        for (int i = 0; i < seqLen; i++) {
            // Gradient w.r.t. z_i
            double[] dZ = new double[HeadSize];
            for (int k = 0; k < EmbedSize; k++) {
                for (int h = 0; h < HeadSize; h++) {
                    dZ[h] += Wo[k, h] * gradOutputs[i][k];
                    dWo[k, h] += gradOutputs[i][k] * cache.Zs[i][h];
                }
            }

            // Backpropagate through attention output z_i = sum_j alpha_{ij} * V_j
            double[] dAlpha = new double[i + 1];
            for (int j = 0; j <= i; j++) {
                for (int h = 0; h < HeadSize; h++) {
                    dVs[j][h] += cache.Softmaxes[i][j] * dZ[h];
                }

                dAlpha[j] = MathOps.Dot (cache.Vs[j], dZ);
            }

            // Backpropagate through softmax
            double[] dScores = MathOps.SoftmaxGradient (cache.Softmaxes[i], dAlpha);

            // Backpropagate through attention scores s_{ij} = Q_i ⋅ K_j / sqrt(HeadSize)
            double scale = 1.0 / Math.Sqrt (HeadSize);
            for (int j = 0; j <= i; j++) {
                for (int h = 0; h < HeadSize; h++) {
                    dQs[i][h] += dScores[j] * cache.Ks[j][h] * scale;
                    dKs[j][h] += dScores[j] * cache.Qs[i][h] * scale;
                }
            }
        }

        // Backpropagate through RoPE embeddings
        for (int i = 0; i < seqLen; i++) {
            int position = i + cache.StartPosition;

            dQs[i] = BackwardRoPE (dQs[i], cache.Qs[i], position);
            dKs[i] = BackwardRoPE (dKs[i], cache.Ks[i], position);
        }

        // Backpropagate through linear layers Wq, Wk, Wv
        for (int i = 0; i < seqLen; i++) {
            // Gradients w.r.t. Wq and input x_i
            for (int h = 0; h < HeadSize; h++) {
                for (int j = 0; j < EmbedSize; j++) {
                    dWq[h, j] += dQs[i][h] * cache.Inputs[i][j];
                    dInputs[i][j] += Wq[h, j] * dQs[i][h];
                }
            }

            // Gradients w.r.t. Wk and input x_i
            for (int h = 0; h < HeadSize; h++) {
                for (int j = 0; j < EmbedSize; j++) {
                    dWk[h, j] += dKs[i][h] * cache.Inputs[i][j];
                    dInputs[i][j] += Wk[h, j] * dKs[i][h];
                }
            }

            // Gradients w.r.t. Wv and input x_i
            for (int h = 0; h < HeadSize; h++) {
                for (int j = 0; j < EmbedSize; j++) {
                    dWv[h, j] += dVs[i][h] * cache.Inputs[i][j];
                    dInputs[i][j] += Wv[h, j] * dVs[i][h];
                }
            }
        }

        return dInputs;
    }

    private double[] ApplyRoPE (double[] x, int position) {
        double[] result = new double[x.Length];
        int halfDim = x.Length / 2;
        double theta = 10000;

        for (int i = 0; i < halfDim; i++) {
            double angle = position / Math.Pow (theta, (double)i / halfDim);
            double cos = Math.Cos (angle);
            double sin = Math.Sin (angle);

            result[2 * i] = x[2 * i] * cos - x[2 * i + 1] * sin;
            result[2 * i + 1] = x[2 * i] * sin + x[2 * i + 1] * cos;
        }

        return result;
    }

    private double[] BackwardRoPE (double[] grad, double[] x, int position) {
        double[] dx = new double[x.Length];
        int halfDim = x.Length / 2;
        double theta = 10000;

        for (int i = 0; i < halfDim; i++) {
            double angle = position / Math.Pow (theta, (double)i / halfDim);
            double cos = Math.Cos (angle);
            double sin = Math.Sin (angle);

            double dReal = grad[2 * i];
            double dImag = grad[2 * i + 1];

            dx[2 * i] = dReal * cos + dImag * sin;
            dx[2 * i + 1] = -dReal * sin + dImag * cos;
        }

        return dx;
    }
}

public class SelfAttentionCache
{
    public List<double[]> Inputs;
    public List<double[]> Qs;
    public List<double[]> Ks;
    public List<double[]> Vs;
    public List<double[]> Zs;
    public List<double[]> Softmaxes;
    public int StartPosition;
}

// Feed-Forward Network with backward pass
public class FeedForward
{
    public double[,] W1, W2;
    public double[] B1, B2;
    public int EmbedSize, HiddenSize;

    public double[,] dW1, dW2;
    public double[] dB1, dB2;

    // For Adam optimizer
    public double[,] mW1, vW1;
    public double[,] mW2, vW2;
    public double[] mB1, vB1;
    public double[] mB2, vB2;

    private Random rand;

    public FeedForward (int embedSize, int hiddenSize, Random random) {
        rand = random;
        EmbedSize = embedSize;
        HiddenSize = hiddenSize;
        W1 = InitializeMatrix (HiddenSize, EmbedSize);
        B1 = new double[HiddenSize];
        W2 = InitializeMatrix (EmbedSize, HiddenSize);
        B2 = new double[EmbedSize];

        dW1 = new double[HiddenSize, EmbedSize];
        dB1 = new double[HiddenSize];
        dW2 = new double[EmbedSize, HiddenSize];
        dB2 = new double[EmbedSize];

        mW1 = new double[HiddenSize, EmbedSize];
        vW1 = new double[HiddenSize, EmbedSize];
        mW2 = new double[EmbedSize, HiddenSize];
        vW2 = new double[EmbedSize, HiddenSize];
        mB1 = new double[HiddenSize];
        vB1 = new double[HiddenSize];
        mB2 = new double[EmbedSize];
        vB2 = new double[EmbedSize];
    }

    private double[,] InitializeMatrix (int rows, int cols) {
        double[,] matrix = new double[rows, cols];
        double limit = Math.Sqrt (6.0 / (rows + cols)); // Xavier Initialization
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble () * 2 * limit - limit;
        return matrix;
    }

    public (double[] output, FeedForwardCache cache) Forward (double[] x) {
        // First layer
        double[] h_linear = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++) {
            h_linear[i] = B1[i];
            for (int j = 0; j < EmbedSize; j++)
                h_linear[i] += W1[i, j] * x[j];
        }

        // ReLU activation
        double[] h_relu = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++)
            h_relu[i] = Math.Max (0, h_linear[i]);

        // Second layer
        double[] y = new double[EmbedSize];
        for (int i = 0; i < EmbedSize; i++) {
            y[i] = B2[i];
            for (int j = 0; j < HiddenSize; j++)
                y[i] += W2[i, j] * h_relu[j];
        }

        var cache = new FeedForwardCache {
            x_input = x,
            h_linear = h_linear,
            h_relu = h_relu
        };

        return (y, cache);
    }

    public double[] Backward (double[] dOut, FeedForwardCache cache) {
        double[] dh_relu = new double[HiddenSize];

        // Backprop through second layer
        for (int i = 0; i < EmbedSize; i++) {
            dB2[i] += dOut[i];
            for (int j = 0; j < HiddenSize; j++) {
                dW2[i, j] += dOut[i] * cache.h_relu[j];
                dh_relu[j] += dOut[i] * W2[i, j];
            }
        }

        // Backprop through ReLU
        double[] dh_linear = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++)
            dh_linear[i] = cache.h_linear[i] > 0 ? dh_relu[i] : 0;

        // Backprop through first layer
        double[] dx = new double[EmbedSize];
        for (int i = 0; i < HiddenSize; i++) {
            dB1[i] += dh_linear[i];
            for (int j = 0; j < EmbedSize; j++) {
                dW1[i, j] += dh_linear[i] * cache.x_input[j];
                dx[j] += dh_linear[i] * W1[i, j];
            }
        }

        return dx;
    }
}

public class FeedForwardCache
{
    public double[] x_input;
    public double[] h_linear;
    public double[] h_relu;
}

// Transformer Block with backward pass
public class TransformerBlock
{
    public LayerNorm Norm1, Norm2;
    public SelfAttention SelfAttention;
    public FeedForward FeedForward;
    public int EmbedSize;

    public TransformerBlock (int embedSize, int hiddenSize, int headSize, Random random) {
        EmbedSize = embedSize;
        Norm1 = new LayerNorm (embedSize);
        Norm2 = new LayerNorm (embedSize);
        SelfAttention = new SelfAttention (embedSize, headSize, random);
        FeedForward = new FeedForward (embedSize, hiddenSize, random);
    }

    public (List<double[]> outputs, TransformerBlockCache cache) Forward (List<double[]> inputs, int startPosition) {
        int seqLen = inputs.Count;

        var cache = new TransformerBlockCache {
            inputs = inputs,
            normedInputs = new List<double[]> (),
            norm1Caches = new List<LayerNormCache> (),
            saOutputs = new List<double[]> (),
            saCache = null,
            saAdded = new List<double[]> (),
            normedSaAdded = new List<double[]> (),
            norm2Caches = new List<LayerNormCache> (),
            ffOutputs = new List<double[]> (),
            ffCaches = new List<FeedForwardCache> (),
            ffAdded = new List<double[]> ()
        };

        // Apply first LayerNorm and Self-Attention
        foreach (var x in inputs) {
            var (normedInput, norm1Cache) = Norm1.Forward (x);
            cache.normedInputs.Add (normedInput);
            cache.norm1Caches.Add (norm1Cache);
        }

        (cache.saOutputs, cache.saCache) = SelfAttention.Forward (cache.normedInputs, startPosition);

        // Add & Norm
        for (int i = 0; i < seqLen; i++) {
            var saAdd = MathOps.Add (inputs[i], cache.saOutputs[i]);
            cache.saAdded.Add (saAdd);

            var (normedSaAdd, norm2Cache) = Norm2.Forward (saAdd);
            cache.normedSaAdded.Add (normedSaAdd);
            cache.norm2Caches.Add (norm2Cache);

            var (ffOutput, ffCache) = FeedForward.Forward (normedSaAdd);
            cache.ffOutputs.Add (ffOutput);
            cache.ffCaches.Add (ffCache);

            var ffAdd = MathOps.Add (saAdd, ffOutput);
            cache.ffAdded.Add (ffAdd);
        }

        return (cache.ffAdded, cache);
    }

    public List<double[]> Backward (List<double[]> gradOutputs, TransformerBlockCache cache) {
        int seqLen = gradOutputs.Count;
        List<double[]> dInputs = new List<double[]> ();
        for (int i = 0; i < seqLen; i++)
            dInputs.Add (new double[EmbedSize]);

        // Backward through FeedForward and second residual connection
        List<double[]> dSaAdded = new List<double[]> ();
        for (int i = 0; i < seqLen; i++) {
            // Gradient w.r.t. the output of the feed-forward layer
            double[] dFfAdded = gradOutputs[i];

            // Backprop through residual connection
            double[] dFfOutput = new double[EmbedSize];
            double[] dSaAdd = new double[EmbedSize];
            for (int j = 0; j < EmbedSize; j++) {
                dFfOutput[j] = dFfAdded[j];
                dSaAdd[j] = dFfAdded[j];
            }

            // Backprop through feed-forward layer
            double[] dNormedSaAddedSingle = FeedForward.Backward (dFfOutput, cache.ffCaches[i]);

            // Backprop through second LayerNorm
            double[] dSaAddedNorm = Norm2.Backward (dNormedSaAddedSingle, cache.norm2Caches[i]);

            // Accumulate gradients
            for (int j = 0; j < EmbedSize; j++)
                dSaAdd[j] += dSaAddedNorm[j];

            // Store gradient for saAdd
            dSaAdded.Add (dSaAdd);
        }

        // Backward through Self-Attention and first residual connection
        List<double[]> dNormedInputs = SelfAttention.Backward (dSaAdded, cache.saCache);

        // Backward through first LayerNorm and residual connection
        for (int i = 0; i < seqLen; i++) {
            double[] dSaOutput = dNormedInputs[i];

            // Backprop through residual connection
            for (int j = 0; j < EmbedSize; j++) {
                dInputs[i][j] += dSaAdded[i][j]; // Accumulate gradient from residual
                dInputs[i][j] += dSaOutput[j]; // Accumulate gradient from Self-Attention
            }

            // Backprop through first LayerNorm
            double[] dInputNorm = Norm1.Backward (dInputs[i], cache.norm1Caches[i]);

            // Assign gradients to avoid double-counting
            for (int j = 0; j < EmbedSize; j++)
                dInputs[i][j] = dInputNorm[j];
        }

        return dInputs;
    }
}

public class TransformerBlockCache
{
    public List<double[]> inputs;
    public List<double[]> normedInputs;
    public List<LayerNormCache> norm1Caches;
    public List<double[]> saOutputs;
    public SelfAttentionCache saCache;
    public List<double[]> saAdded;
    public List<double[]> normedSaAdded;
    public List<LayerNormCache> norm2Caches;
    public List<double[]> ffOutputs;
    public List<FeedForwardCache> ffCaches;
    public List<double[]> ffAdded;
}

// LlamaForCausalLM Model with backward pass
public class LlamaForCausalLM
{
    public Embedding TokenEmbedding;
    public List<TransformerBlock> TransformerBlocks;
    public LayerNorm FinalLayerNorm;
    public double[,] OutputProjection;

    public double[,] dOutputProjection;

    // For Adam optimizer
    public double[,] mOutputProjection;
    public double[,] vOutputProjection;

    public int VocabSize, EmbedSize, HiddenSize, HeadSize, NumLayers;

    private List<double[]> embeddings;
    private int[] inputTokens;
    private List<double[]> finalEmbeddings;
    private List<LayerNormCache> finalLayerNormCaches;
    private List<TransformerBlockCache> transformerCaches;

    private Random rand;

    public LlamaForCausalLM (int vocabSize, int embedSize, int hiddenSize, int headSize, int numLayers, Random random) {
        rand = random;
        VocabSize = vocabSize;
        EmbedSize = embedSize;
        HiddenSize = hiddenSize;
        HeadSize = headSize;
        NumLayers = numLayers;
        TokenEmbedding = new Embedding (vocabSize, embedSize, rand);
        TransformerBlocks = new List<TransformerBlock> ();
        for (int i = 0; i < numLayers; i++)
            TransformerBlocks.Add (new TransformerBlock (embedSize, hiddenSize, headSize, rand));
        FinalLayerNorm = new LayerNorm (embedSize);
        OutputProjection = InitializeMatrix (VocabSize, EmbedSize);
        dOutputProjection = new double[VocabSize, EmbedSize];
        mOutputProjection = new double[VocabSize, EmbedSize];
        vOutputProjection = new double[VocabSize, EmbedSize];
    }

    private double[,] InitializeMatrix (int rows, int cols) {
        double[,] matrix = new double[rows, cols];
        double limit = Math.Sqrt (6.0 / (rows + cols)); // Xavier Initialization
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble () * 2 * limit - limit;
        return matrix;
    }

    public List<double[]> Forward (int[] inputTokens) {
        this.inputTokens = inputTokens;
        embeddings = inputTokens.Select (token => TokenEmbedding.Forward (token)).ToList ();

        // Process through transformer blocks
        transformerCaches = new List<TransformerBlockCache> ();
        int startPosition = 0;
        foreach (var block in TransformerBlocks) {
            (embeddings, var cache) = block.Forward (embeddings, startPosition);
            transformerCaches.Add (cache);
        }

        // Apply final layer normalization to each token's embedding
        finalLayerNormCaches = new List<LayerNormCache> ();
        finalEmbeddings = new List<double[]> ();
        foreach (var e in embeddings) {
            (var finalEmbedding, var cache) = FinalLayerNorm.Forward (e);
            finalEmbeddings.Add (finalEmbedding);
            finalLayerNormCaches.Add (cache);
        }

        // Compute logits for each time step
        List<double[]> logitsList = new List<double[]> ();
        foreach (var finalEmbedding in finalEmbeddings) {
            double[] logits = new double[VocabSize];
            for (int i = 0; i < VocabSize; i++)
                logits[i] = MathOps.Dot (OutputProjection.GetRow (i), finalEmbedding);
            logitsList.Add (logits);
        }

        return logitsList;
    }

    public void Backward (List<double[]> dLogitsList) {
        // Initialize gradients for embeddings
        List<double[]> dEmbeddings = new List<double[]> ();
        for (int i = 0; i < embeddings.Count; i++)
            dEmbeddings.Add (new double[EmbedSize]);

        // Backward through OutputProjection for each time step
        for (int t = 0; t < embeddings.Count; t++) {
            // Use cached final embeddings
            var finalEmbedding = finalEmbeddings[t];

            // Gradients w.r.t. OutputProjection
            for (int i = 0; i < VocabSize; i++) {
                for (int j = 0; j < EmbedSize; j++) {
                    dOutputProjection[i, j] += dLogitsList[t][i] * finalEmbedding[j];
                }
            }

            // Gradients w.r.t. final embeddings
            double[] dFinalEmbedding = new double[EmbedSize];
            for (int j = 0; j < EmbedSize; j++) {
                for (int i = 0; i < VocabSize; i++)
                    dFinalEmbedding[j] += OutputProjection[i, j] * dLogitsList[t][i];
            }

            // Backward through FinalLayerNorm
            double[] dEmbedding = FinalLayerNorm.Backward (dFinalEmbedding, finalLayerNormCaches[t]);

            // Accumulate gradients
            for (int j = 0; j < EmbedSize; j++)
                dEmbeddings[t][j] += dEmbedding[j];
        }

        // Backward through TransformerBlocks in reverse order
        for (int i = TransformerBlocks.Count - 1; i >= 0; i--) {
            var block = TransformerBlocks[i];
            dEmbeddings = block.Backward (dEmbeddings, transformerCaches[i]);
        }

        // Backward through TokenEmbedding
        for (int i = 0; i < inputTokens.Length; i++) {
            TokenEmbedding.Backward (inputTokens[i], dEmbeddings[i]);
        }
    }
}

// Extension method to get a row from a 2D array
public static class Extensions
{
    public static double[] GetRow (this double[,] matrix, int row) {
        int cols = matrix.GetLength (1);
        double[] result = new double[cols];
        for (int i = 0; i < cols; i++)
            result[i] = matrix[row, i];
        return result;
    }
}

// Loss Function with gradient
public static class LossFunctions
{
    public static double CrossEntropyLoss (double[] logits, int targetToken, out double[] dLogits) {
        double[] probabilities = MathOps.Softmax (logits);
        dLogits = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            dLogits[i] = probabilities[i];
        dLogits[targetToken] -= 1; // Gradient of softmax cross-entropy
        return -Math.Log (probabilities[targetToken] + 1e-9);
    }
}

// Adam Optimizer with gradient clipping
public class AdamOptimizer
{
    public double LearningRate;
    public double Beta1;
    public double Beta2;
    public double Epsilon;
    private int timestep;
    public double GradientClipValue; // Added for gradient clipping

    public AdamOptimizer (double learningRate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double gradientClipValue = 1.0) {
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        timestep = 0;
        GradientClipValue = gradientClipValue;
    }

    public void Step (LlamaForCausalLM model) {
        timestep++;

        // Update TokenEmbedding weights
        UpdateParameters (model.TokenEmbedding.Weights, model.TokenEmbedding.Gradients,
            model.TokenEmbedding.mWeights, model.TokenEmbedding.vWeights);

        ZeroGradients (model.TokenEmbedding.Gradients);

        // Update OutputProjection
        UpdateParameters (model.OutputProjection, model.dOutputProjection,
            model.mOutputProjection, model.vOutputProjection);

        ZeroGradients (model.dOutputProjection);

        // Update LayerNorm parameters
        UpdateLayerNorm (model.FinalLayerNorm);

        // Update TransformerBlocks
        foreach (var block in model.TransformerBlocks) {
            UpdateLayerNorm (block.Norm1);
            UpdateLayerNorm (block.Norm2);

            // Update SelfAttention parameters
            UpdateParameters (block.SelfAttention.Wq, block.SelfAttention.dWq,
                block.SelfAttention.mWq, block.SelfAttention.vWq);
            UpdateParameters (block.SelfAttention.Wk, block.SelfAttention.dWk,
                block.SelfAttention.mWk, block.SelfAttention.vWk);
            UpdateParameters (block.SelfAttention.Wv, block.SelfAttention.dWv,
                block.SelfAttention.mWv, block.SelfAttention.vWv);
            UpdateParameters (block.SelfAttention.Wo, block.SelfAttention.dWo,
                block.SelfAttention.mWo, block.SelfAttention.vWo);

            ZeroGradients (block.SelfAttention.dWq);
            ZeroGradients (block.SelfAttention.dWk);
            ZeroGradients (block.SelfAttention.dWv);
            ZeroGradients (block.SelfAttention.dWo);

            // Update FeedForward parameters
            UpdateFeedForward (block.FeedForward);
        }
    }

    private void UpdateLayerNorm (LayerNorm layerNorm) {
        double biasCorrection1 = 1 - Math.Pow (Beta1, timestep);
        double biasCorrection2 = 1 - Math.Pow (Beta2, timestep);

        // Compute global norm
        double globalNormGamma = ComputeGlobalNorm (layerNorm.dGamma);
        double globalNormBeta = ComputeGlobalNorm (layerNorm.dBeta);

        double clipCoeffGamma = GradientClipValue / (globalNormGamma + 1e-6);
        if (clipCoeffGamma > 1.0) clipCoeffGamma = 1.0;

        double clipCoeffBeta = GradientClipValue / (globalNormBeta + 1e-6);
        if (clipCoeffBeta > 1.0) clipCoeffBeta = 1.0;

        for (int i = 0; i < layerNorm.Size; i++) {
            // Apply global norm clipping
            double gradGamma = layerNorm.dGamma[i] * clipCoeffGamma;
            double gradBeta = layerNorm.dBeta[i] * clipCoeffBeta;

            // Update Gamma
            layerNorm.mGamma[i] = Beta1 * layerNorm.mGamma[i] + (1 - Beta1) * gradGamma;
            layerNorm.vGamma[i] = Beta2 * layerNorm.vGamma[i] + (1 - Beta2) * gradGamma * gradGamma;

            double mHatGamma = layerNorm.mGamma[i] / biasCorrection1;
            double vHatGamma = layerNorm.vGamma[i] / biasCorrection2;

            layerNorm.Gamma[i] -= LearningRate * mHatGamma / (Math.Sqrt (vHatGamma) + Epsilon);

            // Update Beta
            layerNorm.mBeta[i] = Beta1 * layerNorm.mBeta[i] + (1 - Beta1) * gradBeta;
            layerNorm.vBeta[i] = Beta2 * layerNorm.vBeta[i] + (1 - Beta2) * gradBeta * gradBeta;

            double mHatBeta = layerNorm.mBeta[i] / biasCorrection1;
            double vHatBeta = layerNorm.vBeta[i] / biasCorrection2;

            layerNorm.Beta[i] -= LearningRate * mHatBeta / (Math.Sqrt (vHatBeta) + Epsilon);

            layerNorm.dGamma[i] = 0;
            layerNorm.dBeta[i] = 0;
        }
    }

    private void UpdateFeedForward (FeedForward feedForward) {
        // Update W1 and B1
        UpdateParameters (feedForward.W1, feedForward.dW1, feedForward.mW1, feedForward.vW1);
        UpdateParameters (feedForward.W2, feedForward.dW2, feedForward.mW2, feedForward.vW2);
        UpdateParameters (feedForward.B1, feedForward.dB1, feedForward.mB1, feedForward.vB1);
        UpdateParameters (feedForward.B2, feedForward.dB2, feedForward.mB2, feedForward.vB2);

        ZeroGradients (feedForward.dW1);
        ZeroGradients (feedForward.dW2);
        ZeroGradients (feedForward.dB1);
        ZeroGradients (feedForward.dB2);
    }

    private double ComputeGlobalNorm (double[,] gradients) {
        double sum = 0.0;
        int rows = gradients.GetLength (0);
        int cols = gradients.GetLength (1);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            sum += gradients[i, j] * gradients[i, j];
        return Math.Sqrt (sum);
    }

    private double ComputeGlobalNorm (double[] gradients) {
        double sum = 0.0;
        int length = gradients.Length;
        for (int i = 0; i < length; i++)
            sum += gradients[i] * gradients[i];
        return Math.Sqrt (sum);
    }

    private void UpdateParameters (double[,] weights, double[,] gradients, double[,] m, double[,] v) {
        int rows = weights.GetLength (0);
        int cols = weights.GetLength (1);
        double biasCorrection1 = 1 - Math.Pow (Beta1, timestep);
        double biasCorrection2 = 1 - Math.Pow (Beta2, timestep);

        // Compute global norm
        double globalNorm = ComputeGlobalNorm (gradients);
        double clipCoeff = GradientClipValue / (globalNorm + 1e-6);
        if (clipCoeff > 1.0) clipCoeff = 1.0;

        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            // Apply global norm clipping
            double grad = gradients[i, j] * clipCoeff;

            m[i, j] = Beta1 * m[i, j] + (1 - Beta1) * grad;
            v[i, j] = Beta2 * v[i, j] + (1 - Beta2) * grad * grad;

            double mHat = m[i, j] / biasCorrection1;
            double vHat = v[i, j] / biasCorrection2;

            weights[i, j] -= LearningRate * mHat / (Math.Sqrt (vHat) + Epsilon);
        }
    }

    private void UpdateParameters (double[] weights, double[] gradients, double[] m, double[] v) {
        int length = weights.Length;
        double biasCorrection1 = 1 - Math.Pow (Beta1, timestep);
        double biasCorrection2 = 1 - Math.Pow (Beta2, timestep);

        // Compute global norm
        double globalNorm = ComputeGlobalNorm (gradients);
        double clipCoeff = GradientClipValue / (globalNorm + 1e-6);
        if (clipCoeff > 1.0) clipCoeff = 1.0;

        for (int i = 0; i < length; i++) {
            // Apply global norm clipping
            double grad = gradients[i] * clipCoeff;

            m[i] = Beta1 * m[i] + (1 - Beta1) * grad;
            v[i] = Beta2 * v[i] + (1 - Beta2) * grad * grad;

            double mHat = m[i] / biasCorrection1;
            double vHat = v[i] / biasCorrection2;

            weights[i] -= LearningRate * mHat / (Math.Sqrt (vHat) + Epsilon);
        }
    }

    private void ZeroGradients (double[,] gradients) {
        int rows = gradients.GetLength (0);
        int cols = gradients.GetLength (1);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            gradients[i, j] = 0;
    }

    private void ZeroGradients (double[] gradients) {
        for (int i = 0; i < gradients.Length; i++)
            gradients[i] = 0;
    }
}

// Training Code with backward pass and parameter updates
public class Trainer
{
    public static void train (LlamaForCausalLM model, AdamOptimizer optimizer, Func<(int[], int[])> data, int epochs, int epochSize,
        Action callback) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            for (int i = 0; i < epochSize; i++) {
                var (inputTokens, targetTokens) = data ();

                // Forward pass
                List<double[]> logitsList = model.Forward (inputTokens);

                // Compute loss and gradient for each time step
                List<double[]> dLogitsList = new List<double[]> ();
                double loss = 0.0;

                for (int t = 0; t < targetTokens.Length; t++) {
                    double[] dLogits;
                    loss += LossFunctions.CrossEntropyLoss (logitsList[t], targetTokens[t], out dLogits);
                    dLogitsList.Add (dLogits);
                }

                totalLoss += loss / targetTokens.Length;

                // Backward pass
                model.Backward (dLogitsList);

                // Update parameters
                optimizer.Step (model);
            }

            Console.WriteLine ($"Epoch {epoch + 1}/{epochs}, Loss: {totalLoss / epochSize}");

            callback?.Invoke ();
        }
    }
}
