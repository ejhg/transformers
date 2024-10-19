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

    // Extension method to get a row from a 2D array
    public static double[] GetRow (double[,] matrix, int row) {
        int cols = matrix.GetLength (1);
        double[] result = new double[cols];
        for (int i = 0; i < cols; i++)
            result[i] = matrix[row, i];
        return result;
    }

    public static double[,] InitializeMatrix (Random rand, int rows, int cols) {
        double[,] matrix = new double[rows, cols];
        double limit = Math.Sqrt (6.0 / (rows + cols)); // Xavier Initialization
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble () * 2 * limit - limit;
        return matrix;
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
        Weights = MathOps.InitializeMatrix (rand, vocabSize, embedSize);
        Gradients = new double[vocabSize, embedSize];
        mWeights = new double[vocabSize, embedSize];
        vWeights = new double[vocabSize, embedSize];
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

// RMSNorm layer with backward pass
public class RMSNorm
{
    public double[] Gamma;
    public int Size;

    public double[] dGamma;

    // For Adam optimizer
    public double[] mGamma;
    public double[] vGamma;

    public RMSNorm (int size) {
        Size = size;
        Gamma = new double[size];
        dGamma = new double[size];
        mGamma = new double[size];
        vGamma = new double[size];
        for (int i = 0; i < size; i++) {
            Gamma[i] = 1.0;
        }
    }

    public (double[] output, RMSNormCache cache) Forward (double[] x) {
        double rms = Math.Sqrt (x.Select (val => val * val).Average () + 1e-6);
        double[] output = new double[Size];
        for (int i = 0; i < Size; i++) {
            output[i] = x[i] * (Gamma[i] / rms);
        }

        var cache = new RMSNormCache {
            x_input = x,
            rms = rms
        };

        return (output, cache);
    }

    public double[] Backward (double[] gradOutput, RMSNormCache cache) {
        double[] dx = new double[Size];
        double[] x = cache.x_input;
        double rms = cache.rms;

        // Compute gradient w.r.t Gamma
        for (int i = 0; i < Size; i++) {
            dGamma[i] += gradOutput[i] * x[i] / rms;
        }

        // Compute gradient w.r.t x
        double dot = 0.0;
        for (int i = 0; i < Size; i++) {
            dot += x[i] * gradOutput[i] * Gamma[i];
        }

        double rms_cubed = rms * rms * rms;
        for (int i = 0; i < Size; i++) {
            dx[i] = Gamma[i] / rms * (gradOutput[i] - x[i] * dot / rms_cubed);
        }

        return dx;
    }
}

public class RMSNormCache
{
    public double[] x_input;
    public double rms;
}

// Self-Attention mechanism with RoPE and backward pass (Multi-Head)
public class SelfAttention
{
    public double[][,] Wq, Wk, Wv; // Each is [HeadDim, EmbedSize], total NumHeads elements
    public double[,] Wo; // [EmbedSize, EmbedSize]

    // Gradients
    public double[][,] dWq, dWk, dWv;
    public double[,] dWo;

    // For Adam optimizer
    public double[][,] mWq, vWq;
    public double[][,] mWk, vWk;
    public double[][,] mWv, vWv;
    public double[,] mWo, vWo;

    public int EmbedSize, NumHeads, HeadDim;

    private Random rand;

    public SelfAttention (int embedSize, int numHeads, Random random) {
        rand = random;
        EmbedSize = embedSize;
        NumHeads = numHeads;
        HeadDim = embedSize / numHeads;

        Wq = new double[NumHeads][,];
        Wk = new double[NumHeads][,];
        Wv = new double[NumHeads][,];
        dWq = new double[NumHeads][,];
        dWk = new double[NumHeads][,];
        dWv = new double[NumHeads][,];
        mWq = new double[NumHeads][,];
        vWq = new double[NumHeads][,];
        mWk = new double[NumHeads][,];
        vWk = new double[NumHeads][,];
        mWv = new double[NumHeads][,];
        vWv = new double[NumHeads][,];

        for (int h = 0; h < NumHeads; h++) {
            Wq[h] = MathOps.InitializeMatrix (rand, HeadDim, EmbedSize);
            Wk[h] = MathOps.InitializeMatrix (rand, HeadDim, EmbedSize);
            Wv[h] = MathOps.InitializeMatrix (rand, HeadDim, EmbedSize);
            dWq[h] = new double[HeadDim, EmbedSize];
            dWk[h] = new double[HeadDim, EmbedSize];
            dWv[h] = new double[HeadDim, EmbedSize];
            mWq[h] = new double[HeadDim, EmbedSize];
            vWq[h] = new double[HeadDim, EmbedSize];
            mWk[h] = new double[HeadDim, EmbedSize];
            vWk[h] = new double[HeadDim, EmbedSize];
            mWv[h] = new double[HeadDim, EmbedSize];
            vWv[h] = new double[HeadDim, EmbedSize];
        }

        Wo = MathOps.InitializeMatrix (rand, EmbedSize, EmbedSize);
        dWo = new double[EmbedSize, EmbedSize];
        mWo = new double[EmbedSize, EmbedSize];
        vWo = new double[EmbedSize, EmbedSize];
    }

    public (List<double[]> outputs, SelfAttentionCache cache) Forward (List<double[]> inputs, int startPosition) {
        int seqLen = inputs.Count;

        var cache = new SelfAttentionCache {
            Inputs = inputs,
            Qs = new List<List<double[]>> (),
            Ks = new List<List<double[]>> (),
            Vs = new List<List<double[]>> (),
            Zs = new List<double[]> (),
            Softmaxes = new List<List<double[]>> (),
            StartPosition = startPosition
        };

        List<double[]> outputs = new List<double[]> ();

        // Initialize cache lists
        for (int h = 0; h < NumHeads; h++) {
            cache.Qs.Add (new List<double[]> ());
            cache.Ks.Add (new List<double[]> ());
            cache.Vs.Add (new List<double[]> ());
            cache.Softmaxes.Add (new List<double[]> ());
        }

        // Compute Q, K, V for each head and position
        for (int t = 0; t < seqLen; t++) {
            double[] x = inputs[t];

            for (int h = 0; h < NumHeads; h++) {
                double[] q = MathOps.MatrixVectorProduct (Wq[h], x);
                double[] k = MathOps.MatrixVectorProduct (Wk[h], x);
                double[] v = MathOps.MatrixVectorProduct (Wv[h], x);

                q = ApplyRoPE (q, t + startPosition);
                k = ApplyRoPE (k, t + startPosition);

                cache.Qs[h].Add (q);
                cache.Ks[h].Add (k);
                cache.Vs[h].Add (v);
            }
        }

        // Compute attention outputs for each position
        for (int t = 0; t < seqLen; t++) {
            double[] concatHeads = new double[EmbedSize];

            for (int h = 0; h < NumHeads; h++) {
                // Compute attention scores up to position t
                double[] scores = new double[t + 1];
                for (int j = 0; j <= t; j++) {
                    scores[j] = MathOps.Dot (cache.Qs[h][t], cache.Ks[h][j]) / Math.Sqrt (HeadDim);
                }

                // Apply softmax to get attention weights
                double[] softmax = MathOps.Softmax (scores);
                cache.Softmaxes[h].Add (softmax);

                // Compute weighted sum of Vs
                double[] z = new double[HeadDim];
                for (int j = 0; j <= t; j++) {
                    for (int k = 0; k < HeadDim; k++) {
                        z[k] += softmax[j] * cache.Vs[h][j][k];
                    }
                }

                // Place z into the appropriate position in concatHeads
                Array.Copy (z, 0, concatHeads, h * HeadDim, HeadDim);
            }

            // Apply output projection
            double[] output = MathOps.MatrixVectorProduct (Wo, concatHeads);
            outputs.Add (output);

            // Store z in cache for backward pass
            cache.Zs.Add (concatHeads);
        }

        return (outputs, cache);
    }

    public List<double[]> Backward (List<double[]> gradOutputs, SelfAttentionCache cache) {
        int seqLen = cache.Inputs.Count;
        List<double[]> dInputs = new List<double[]> ();
        for (int i = 0; i < seqLen; i++)
            dInputs.Add (new double[EmbedSize]);

        double[,] dWo = new double[EmbedSize, EmbedSize];
        double[][] dConcatHeads = new double[seqLen][];
        for (int t = 0; t < seqLen; t++) {
            dConcatHeads[t] = new double[EmbedSize];
        }

        // Backprop through output projection
        for (int t = 0; t < seqLen; t++) {
            double[] gradOutput = gradOutputs[t];
            double[] concatHeads = cache.Zs[t];

            // Gradient w.r.t. concatHeads
            double[] dConcatHead = new double[EmbedSize];
            for (int i = 0; i < EmbedSize; i++) {
                for (int j = 0; j < EmbedSize; j++) {
                    dConcatHead[j] += Wo[i, j] * gradOutput[i];
                    dWo[i, j] += gradOutput[i] * concatHeads[j];
                }
            }

            // Store dConcatHead for each position
            dConcatHeads[t] = dConcatHead;
        }

        // Accumulate gradients for W_o
        for (int i = 0; i < EmbedSize; i++)
        for (int j = 0; j < EmbedSize; j++)
            this.dWo[i, j] += dWo[i, j];

        // Backprop through attention heads
        for (int h = 0; h < NumHeads; h++) {
            // Initialize gradients for Q, K, V
            double[][] dQs = new double[seqLen][];
            double[][] dKs = new double[seqLen][];
            double[][] dVs = new double[seqLen][];
            for (int i = 0; i < seqLen; i++) {
                dQs[i] = new double[HeadDim];
                dKs[i] = new double[HeadDim];
                dVs[i] = new double[HeadDim];
            }

            // Backprop through attention mechanism
            for (int t = seqLen - 1; t >= 0; t--) {
                double[] dZ = new double[HeadDim];
                Array.Copy (dConcatHeads[t], h * HeadDim, dZ, 0, HeadDim);

                // Backprop through weighted sum of Vs
                double[] dAlpha = new double[t + 1];
                for (int j = 0; j <= t; j++) {
                    for (int i = 0; i < HeadDim; i++) {
                        dVs[j][i] += cache.Softmaxes[h][t][j] * dZ[i];
                    }

                    dAlpha[j] = MathOps.Dot (cache.Vs[h][j], dZ);
                }

                // Backprop through softmax
                double[] dScores = MathOps.SoftmaxGradient (cache.Softmaxes[h][t], dAlpha);

                // Backprop through attention scores
                double scale = 1.0 / Math.Sqrt (HeadDim);
                for (int j = 0; j <= t; j++) {
                    for (int i = 0; i < HeadDim; i++) {
                        dQs[t][i] += dScores[j] * cache.Ks[h][j][i] * scale;
                        dKs[j][i] += dScores[j] * cache.Qs[h][t][i] * scale;
                    }
                }
            }

            // Backprop through RoPE embeddings
            for (int t = 0; t < seqLen; t++) {
                int position = t + cache.StartPosition;

                dQs[t] = BackwardRoPE (dQs[t], cache.Qs[h][t], position);
                dKs[t] = BackwardRoPE (dKs[t], cache.Ks[h][t], position);
            }

            // Backprop through linear layers Wq, Wk, Wv
            for (int t = 0; t < seqLen; t++) {
                double[] x = cache.Inputs[t];

                // Gradients w.r.t. Wq and input x
                for (int i = 0; i < HeadDim; i++) {
                    for (int j = 0; j < EmbedSize; j++) {
                        dWq[h][i, j] += dQs[t][i] * x[j];
                        dInputs[t][j] += Wq[h][i, j] * dQs[t][i];
                    }
                }

                // Gradients w.r.t. Wk and input x
                for (int i = 0; i < HeadDim; i++) {
                    for (int j = 0; j < EmbedSize; j++) {
                        dWk[h][i, j] += dKs[t][i] * x[j];
                        dInputs[t][j] += Wk[h][i, j] * dKs[t][i];
                    }
                }

                // Gradients w.r.t. Wv and input x
                for (int i = 0; i < HeadDim; i++) {
                    for (int j = 0; j < EmbedSize; j++) {
                        dWv[h][i, j] += dVs[t][i] * x[j];
                        dInputs[t][j] += Wv[h][i, j] * dVs[t][i];
                    }
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
    public List<List<double[]>> Qs;
    public List<List<double[]>> Ks;
    public List<List<double[]>> Vs;
    public List<double[]> Zs;
    public List<List<double[]>> Softmaxes;
    public int StartPosition;
}

// Feed-Forward Network with SwiGLU activation and backward pass
public class FeedForward
{
    public double[,] W1; // Shape: (HiddenSize * 2, EmbedSize)
    public double[,] W2; // Shape: (EmbedSize, HiddenSize)
    public double[] B1; // Shape: (HiddenSize * 2)
    public double[] B2; // Shape: (EmbedSize)

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

        // Adjusted dimensions for SwiGLU
        W1 = MathOps.InitializeMatrix (rand, HiddenSize * 2, EmbedSize);
        B1 = new double[HiddenSize * 2];
        W2 = MathOps.InitializeMatrix (rand, EmbedSize, HiddenSize);
        B2 = new double[EmbedSize];

        dW1 = new double[HiddenSize * 2, EmbedSize];
        dB1 = new double[HiddenSize * 2];
        dW2 = new double[EmbedSize, HiddenSize];
        dB2 = new double[EmbedSize];

        mW1 = new double[HiddenSize * 2, EmbedSize];
        vW1 = new double[HiddenSize * 2, EmbedSize];
        mW2 = new double[EmbedSize, HiddenSize];
        vW2 = new double[EmbedSize, HiddenSize];
        mB1 = new double[HiddenSize * 2];
        vB1 = new double[HiddenSize * 2];
        mB2 = new double[EmbedSize];
        vB2 = new double[EmbedSize];
    }

    public (double[] output, FeedForwardCache cache) Forward (double[] x) {
        // First layer
        double[] h_linear = new double[HiddenSize * 2];
        for (int i = 0; i < HiddenSize * 2; i++) {
            h_linear[i] = B1[i];
            for (int j = 0; j < EmbedSize; j++)
                h_linear[i] += W1[i, j] * x[j];
        }

        // Split and apply SwiGLU activation
        double[] h_a = new double[HiddenSize];
        double[] h_b = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++) {
            h_a[i] = h_linear[i];
            h_b[i] = h_linear[i + HiddenSize];
        }

        double[] h_swiglu = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++) {
            double swish = h_b[i] * Sigmoid (h_b[i]);
            h_swiglu[i] = h_a[i] * swish;
        }

        // Second layer
        double[] y = new double[EmbedSize];
        for (int i = 0; i < EmbedSize; i++) {
            y[i] = B2[i];
            for (int j = 0; j < HiddenSize; j++)
                y[i] += W2[i, j] * h_swiglu[j];
        }

        var cache = new FeedForwardCache {
            x_input = x,
            h_linear = h_linear,
            h_a = h_a,
            h_b = h_b,
            h_swiglu = h_swiglu
        };

        return (y, cache);
    }

    public double[] Backward (double[] dOut, FeedForwardCache cache) {
        double[] dh_swiglu = new double[HiddenSize];

        // Backprop through second layer
        for (int i = 0; i < EmbedSize; i++) {
            dB2[i] += dOut[i];
            for (int j = 0; j < HiddenSize; j++) {
                dW2[i, j] += dOut[i] * cache.h_swiglu[j];
                dh_swiglu[j] += dOut[i] * W2[i, j];
            }
        }

        // Backprop through SwiGLU activation
        double[] dh_a = new double[HiddenSize];
        double[] dh_b = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++) {
            double swish = cache.h_b[i] * Sigmoid (cache.h_b[i]);
            double sigmoid = Sigmoid (cache.h_b[i]);
            dh_a[i] = dh_swiglu[i] * swish;
            dh_b[i] = dh_swiglu[i] * cache.h_a[i] * (sigmoid + cache.h_b[i] * sigmoid * (1 - sigmoid));
        }

        // Combine gradients
        double[] dh_linear = new double[HiddenSize * 2];
        for (int i = 0; i < HiddenSize; i++) {
            dh_linear[i] = dh_a[i];
            dh_linear[i + HiddenSize] = dh_b[i];
        }

        // Backprop through first layer
        double[] dx = new double[EmbedSize];
        for (int i = 0; i < HiddenSize * 2; i++) {
            dB1[i] += dh_linear[i];
            for (int j = 0; j < EmbedSize; j++) {
                dW1[i, j] += dh_linear[i] * cache.x_input[j];
                dx[j] += dh_linear[i] * W1[i, j];
            }
        }

        return dx;
    }

    private double Sigmoid (double x) {
        return 1.0 / (1.0 + Math.Exp (-x));
    }
}

public class FeedForwardCache
{
    public double[] x_input;
    public double[] h_linear;
    public double[] h_a;
    public double[] h_b;
    public double[] h_swiglu;
}

// Transformer Block with RMSNorm and updated components
public class TransformerBlock
{
    public RMSNorm Norm1, Norm2;
    public SelfAttention SelfAttention;
    public FeedForward FeedForward;
    public int EmbedSize;

    public TransformerBlock (int embedSize, int hiddenSize, int numHeads, Random random) {
        EmbedSize = embedSize;
        Norm1 = new RMSNorm (embedSize);
        Norm2 = new RMSNorm (embedSize);
        SelfAttention = new SelfAttention (embedSize, numHeads, random);
        FeedForward = new FeedForward (embedSize, hiddenSize, random);
    }

    public (List<double[]> outputs, TransformerBlockCache cache) Forward (List<double[]> inputs, int startPosition) {
        int seqLen = inputs.Count;

        var cache = new TransformerBlockCache {
            normedInputs = new List<double[]> (),
            norm1Caches = new List<RMSNormCache> (),
            saOutputs = new List<double[]> (),
            saCache = null,
            saAdded = new List<double[]> (),
            normedSaAdded = new List<double[]> (),
            norm2Caches = new List<RMSNormCache> (),
            ffOutputs = new List<double[]> (),
            ffCaches = new List<FeedForwardCache> (),
            ffAdded = new List<double[]> ()
        };

        // Apply first RMSNorm and Self-Attention
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

            // Backprop through second RMSNorm
            double[] dSaAddedNorm = Norm2.Backward (dNormedSaAddedSingle, cache.norm2Caches[i]);

            // Accumulate gradients
            for (int j = 0; j < EmbedSize; j++)
                dSaAdd[j] += dSaAddedNorm[j];

            // Store gradient for saAdd
            dSaAdded.Add (dSaAdd);
        }

        // Backward through Self-Attention and first residual connection
        List<double[]> dNormedInputs = SelfAttention.Backward (dSaAdded, cache.saCache);

        // Backward through first RMSNorm and residual connection
        for (int i = 0; i < seqLen; i++) {
            double[] dSaOutput = dNormedInputs[i];

            // Backprop through residual connection
            for (int j = 0; j < EmbedSize; j++) {
                dInputs[i][j] += dSaAdded[i][j]; // Accumulate gradient from residual
                dInputs[i][j] += dSaOutput[j]; // Accumulate gradient from Self-Attention
            }

            // Backprop through first RMSNorm
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
    public List<double[]> normedInputs;
    public List<RMSNormCache> norm1Caches;
    public List<double[]> saOutputs;
    public SelfAttentionCache saCache;
    public List<double[]> saAdded;
    public List<double[]> normedSaAdded;
    public List<RMSNormCache> norm2Caches;
    public List<double[]> ffOutputs;
    public List<FeedForwardCache> ffCaches;
    public List<double[]> ffAdded;
}

// LlamaForCausalLM Model with backward pass
public class LlamaForCausalLM
{
    public Embedding TokenEmbedding;
    public List<TransformerBlock> TransformerBlocks;
    public RMSNorm FinalRMSNorm;
    public double[,] OutputProjection;

    public double[,] dOutputProjection;

    // For Adam optimizer
    public double[,] mOutputProjection;
    public double[,] vOutputProjection;

    public int VocabSize, EmbedSize, HiddenSize, NumHeads, NumLayers;

    private List<double[]> embeddings;
    private int[] inputTokens;
    private List<double[]> finalEmbeddings;
    private List<RMSNormCache> finalRMSNormCaches;
    private List<TransformerBlockCache> transformerCaches;

    private Random rand;

    public LlamaForCausalLM (int vocabSize, int embedSize, int hiddenSize, int numHeads, int numLayers, Random random) {
        rand = random;
        VocabSize = vocabSize;
        EmbedSize = embedSize;
        HiddenSize = hiddenSize;
        NumHeads = numHeads;
        NumLayers = numLayers;
        TokenEmbedding = new Embedding (vocabSize, embedSize, rand);
        TransformerBlocks = new List<TransformerBlock> ();
        for (int i = 0; i < numLayers; i++)
            TransformerBlocks.Add (new TransformerBlock (embedSize, hiddenSize, numHeads, rand));
        FinalRMSNorm = new RMSNorm (embedSize);
        OutputProjection = MathOps.InitializeMatrix (rand, VocabSize, EmbedSize);
        dOutputProjection = new double[VocabSize, EmbedSize];
        mOutputProjection = new double[VocabSize, EmbedSize];
        vOutputProjection = new double[VocabSize, EmbedSize];
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

        // Apply final RMS normalization to each token's embedding
        finalRMSNormCaches = new List<RMSNormCache> ();
        finalEmbeddings = new List<double[]> ();
        foreach (var e in embeddings) {
            (var finalEmbedding, var cache) = FinalRMSNorm.Forward (e);
            finalEmbeddings.Add (finalEmbedding);
            finalRMSNormCaches.Add (cache);
        }

        // Compute logits for each time step
        List<double[]> logitsList = new List<double[]> ();
        foreach (var finalEmbedding in finalEmbeddings) {
            double[] logits = new double[VocabSize];
            for (int i = 0; i < VocabSize; i++)
                logits[i] = MathOps.Dot (MathOps.GetRow (OutputProjection, i), finalEmbedding);
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

            // Backward through FinalRMSNorm
            double[] dEmbedding = FinalRMSNorm.Backward (dFinalEmbedding, finalRMSNormCaches[t]);

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

        // Update RMSNorm parameters
        UpdateRMSNorm (model.FinalRMSNorm);

        // Update TransformerBlocks
        foreach (var block in model.TransformerBlocks) {
            UpdateRMSNorm (block.Norm1);
            UpdateRMSNorm (block.Norm2);

            // Update SelfAttention parameters
            for (int h = 0; h < block.SelfAttention.NumHeads; h++) {
                UpdateParameters (block.SelfAttention.Wq[h], block.SelfAttention.dWq[h],
                    block.SelfAttention.mWq[h], block.SelfAttention.vWq[h]);
                UpdateParameters (block.SelfAttention.Wk[h], block.SelfAttention.dWk[h],
                    block.SelfAttention.mWk[h], block.SelfAttention.vWk[h]);
                UpdateParameters (block.SelfAttention.Wv[h], block.SelfAttention.dWv[h],
                    block.SelfAttention.mWv[h], block.SelfAttention.vWv[h]);

                ZeroGradients (block.SelfAttention.dWq[h]);
                ZeroGradients (block.SelfAttention.dWk[h]);
                ZeroGradients (block.SelfAttention.dWv[h]);
            }

            UpdateParameters (block.SelfAttention.Wo, block.SelfAttention.dWo,
                block.SelfAttention.mWo, block.SelfAttention.vWo);

            ZeroGradients (block.SelfAttention.dWo);

            // Update FeedForward parameters
            UpdateFeedForward (block.FeedForward);
        }
    }

    private void UpdateRMSNorm (RMSNorm rmsNorm) {
        double biasCorrection1 = 1 - Math.Pow (Beta1, timestep);
        double biasCorrection2 = 1 - Math.Pow (Beta2, timestep);

        // Compute global norm
        double globalNormGamma = ComputeGlobalNorm (rmsNorm.dGamma);

        double clipCoeffGamma = GradientClipValue / (globalNormGamma + 1e-6);
        if (clipCoeffGamma > 1.0) clipCoeffGamma = 1.0;

        for (int i = 0; i < rmsNorm.Size; i++) {
            // Apply global norm clipping
            double gradGamma = rmsNorm.dGamma[i] * clipCoeffGamma;

            // Update Gamma
            rmsNorm.mGamma[i] = Beta1 * rmsNorm.mGamma[i] + (1 - Beta1) * gradGamma;
            rmsNorm.vGamma[i] = Beta2 * rmsNorm.vGamma[i] + (1 - Beta2) * gradGamma * gradGamma;

            double mHatGamma = rmsNorm.mGamma[i] / biasCorrection1;
            double vHatGamma = rmsNorm.vGamma[i] / biasCorrection2;

            rmsNorm.Gamma[i] -= LearningRate * mHatGamma / (Math.Sqrt (vHatGamma) + Epsilon);

            rmsNorm.dGamma[i] = 0;
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

    private double ComputeGlobalNorm (double[] gradients) {
        double sum = 0.0;
        int length = gradients.Length;
        for (int i = 0; i < length; i++)
            sum += gradients[i] * gradients[i];
        return Math.Sqrt (sum);
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
        Array.Fill (gradients, 0);
    }
}
