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
            var accum = 0.0;
            for (int j = 0; j < cols; j++)
                accum += matrix[i, j] * vector[j];
            result[i] = accum;
        }

        return result;
    }

    public static double[] Softmax (double[] logits) {
        double maxLogit = logits.Max ();
        double[] expLogits = logits.Select (x => Math.Exp (x - maxLogit)).ToArray ();
        double sumExp = expLogits.Sum ();
        return expLogits.Select (x => x / sumExp).ToArray ();
    }

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

    public static void ZeroGradients (double[,] gradients) {
        Array.Clear (gradients, 0, gradients.Length);
    }

    public static void ZeroGradients (double[] gradients) {
        Array.Clear (gradients, 0, gradients.Length);
    }
}

// Embedding layer with backward pass and weight tying
public class Embedding
{
    public double[,] Weights;
    public double[,] Gradients;

    // For Adam optimizer
    public double[,] mWeights;
    public double[,] vWeights;

    public Embedding (int vocabSize, int embedSize, Random random) {
        Weights = MathOps.InitializeMatrix (random, vocabSize, embedSize);
        Gradients = new double[vocabSize, embedSize];
        mWeights = new double[vocabSize, embedSize];
        vWeights = new double[vocabSize, embedSize];
    }

    public double[] Forward (int token) {
        return MathOps.GetRow (Weights, token);
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
    public double[] dGamma;
    public double[] mGamma;
    public double[] vGamma;
    public int Size;

    public RMSNorm (int size) {
        Size = size;
        Gamma = new double[size];
        dGamma = new double[size];
        mGamma = new double[size];
        vGamma = new double[size];
        Array.Fill (Gamma, 1.0);
    }

    public (double[] output, RMSNormCache cache) Forward (double[] x) {
        double meanSquare = x.Select (val => val * val).Average ();
        double rms = Math.Sqrt (meanSquare + 1e-6);
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
        double gamma_over_rms = 0.0;
        for (int i = 0; i < Size; i++) {
            gamma_over_rms += gradOutput[i] * Gamma[i];
        }

        gamma_over_rms /= rms;

        for (int i = 0; i < Size; i++) {
            dx[i] = (gradOutput[i] * Gamma[i] / rms) - (x[i] * gamma_over_rms / (rms * rms));
        }

        return dx;
    }
}

public class RMSNormCache
{
    public double[] x_input;
    public double rms;
}

// Self-Attention mechanism with RoPE and backward pass
public class SelfAttention
{
    public double[][,] Wq;
    public double[][,] Wk, Wv;
    public double[,] Wo;

    public double[][,] dWq;
    public double[][,] dWk, dWv;
    public double[,] dWo;

    public double[][,] mWq, vWq;
    public double[][,] mWk, vWk;
    public double[][,] mWv, vWv;
    public double[,] mWo, vWo;

    public int EmbedSize, NumHeads, HeadDim, NumQueryGroups;
    public int HeadsPerGroup;

    public SelfAttention (int embedSize, int numHeads, int numQueryGroups, Random random) {
        EmbedSize = embedSize;
        NumHeads = numHeads;
        NumQueryGroups = numQueryGroups;
        HeadDim = embedSize / numHeads;

        HeadsPerGroup = NumHeads / NumQueryGroups;

        // Initialize Wq for each query group
        Wq = new double[NumQueryGroups][,];
        dWq = new double[NumQueryGroups][,];
        mWq = new double[NumQueryGroups][,];
        vWq = new double[NumQueryGroups][,];

        for (int g = 0; g < NumQueryGroups; g++) {
            int WqRows = HeadDim * HeadsPerGroup;
            Wq[g] = MathOps.InitializeMatrix (random, WqRows, EmbedSize);
            dWq[g] = new double[WqRows, EmbedSize];
            mWq[g] = new double[WqRows, EmbedSize];
            vWq[g] = new double[WqRows, EmbedSize];
        }

        // Initialize Wk and Wv for each head
        Wk = new double[NumHeads][,];
        Wv = new double[NumHeads][,];
        dWk = new double[NumHeads][,];
        dWv = new double[NumHeads][,];
        mWk = new double[NumHeads][,];
        vWk = new double[NumHeads][,];
        mWv = new double[NumHeads][,];
        vWv = new double[NumHeads][,];

        for (int h = 0; h < NumHeads; h++) {
            Wk[h] = MathOps.InitializeMatrix (random, HeadDim, EmbedSize);
            Wv[h] = MathOps.InitializeMatrix (random, HeadDim, EmbedSize);
            dWk[h] = new double[HeadDim, EmbedSize];
            dWv[h] = new double[HeadDim, EmbedSize];
            mWk[h] = new double[HeadDim, EmbedSize];
            vWk[h] = new double[HeadDim, EmbedSize];
            mWv[h] = new double[HeadDim, EmbedSize];
            vWv[h] = new double[HeadDim, EmbedSize];
        }

        Wo = MathOps.InitializeMatrix (random, EmbedSize, EmbedSize);
        dWo = new double[EmbedSize, EmbedSize];
        mWo = new double[EmbedSize, EmbedSize];
        vWo = new double[EmbedSize, EmbedSize];
    }

    public (List<double[]> outputs, SelfAttentionCache cache) Forward (List<double[]> inputs, int startPosition) {
        int seqLen = inputs.Count;

        var cache = new SelfAttentionCache {
            Inputs = inputs,
            Qs = new List<List<double[]>> (NumHeads),
            Ks = new List<List<double[]>> (NumHeads),
            Vs = new List<List<double[]>> (NumHeads),
            Zs = new List<double[]> (),
            Softmaxes = new List<List<double[]>> (NumHeads),
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

            // Compute Q for each query group
            for (int g = 0; g < NumQueryGroups; g++) {
                double[] qGroup = MathOps.MatrixVectorProduct (Wq[g], x);
                qGroup = ApplyRoPE (qGroup, t + startPosition);

                // Split qGroup into headsPerGroup query vectors
                for (int h = 0; h < HeadsPerGroup; h++) {
                    int headIndex = g * HeadsPerGroup + h;
                    int offset = h * HeadDim;
                    double[] qHead = new double[HeadDim];
                    Array.Copy (qGroup, offset, qHead, 0, HeadDim);
                    cache.Qs[headIndex].Add (qHead);
                }
            }

            // Compute K and V for each head
            for (int h = 0; h < NumHeads; h++) {
                double[] k = ApplyRoPE (MathOps.MatrixVectorProduct (Wk[h], x), t + startPosition);
                double[] v = MathOps.MatrixVectorProduct (Wv[h], x);

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
        List<double[]> dInputs = new List<double[]> (seqLen);
        for (int i = 0; i < seqLen; i++)
            dInputs.Add (new double[EmbedSize]);

        double[,] dWo = new double[EmbedSize, EmbedSize];
        double[][] dConcatHeads = new double[seqLen][];
        for (int t = 0; t < seqLen; t++)
            dConcatHeads[t] = new double[EmbedSize];

        // Backprop through output projection
        for (int t = 0; t < seqLen; t++) {
            double[] gradOutput = gradOutputs[t];
            double[] concatHeads = cache.Zs[t];

            // Gradient w.r.t. concatHeads
            for (int i = 0; i < EmbedSize; i++) {
                for (int j = 0; j < EmbedSize; j++) {
                    double grad = Wo[i, j] * gradOutput[i];
                    dWo[i, j] += gradOutput[i] * concatHeads[j];
                    dConcatHeads[t][j] += grad;
                }
            }
        }

        // Initialize per-head gradients
        List<double[][]> perHead_dQs = new List<double[][]> (NumHeads);
        List<double[][]> perHead_dKs = new List<double[][]> (NumHeads);
        List<double[][]> perHead_dVs = new List<double[][]> (NumHeads);

        for (int h = 0; h < NumHeads; h++) {
            perHead_dQs.Add (new double[seqLen][]);
            perHead_dKs.Add (new double[seqLen][]);
            perHead_dVs.Add (new double[seqLen][]);
            for (int t = 0; t < seqLen; t++) {
                perHead_dQs[h][t] = new double[HeadDim];
                perHead_dKs[h][t] = new double[HeadDim];
                perHead_dVs[h][t] = new double[HeadDim];
            }
        }

        // Backprop through attention heads
        for (int h = 0; h < NumHeads; h++) {
            // Accumulate gradients over time
            for (int t = seqLen - 1; t >= 0; t--) {
                double[] dZ = new double[HeadDim];
                Array.Copy (dConcatHeads[t], h * HeadDim, dZ, 0, HeadDim);

                // Backprop through weighted sum of Vs
                double[] dAlpha = new double[t + 1];
                for (int j = 0; j <= t; j++) {
                    for (int k = 0; k < HeadDim; k++) {
                        perHead_dVs[h][j][k] += cache.Softmaxes[h][t][j] * dZ[k];
                    }

                    dAlpha[j] += MathOps.Dot (cache.Vs[h][j], dZ);
                }

                // Backprop through softmax
                double[] dScores = new double[t + 1];
                double sum = 0.0;
                for (int k = 0; k <= t; k++)
                    sum += cache.Softmaxes[h][t][k] * dAlpha[k];

                for (int j = 0; j <= t; j++) {
                    dScores[j] = cache.Softmaxes[h][t][j] * (dAlpha[j] - sum);
                }

                // Backprop through attention scores
                double scale = 1.0 / Math.Sqrt (HeadDim);
                for (int j = 0; j <= t; j++) {
                    for (int i = 0; i < HeadDim; i++) {
                        perHead_dQs[h][t][i] += dScores[j] * cache.Ks[h][j][i] * scale;
                        perHead_dKs[h][j][i] += dScores[j] * cache.Qs[h][t][i] * scale;
                    }
                }
            }

            // Backprop through RoPE embeddings
            for (int t = 0; t < seqLen; t++) {
                int position = t + cache.StartPosition;

                perHead_dQs[h][t] = BackwardRoPE (perHead_dQs[h][t], cache.Qs[h][t], position);
                perHead_dKs[h][t] = BackwardRoPE (perHead_dKs[h][t], cache.Ks[h][t], position);
            }
        }

        // Group-level accumulation for Wq and dInputs
        for (int g = 0; g < NumQueryGroups; g++) {
            int startHead = g * HeadsPerGroup;
            int endHead = startHead + HeadsPerGroup;

            // Initialize dQGroup for the group
            double[][] dQGroup = new double[seqLen][];
            for (int t = 0; t < seqLen; t++) {
                dQGroup[t] = new double[HeadDim * HeadsPerGroup];
            }

            // Concatenate perHead_dQs into dQGroup
            for (int h = startHead; h < endHead; h++) {
                for (int t = 0; t < seqLen; t++) {
                    int offset = (h - startHead) * HeadDim;
                    Array.Copy (perHead_dQs[h][t], 0, dQGroup[t], offset, HeadDim);
                }
            }

            // Update Wq and dInputs using dQGroup
            for (int t = 0; t < seqLen; t++) {
                double[] x = cache.Inputs[t];

                // Accumulate gradients for Wq
                for (int i = 0; i < HeadDim * HeadsPerGroup; i++) {
                    for (int j = 0; j < EmbedSize; j++) {
                        dWq[g][i, j] += dQGroup[t][i] * x[j];
                    }
                }

                // Update dInputs
                for (int j = 0; j < EmbedSize; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < HeadDim * HeadsPerGroup; i++) {
                        sum += Wq[g][i, j] * dQGroup[t][i];
                    }

                    dInputs[t][j] += sum;
                }
            }

            // Zero gradients for next iteration
            MathOps.ZeroGradients (dWq[g]);
        }

        // Backprop through Wk, Wv, and accumulate dInputs
        for (int h = 0; h < NumHeads; h++) {
            for (int t = 0; t < seqLen; t++) {
                double[] x = cache.Inputs[t];

                // Gradients w.r.t. Wk and Wv
                for (int i = 0; i < HeadDim; i++) {
                    for (int j = 0; j < EmbedSize; j++) {
                        dWk[h][i, j] += perHead_dKs[h][t][i] * x[j];
                        dInputs[t][j] += Wk[h][i, j] * perHead_dKs[h][t][i];

                        dWv[h][i, j] += perHead_dVs[h][t][i] * x[j];
                        dInputs[t][j] += Wv[h][i, j] * perHead_dVs[h][t][i];
                    }
                }
            }

            // Accumulate gradients for Wk and Wv
            for (int i = 0; i < HeadDim; i++) {
                for (int j = 0; j < EmbedSize; j++) {
                    this.dWk[h][i, j] += dWk[h][i, j];
                    this.dWv[h][i, j] += dWv[h][i, j];
                }
            }

            // Zero gradients for next iteration
            MathOps.ZeroGradients (dWk[h]);
            MathOps.ZeroGradients (dWv[h]);
        }

        // Accumulate gradients for Wo
        for (int i = 0; i < EmbedSize; i++) {
            for (int j = 0; j < EmbedSize; j++) {
                this.dWo[i, j] += dWo[i, j];
            }
        }

        MathOps.ZeroGradients (dWo);

        return dInputs;
    }

    private double[] ApplyRoPE (double[] x, int position) {
        int halfDim = x.Length / 2;
        double[] result = new double[x.Length];
        double theta = 10000;

        for (int i = 0; i < halfDim; i++) {
            double angle = position * Math.Pow (theta, -2.0 * i / x.Length);
            double cos = Math.Cos (angle);
            double sin = Math.Sin (angle);

            result[2 * i] = x[2 * i] * cos - x[2 * i + 1] * sin;
            result[2 * i + 1] = x[2 * i] * sin + x[2 * i + 1] * cos;
        }

        return result;
    }

    private double[] BackwardRoPE (double[] grad, double[] x, int position) {
        int halfDim = x.Length / 2;
        double[] dx = new double[x.Length];
        double theta = 10000;

        for (int i = 0; i < halfDim; i++) {
            double angle = position * Math.Pow (theta, -2.0 * i / x.Length);
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

    public double[,] dW1, dW2;
    public double[] dB1, dB2;

    public double[,] mW1, vW1;
    public double[,] mW2, vW2;
    public double[] mB1, vB1;
    public double[] mB2, vB2;

    public int EmbedSize, HiddenSize;

    public FeedForward (int embedSize, int hiddenSize, Random random) {
        EmbedSize = embedSize;
        HiddenSize = hiddenSize;

        W1 = MathOps.InitializeMatrix (random, HiddenSize * 2, EmbedSize);
        W2 = MathOps.InitializeMatrix (random, EmbedSize, HiddenSize);
        B1 = new double[HiddenSize * 2];
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
            double swish_grad = sigmoid + cache.h_b[i] * sigmoid * (1 - sigmoid);

            dh_a[i] = dh_swiglu[i] * swish;
            dh_b[i] = dh_swiglu[i] * cache.h_a[i] * swish_grad;
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

    public TransformerBlock (int embedSize, int hiddenSize, int numHeads, int numQueryGroups, Random random) {
        EmbedSize = embedSize;
        Norm1 = new RMSNorm (embedSize);
        Norm2 = new RMSNorm (embedSize);
        SelfAttention = new SelfAttention (embedSize, numHeads, numQueryGroups, random);
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

        // Apply first RMSNorm
        foreach (var x in inputs) {
            var (normedInput, norm1Cache) = Norm1.Forward (x);
            cache.normedInputs.Add (normedInput);
            cache.norm1Caches.Add (norm1Cache);
        }

        // Apply Self-Attention
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

        // Initialize gradients for intermediate variables
        List<double[]> dSaAdded = new List<double[]> ();

        // Backward through FeedForward and second residual connection
        for (int i = 0; i < seqLen; i++) {
            double[] dFfAdded = gradOutputs[i];

            // Split gradient at the second residual connection
            double[] dFfOutput = dFfAdded.ToArray (); // For feedforward
            double[] dSaAdd = dFfAdded.ToArray (); // For residual connection

            // Backprop through feed-forward layer
            double[] dNormedSaAdded = FeedForward.Backward (dFfOutput, cache.ffCaches[i]);

            // Backprop through second RMSNorm
            double[] dSaAddedNorm = Norm2.Backward (dNormedSaAdded, cache.norm2Caches[i]);

            // Accumulate gradients
            for (int j = 0; j < EmbedSize; j++)
                dSaAdd[j] += dSaAddedNorm[j];

            dSaAdded.Add (dSaAdd);
        }

        // Backward through Self-Attention and first residual connection
        List<double[]> dNormedInputs = SelfAttention.Backward (dSaAdded, cache.saCache);

        for (int i = 0; i < seqLen; i++) {
            double[] dSaOutput = dNormedInputs[i];

            // Backprop through first RMSNorm
            double[] dNormedInput = Norm1.Backward (dSaOutput, cache.norm1Caches[i]);

            // Accumulate gradients
            for (int j = 0; j < EmbedSize; j++)
                dInputs[i][j] += dNormedInput[j] + dSaAdded[i][j]; // Correct accumulation
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

// LlamaForCausalLM Model with backward pass and weight tying
public class LlamaForCausalLM
{
    public Embedding TokenEmbedding;
    public List<TransformerBlock> TransformerBlocks;
    public RMSNorm FinalRMSNorm;

    public int VocabSize, EmbedSize, HiddenSize, NumHeads, NumLayers, NumQueryGroups;

    private List<double[]> embeddings;
    private int[] inputTokens;
    private List<double[]> finalEmbeddings;
    private List<RMSNormCache> finalRMSNormCaches;
    private List<TransformerBlockCache> transformerCaches;

    public LlamaForCausalLM (int vocabSize, int embedSize, int hiddenSize, int numHeads, int numLayers, int numQueryGroups, Random random) {
        VocabSize = vocabSize;
        EmbedSize = embedSize;
        HiddenSize = hiddenSize;
        NumHeads = numHeads;
        NumLayers = numLayers;
        NumQueryGroups = numQueryGroups;
        TokenEmbedding = new Embedding (vocabSize, embedSize, random);
        TransformerBlocks = new List<TransformerBlock> ();
        for (int i = 0; i < numLayers; i++)
            TransformerBlocks.Add (new TransformerBlock (embedSize, hiddenSize, numHeads, numQueryGroups, random));
        FinalRMSNorm = new RMSNorm (embedSize);
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

        // Compute logits for each time step using weight tying
        List<double[]> logitsList = new List<double[]> ();
        foreach (var finalEmbedding in finalEmbeddings) {
            double[] logits = new double[VocabSize];
            for (int i = 0; i < VocabSize; i++)
                logits[i] = MathOps.Dot (MathOps.GetRow (TokenEmbedding.Weights, i), finalEmbedding); // Weight tying
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
            var finalEmbedding = finalEmbeddings[t];

            // Gradients w.r.t. TokenEmbedding.Weights (due to weight tying)
            for (int i = 0; i < VocabSize; i++)
            for (int j = 0; j < EmbedSize; j++)
                TokenEmbedding.Gradients[i, j] += dLogitsList[t][i] * finalEmbedding[j];

            // Gradients w.r.t. final embeddings
            double[] dFinalEmbedding = new double[EmbedSize];
            for (int j = 0; j < EmbedSize; j++)
            for (int i = 0; i < VocabSize; i++)
                dFinalEmbedding[j] += TokenEmbedding.Weights[i, j] * dLogitsList[t][i];

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
        for (int i = 0; i < inputTokens.Length; i++)
            TokenEmbedding.Backward (inputTokens[i], dEmbeddings[i]);
    }
}

// Loss Function with gradient
public static class LossFunctions
{
    public static double CrossEntropyLoss (double[] logits, int targetToken, out double[] dLogits) {
        double[] probabilities = MathOps.Softmax (logits);
        dLogits = probabilities.ToArray ();
        dLogits[targetToken] -= 1.0; // Gradient of softmax cross-entropy
        return -Math.Log (probabilities[targetToken] + 1e-12);
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
    public double GradientClipValue;

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
        MathOps.ZeroGradients (model.TokenEmbedding.Gradients);

        // Update RMSNorm parameters
        UpdateRMSNorm (model.FinalRMSNorm);

        // Update TransformerBlocks
        foreach (var block in model.TransformerBlocks) {
            UpdateRMSNorm (block.Norm1);
            UpdateRMSNorm (block.Norm2);

            // Update SelfAttention parameters
            for (int g = 0; g < block.SelfAttention.NumQueryGroups; g++) {
                UpdateParameters (block.SelfAttention.Wq[g], block.SelfAttention.dWq[g],
                    block.SelfAttention.mWq[g], block.SelfAttention.vWq[g]);
                MathOps.ZeroGradients (block.SelfAttention.dWq[g]);
            }

            UpdateParameters (block.SelfAttention.Wo, block.SelfAttention.dWo,
                block.SelfAttention.mWo, block.SelfAttention.vWo);
            MathOps.ZeroGradients (block.SelfAttention.dWo);

            for (int h = 0; h < block.SelfAttention.NumHeads; h++) {
                UpdateParameters (block.SelfAttention.Wk[h], block.SelfAttention.dWk[h],
                    block.SelfAttention.mWk[h], block.SelfAttention.vWk[h]);
                MathOps.ZeroGradients (block.SelfAttention.dWk[h]);

                UpdateParameters (block.SelfAttention.Wv[h], block.SelfAttention.dWv[h],
                    block.SelfAttention.mWv[h], block.SelfAttention.vWv[h]);
                MathOps.ZeroGradients (block.SelfAttention.dWv[h]);
            }

            // Update FeedForward parameters
            UpdateFeedForward (block.FeedForward);
        }
    }

    private void UpdateRMSNorm (RMSNorm rmsNorm) {
        double biasCorrection1 = 1 - Math.Pow (Beta1, timestep);
        double biasCorrection2 = 1 - Math.Pow (Beta2, timestep);

        for (int i = 0; i < rmsNorm.Size; i++) {
            double gradGamma = rmsNorm.dGamma[i];

            rmsNorm.mGamma[i] = Beta1 * rmsNorm.mGamma[i] + (1 - Beta1) * gradGamma;
            rmsNorm.vGamma[i] = Beta2 * rmsNorm.vGamma[i] + (1 - Beta2) * gradGamma * gradGamma;

            double mHatGamma = rmsNorm.mGamma[i] / biasCorrection1;
            double vHatGamma = rmsNorm.vGamma[i] / biasCorrection2;

            // Apply gradient clipping
            double update = LearningRate * mHatGamma / (Math.Sqrt (vHatGamma) + Epsilon);
            update = Math.Min (Math.Max (update, -GradientClipValue), GradientClipValue);

            rmsNorm.Gamma[i] -= update;

            rmsNorm.dGamma[i] = 0.0;
        }
    }

    private void UpdateFeedForward (FeedForward feedForward) {
        UpdateParameters (feedForward.W1, feedForward.dW1, feedForward.mW1, feedForward.vW1);
        UpdateParameters (feedForward.B1, feedForward.dB1, feedForward.mB1, feedForward.vB1);
        MathOps.ZeroGradients (feedForward.dW1);
        MathOps.ZeroGradients (feedForward.dB1);

        UpdateParameters (feedForward.W2, feedForward.dW2, feedForward.mW2, feedForward.vW2);
        UpdateParameters (feedForward.B2, feedForward.dB2, feedForward.mB2, feedForward.vB2);
        MathOps.ZeroGradients (feedForward.dW2);
        MathOps.ZeroGradients (feedForward.dB2);
    }

    private void UpdateParameters (double[,] weights, double[,] gradients, double[,] m, double[,] v) {
        int rows = weights.GetLength (0);
        int cols = weights.GetLength (1);
        double biasCorrection1 = 1 - Math.Pow (Beta1, timestep);
        double biasCorrection2 = 1 - Math.Pow (Beta2, timestep);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double grad = gradients[i, j];

                m[i, j] = Beta1 * m[i, j] + (1 - Beta1) * grad;
                v[i, j] = Beta2 * v[i, j] + (1 - Beta2) * grad * grad;

                double mHat = m[i, j] / biasCorrection1;
                double vHat = v[i, j] / biasCorrection2;

                // Apply gradient clipping
                double update = LearningRate * mHat / (Math.Sqrt (vHat) + Epsilon);
                update = Math.Min (Math.Max (update, -GradientClipValue), GradientClipValue);

                weights[i, j] -= update;
            }
        }
    }

    private void UpdateParameters (double[] weights, double[] gradients, double[] m, double[] v) {
        int length = weights.Length;
        double biasCorrection1 = 1 - Math.Pow (Beta1, timestep);
        double biasCorrection2 = 1 - Math.Pow (Beta2, timestep);

        for (int i = 0; i < length; i++) {
            double grad = gradients[i];

            m[i] = Beta1 * m[i] + (1 - Beta1) * grad;
            v[i] = Beta2 * v[i] + (1 - Beta2) * grad * grad;

            double mHat = m[i] / biasCorrection1;
            double vHat = v[i] / biasCorrection2;

            // Apply gradient clipping
            double update = LearningRate * mHat / (Math.Sqrt (vHat) + Epsilon);
            update = Math.Min (Math.Max (update, -GradientClipValue), GradientClipValue);

            weights[i] -= update;
        }
    }
}
