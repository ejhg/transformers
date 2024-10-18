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

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double delta = (i == j) ? 1.0 : 0.0;
                dScores[i] += softmax[i] * (delta - softmax[j]) * dLoss_dSoftmax[j];
            }
        }

        return dScores;
    }
}

// Embedding layer with backward pass
public class Embedding
{
    public double[,] Weights;
    public double[,] Gradients;
    private Random rand;

    public Embedding (int vocabSize, int embedSize, Random random) {
        rand = random;
        Weights = new double[vocabSize, embedSize];
        Gradients = new double[vocabSize, embedSize];
        for (int i = 0; i < vocabSize; i++)
        for (int j = 0; j < embedSize; j++)
            Weights[i, j] = rand.NextDouble () * 0.02 - 0.01;
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
    private double[] x_input;
    private double mean;
    private double variance;
    private double[] normalized;

    public double[] dGamma;
    public double[] dBeta;

    public LayerNorm (int size) {
        Size = size;
        Gamma = new double[size];
        Beta = new double[size];
        dGamma = new double[size];
        dBeta = new double[size];
        for (int i = 0; i < size; i++) {
            Gamma[i] = 1.0;
            Beta[i] = 0.0;
        }
    }

    public double[] Forward (double[] x) {
        x_input = x;
        mean = x.Average ();
        variance = x.Select (val => Math.Pow (val - mean, 2)).Average ();
        normalized = x.Select (val => (val - mean) / Math.Sqrt (variance + 1e-5)).ToArray ();
        double[] output = new double[Size];
        for (int i = 0; i < Size; i++)
            output[i] = Gamma[i] * normalized[i] + Beta[i];
        return output;
    }

    public double[] Backward (double[] gradOutput) {
        double[] dxhat = new double[Size];
        for (int i = 0; i < Size; i++) {
            dGamma[i] += gradOutput[i] * normalized[i];
            dBeta[i] += gradOutput[i];
            dxhat[i] = gradOutput[i] * Gamma[i];
        }

        double stdInv = 1.0 / Math.Sqrt (variance + 1e-5);
        double[] dx = new double[Size];
        double dvar = -0.5 * stdInv * stdInv * stdInv * dxhat.Select ((dxh, i) => (x_input[i] - mean) * dxh).Sum ();
        double dmean = -stdInv * dxhat.Sum () + dvar * (-2.0 / Size) * (x_input.Sum () - Size * mean);
        for (int i = 0; i < Size; i++)
            dx[i] = stdInv * dxhat[i] + dvar * 2.0 * (x_input[i] - mean) / Size + dmean / Size;
        return dx;
    }
}

// Self-Attention mechanism with backward pass
public class SelfAttention
{
    public double[,] Wq, Wk, Wv, Wo;
    public double[,] dWq, dWk, dWv, dWo;
    public int EmbedSize, HeadSize;

    private List<double[]> Qs, Ks, Vs, Zs;
    private List<double[]> Inputs;
    private List<double[]> Softmaxes;

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
    }

    private double[,] InitializeMatrix (int rows, int cols) {
        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble () * 0.02 - 0.01;
        return matrix;
    }

    public List<double[]> Forward (List<double[]> inputs) {
        Inputs = inputs;
        int seqLen = inputs.Count;
        Qs = new List<double[]> ();
        Ks = new List<double[]> ();
        Vs = new List<double[]> ();
        Zs = new List<double[]> ();
        Softmaxes = new List<double[]> ();

        // Compute Qs, Ks, Vs
        foreach (var x in inputs) {
            Qs.Add (MathOps.MatrixVectorProduct (Wq, x));
            Ks.Add (MathOps.MatrixVectorProduct (Wk, x));
            Vs.Add (MathOps.MatrixVectorProduct (Wv, x));
        }

        // Compute attention outputs
        for (int i = 0; i < seqLen; i++) {
            // Compute attention scores up to position i
            double[] scores = new double[i + 1];
            for (int j = 0; j <= i; j++)
                scores[j] = MathOps.Dot (Qs[i], Ks[j]) / Math.Sqrt (HeadSize);

            // Apply softmax to get attention weights
            double[] softmax = MathOps.Softmax (scores);
            Softmaxes.Add (softmax);

            // Compute weighted sum of Vs
            double[] z = new double[HeadSize];
            for (int j = 0; j <= i; j++)
            for (int k = 0; k < HeadSize; k++)
                z[k] += softmax[j] * Vs[j][k];
            Zs.Add (z);
        }

        // Output projection
        List<double[]> outputs = new List<double[]> ();
        for (int i = 0; i < seqLen; i++) {
            double[] output = MathOps.MatrixVectorProduct (Wo, Zs[i]);
            outputs.Add (output);
        }

        return outputs;
    }

    public List<double[]> Backward (List<double[]> gradOutputs) {
        int seqLen = Inputs.Count;
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
                    dWo[k, h] += gradOutputs[i][k] * Zs[i][h];
                }
            }

            // Backpropagate through attention output z_i = sum_j alpha_{ij} * V_j
            double[] dAlpha = new double[i + 1];
            for (int j = 0; j <= i; j++) {
                for (int h = 0; h < HeadSize; h++) {
                    dVs[j][h] += Softmaxes[i][j] * dZ[h];
                }

                dAlpha[j] = MathOps.Dot (Vs[j], dZ);
            }

            // Backpropagate through softmax
            double[] dScores = MathOps.SoftmaxGradient (Softmaxes[i], dAlpha);

            // Backpropagate through attention scores s_{ij} = Q_i ⋅ K_j / sqrt(HeadSize)
            double scale = 1.0 / Math.Sqrt (HeadSize);
            for (int j = 0; j <= i; j++) {
                for (int h = 0; h < HeadSize; h++) {
                    dQs[i][h] += dScores[j] * Ks[j][h] * scale;
                    dKs[j][h] += dScores[j] * Qs[i][h] * scale;
                }
            }
        }

        // Backpropagate through linear layers Wq, Wk, Wv
        for (int i = 0; i < seqLen; i++) {
            // Gradients w.r.t. Wq and input x_i
            for (int h = 0; h < HeadSize; h++) {
                for (int j = 0; j < EmbedSize; j++) {
                    dWq[h, j] += dQs[i][h] * Inputs[i][j];
                    dInputs[i][j] += Wq[h, j] * dQs[i][h];
                }
            }

            // Gradients w.r.t. Wk and input x_i
            for (int h = 0; h < HeadSize; h++) {
                for (int j = 0; j < EmbedSize; j++) {
                    dWk[h, j] += dKs[i][h] * Inputs[i][j];
                    dInputs[i][j] += Wk[h, j] * dKs[i][h];
                }
            }

            // Gradients w.r.t. Wv and input x_i
            for (int h = 0; h < HeadSize; h++) {
                for (int j = 0; j < EmbedSize; j++) {
                    dWv[h, j] += dVs[i][h] * Inputs[i][j];
                    dInputs[i][j] += Wv[h, j] * dVs[i][h];
                }
            }
        }

        return dInputs;
    }
}

// Feed-Forward Network with backward pass
public class FeedForward
{
    public double[,] W1, W2;
    public double[] B1, B2;
    public int EmbedSize, HiddenSize;
    private double[] x_input, h_linear, h_relu;
    public double[,] dW1, dW2;
    public double[] dB1, dB2;
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
    }

    private double[,] InitializeMatrix (int rows, int cols) {
        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble () * 0.02 - 0.01;
        return matrix;
    }

    public double[] Forward (double[] x) {
        x_input = x;
        h_linear = new double[HiddenSize];
        h_relu = new double[HiddenSize];

        // First layer
        for (int i = 0; i < HiddenSize; i++) {
            h_linear[i] = B1[i];
            for (int j = 0; j < EmbedSize; j++)
                h_linear[i] += W1[i, j] * x[j];
            h_relu[i] = Math.Max (0, h_linear[i]);
        }

        // Second layer
        double[] y = new double[EmbedSize];
        for (int i = 0; i < EmbedSize; i++) {
            y[i] = B2[i];
            for (int j = 0; j < HiddenSize; j++)
                y[i] += W2[i, j] * h_relu[j];
        }

        return y;
    }

    public double[] Backward (double[] dOut) {
        double[] dh_relu = new double[HiddenSize];

        // Backprop through second layer
        for (int i = 0; i < EmbedSize; i++) {
            dB2[i] += dOut[i];
            for (int j = 0; j < HiddenSize; j++) {
                dW2[i, j] += dOut[i] * h_relu[j];
                dh_relu[j] += dOut[i] * W2[i, j];
            }
        }

        // Backprop through ReLU
        double[] dh_linear = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++)
            dh_linear[i] = h_linear[i] > 0 ? dh_relu[i] : 0;

        // Backprop through first layer
        double[] dx = new double[EmbedSize];
        for (int i = 0; i < HiddenSize; i++) {
            dB1[i] += dh_linear[i];
            for (int j = 0; j < EmbedSize; j++) {
                dW1[i, j] += dh_linear[i] * x_input[j];
                dx[j] += dh_linear[i] * W1[i, j];
            }
        }

        return dx;
    }
}

// Transformer Block with backward pass
public class TransformerBlock
{
    public LayerNorm Norm1, Norm2;
    public SelfAttention SelfAttention;
    public FeedForward FeedForward;
    public int EmbedSize;

    private List<double[]> inputs;
    private List<double[]> normedInputs;
    private List<double[]> saOutputs;
    private List<double[]> saAdded;
    private List<double[]> normedSaAdded;
    private List<double[]> ffOutputs;
    private List<double[]> ffAdded;

    public TransformerBlock (int embedSize, int hiddenSize, int headSize, Random random) {
        EmbedSize = embedSize;
        Norm1 = new LayerNorm (embedSize);
        Norm2 = new LayerNorm (embedSize);
        SelfAttention = new SelfAttention (embedSize, headSize, random);
        FeedForward = new FeedForward (embedSize, hiddenSize, random);
    }

    public List<double[]> Forward (List<double[]> inputs) {
        this.inputs = inputs;
        int seqLen = inputs.Count;
        normedInputs = new List<double[]> ();
        saOutputs = new List<double[]> ();
        saAdded = new List<double[]> ();
        normedSaAdded = new List<double[]> ();
        ffOutputs = new List<double[]> ();
        ffAdded = new List<double[]> ();

        // Apply first LayerNorm and Self-Attention
        foreach (var x in inputs) {
            var normedInput = Norm1.Forward (x);
            normedInputs.Add (normedInput);
        }

        saOutputs = SelfAttention.Forward (normedInputs);

        // Add & Norm
        for (int i = 0; i < seqLen; i++) {
            var saAdd = MathOps.Add (inputs[i], saOutputs[i]);
            saAdded.Add (saAdd);

            var normedSaAdd = Norm2.Forward (saAdd);
            normedSaAdded.Add (normedSaAdd);

            var ffOutput = FeedForward.Forward (normedSaAdd);
            ffOutputs.Add (ffOutput);

            var ffAdd = MathOps.Add (saAdd, ffOutput);
            ffAdded.Add (ffAdd);
        }

        return ffAdded;
    }

    public List<double[]> Backward (List<double[]> gradOutputs) {
        int seqLen = gradOutputs.Count;
        List<double[]> dInputs = new List<double[]> ();
        for (int i = 0; i < seqLen; i++)
            dInputs.Add (new double[EmbedSize]);

        List<double[]> dSaAdded = new List<double[]> ();
        for (int i = 0; i < seqLen; i++)
            dSaAdded.Add (new double[EmbedSize]);

        // Backward through FeedForward and second residual connection
        List<double[]> dNormedSaAdded = new List<double[]> ();
        for (int i = 0; i < seqLen; i++) {
            // Gradient w.r.t. the output of the feed-forward layer
            double[] dFfAdded = gradOutputs[i];

            // Backprop through residual connection
            double[] dFfOutput = new double[EmbedSize];
            double[] dSaAdd = new double[EmbedSize];
            for (int j = 0; j < EmbedSize; j++) {
                dFfOutput[j] = dFfAdded[j];
                dSaAdd[j] += dFfAdded[j]; // Accumulate gradient from the residual path
            }

            // Backprop through feed-forward layer
            double[] dNormedSaAddedSingle = FeedForward.Backward (dFfOutput);
            dNormedSaAdded.Add (dNormedSaAddedSingle);

            // Accumulate gradients
            for (int j = 0; j < EmbedSize; j++)
                dSaAdd[j] += dNormedSaAddedSingle[j];

            // Store gradient for saAdd
            dSaAdded[i] = dSaAdd;
        }

        // Backward through second LayerNorm
        List<double[]> dSaAddedNorm = new List<double[]> ();
        for (int i = 0; i < seqLen; i++) {
            dSaAddedNorm.Add (Norm2.Backward (dSaAdded[i]));
        }

        // Backward through Self-Attention and first residual connection
        List<double[]> dNormedInputs = SelfAttention.Backward (dSaAddedNorm);

        for (int i = 0; i < seqLen; i++) {
            // Backprop through first residual connection
            for (int j = 0; j < EmbedSize; j++) {
                dInputs[i][j] += dSaAddedNorm[i][j];
            }

            // Backprop through first LayerNorm
            double[] dInputNorm = Norm1.Backward (dNormedInputs[i]);

            // Accumulate gradients
            for (int j = 0; j < EmbedSize; j++) {
                dInputs[i][j] += dInputNorm[j];
            }
        }

        return dInputs;
    }
}

// LlamaForCausalLM Model with backward pass
public class LlamaForCausalLM
{
    public Embedding TokenEmbedding;
    public List<TransformerBlock> TransformerBlocks;
    public LayerNorm FinalLayerNorm;
    public double[,] OutputProjection;
    public double[,] dOutputProjection;
    public int VocabSize, EmbedSize, HiddenSize, HeadSize, NumLayers;

    private List<double[]> embeddings;
    private int[] inputTokens;

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
    }

    private double[,] InitializeMatrix (int rows, int cols) {
        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble () * 0.02 - 0.01;
        return matrix;
    }

    public List<double[]> Forward (int[] inputTokens) {
        this.inputTokens = inputTokens;
        embeddings = inputTokens.Select (token => TokenEmbedding.Forward (token)).ToList ();

        // Process through transformer blocks
        foreach (var block in TransformerBlocks) {
            embeddings = block.Forward (embeddings);
        }

        // Apply final layer normalization to each token's embedding
        List<double[]> finalEmbeddings = embeddings.Select (e => FinalLayerNorm.Forward (e)).ToList ();

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
            double[] finalEmbedding = FinalLayerNorm.Forward (embeddings[t]);

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
            double[] dEmbedding = FinalLayerNorm.Backward (dFinalEmbedding);

            // Accumulate gradients
            for (int j = 0; j < EmbedSize; j++)
                dEmbeddings[t][j] += dEmbedding[j];
        }

        // Backward through TransformerBlocks
        for (int i = TransformerBlocks.Count - 1; i >= 0; i--) {
            var block = TransformerBlocks[i];
            dEmbeddings = block.Backward (dEmbeddings);
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

// Optimizer updated to handle all parameters
public class SGDOptimizer
{
    public double LearningRate;

    public SGDOptimizer (double learningRate) {
        LearningRate = learningRate;
    }

    public void Step (LlamaForCausalLM model) {
        // Update TokenEmbedding weights
        for (int i = 0; i < model.TokenEmbedding.Weights.GetLength (0); i++)
        for (int j = 0; j < model.TokenEmbedding.Weights.GetLength (1); j++) {
            model.TokenEmbedding.Weights[i, j] -= LearningRate * model.TokenEmbedding.Gradients[i, j];
            model.TokenEmbedding.Gradients[i, j] = 0; // Reset gradients
        }

        // Update OutputProjection
        for (int i = 0; i < model.VocabSize; i++)
        for (int j = 0; j < model.EmbedSize; j++) {
            model.OutputProjection[i, j] -= LearningRate * model.dOutputProjection[i, j];
            model.dOutputProjection[i, j] = 0;
        }

        // Update LayerNorm parameters
        UpdateLayerNorm (model.FinalLayerNorm);

        // Update TransformerBlocks
        foreach (var block in model.TransformerBlocks) {
            UpdateLayerNorm (block.Norm1);
            UpdateLayerNorm (block.Norm2);

            // Update SelfAttention parameters
            UpdateMatrix (block.SelfAttention.Wq, block.SelfAttention.dWq);
            UpdateMatrix (block.SelfAttention.Wk, block.SelfAttention.dWk);
            UpdateMatrix (block.SelfAttention.Wv, block.SelfAttention.dWv);
            UpdateMatrix (block.SelfAttention.Wo, block.SelfAttention.dWo);

            // Reset gradients
            ZeroMatrix (block.SelfAttention.dWq);
            ZeroMatrix (block.SelfAttention.dWk);
            ZeroMatrix (block.SelfAttention.dWv);
            ZeroMatrix (block.SelfAttention.dWo);

            // Update FeedForward parameters
            UpdateFeedForward (block.FeedForward);
        }
    }

    private void UpdateLayerNorm (LayerNorm layerNorm) {
        for (int i = 0; i < layerNorm.Size; i++) {
            layerNorm.Gamma[i] -= LearningRate * layerNorm.dGamma[i];
            layerNorm.Beta[i] -= LearningRate * layerNorm.dBeta[i];
            layerNorm.dGamma[i] = 0;
            layerNorm.dBeta[i] = 0;
        }
    }

    private void UpdateFeedForward (FeedForward feedForward) {
        // Update W1 and B1
        for (int i = 0; i < feedForward.HiddenSize; i++) {
            feedForward.B1[i] -= LearningRate * feedForward.dB1[i];
            for (int j = 0; j < feedForward.EmbedSize; j++) {
                feedForward.W1[i, j] -= LearningRate * feedForward.dW1[i, j];
                feedForward.dW1[i, j] = 0;
            }

            feedForward.dB1[i] = 0;
        }

        // Update W2 and B2
        for (int i = 0; i < feedForward.EmbedSize; i++) {
            feedForward.B2[i] -= LearningRate * feedForward.dB2[i];
            for (int j = 0; j < feedForward.HiddenSize; j++) {
                feedForward.W2[i, j] -= LearningRate * feedForward.dW2[i, j];
                feedForward.dW2[i, j] = 0;
            }

            feedForward.dB2[i] = 0;
        }
    }

    private void UpdateMatrix (double[,] weights, double[,] gradients) {
        int rows = weights.GetLength (0);
        int cols = weights.GetLength (1);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            weights[i, j] -= LearningRate * gradients[i, j];
            gradients[i, j] = 0;
        }
    }

    private void ZeroMatrix (double[,] matrix) {
        int rows = matrix.GetLength (0);
        int cols = matrix.GetLength (1);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = 0;
    }
}

// Training Code with backward pass and parameter updates
public class Trainer
{
    public static void train (LlamaForCausalLM model, SGDOptimizer optimizer, Func<(int[], int[])> data, int epochs, int epochSize, Action callback) {
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
