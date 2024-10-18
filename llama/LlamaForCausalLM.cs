using System;
using System.Collections.Generic;
using System.Linq;

namespace LlamaForCausalLM
{
    // Utility functions for vector and matrix operations
    public static class MathOps
    {
        public static double[] Add(double[] a, double[] b)
        {
            double[] result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] + b[i];
            return result;
        }

        public static double[] Subtract(double[] a, double[] b)
        {
            double[] result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] - b[i];
            return result;
        }

        public static double[] Multiply(double[] a, double scalar)
        {
            double[] result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] * scalar;
            return result;
        }

        public static double Dot(double[] a, double[] b)
        {
            double result = 0;
            for (int i = 0; i < a.Length; i++)
                result += a[i] * b[i];
            return result;
        }

        public static double[] MatrixVectorProduct(double[,] matrix, double[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[] result = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                result[i] = 0;
                for (int j = 0; j < cols; j++)
                    result[i] += matrix[i, j] * vector[j];
            }
            return result;
        }

        public static double[] MatrixVectorProductTranspose(double[,] matrix, double[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[] result = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                result[j] = 0;
                for (int i = 0; i < rows; i++)
                    result[j] += matrix[i, j] * vector[i];
            }
            return result;
        }

        public static double[,] OuterProduct(double[] a, double[] b)
        {
            int rows = a.Length;
            int cols = b.Length;
            double[,] result = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = a[i] * b[j];
            return result;
        }

        public static double[] Softmax(double[] logits)
        {
            double maxLogit = logits.Max();
            double[] expLogits = logits.Select(x => Math.Exp(x - maxLogit)).ToArray();
            double sumExp = expLogits.Sum();
            return expLogits.Select(x => x / sumExp).ToArray();
        }

        public static double[] SoftmaxGradient(double[] softmax, double[] dSoftmax)
        {
            int n = softmax.Length;
            double[] dLogits = new double[n];
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++)
                {
                    double delta = i == j ? 1.0 : 0.0;
                    sum += dSoftmax[j] * softmax[i] * (delta - softmax[j]);
                }
                dLogits[i] = sum;
            }
            return dLogits;
        }
    }

    // Embedding layer with backward pass
    public class Embedding
    {
        public double[,] Weights;
        public double[,] Gradients;
        public Embedding(int vocabSize, int embedSize)
        {
            Weights = new double[vocabSize, embedSize];
            Gradients = new double[vocabSize, embedSize];
            Random rand = new Random();
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < embedSize; j++)
                    Weights[i, j] = rand.NextDouble() * 0.02 - 0.01;
        }

        public double[] Forward(int token)
        {
            double[] embedding = new double[Weights.GetLength(1)];
            for (int i = 0; i < Weights.GetLength(1); i++)
                embedding[i] = Weights[token, i];
            return embedding;
        }

        public void Backward(int token, double[] grad)
        {
            for (int i = 0; i < Weights.GetLength(1); i++)
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

        public LayerNorm(int size)
        {
            Size = size;
            Gamma = new double[size];
            Beta = new double[size];
            dGamma = new double[size];
            dBeta = new double[size];
            for (int i = 0; i < size; i++)
            {
                Gamma[i] = 1.0;
                Beta[i] = 0.0;
            }
        }

        public double[] Forward(double[] x)
        {
            x_input = x;
            mean = x.Average();
            variance = x.Select(val => Math.Pow(val - mean, 2)).Average();
            normalized = x.Select(val => (val - mean) / Math.Sqrt(variance + 1e-5)).ToArray();
            double[] output = new double[Size];
            for (int i = 0; i < Size; i++)
                output[i] = Gamma[i] * normalized[i] + Beta[i];
            return output;
        }

        public double[] Backward(double[] gradOutput)
        {
            double[] dxhat = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                dGamma[i] += gradOutput[i] * normalized[i];
                dBeta[i] += gradOutput[i];
                dxhat[i] = gradOutput[i] * Gamma[i];
            }

            double stdInv = 1.0 / Math.Sqrt(variance + 1e-5);
            double[] dx = new double[Size];
            double dvar = -0.5 * stdInv * stdInv * stdInv * dxhat.Select((dxh, i) => (x_input[i] - mean) * dxh).Sum();
            double dmean = -stdInv * dxhat.Sum() + dvar * (-2.0 / Size) * (x_input.Sum() - Size * mean);
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
        private List<double[]> ScoresList;
        private List<double[]> Softmaxes;

        public SelfAttention(int embedSize, int headSize)
        {
            EmbedSize = embedSize;
            HeadSize = headSize;
            Wq = InitializeMatrix(HeadSize, EmbedSize);
            Wk = InitializeMatrix(HeadSize, EmbedSize);
            Wv = InitializeMatrix(HeadSize, EmbedSize);
            Wo = InitializeMatrix(EmbedSize, HeadSize);

            dWq = new double[HeadSize, EmbedSize];
            dWk = new double[HeadSize, EmbedSize];
            dWv = new double[HeadSize, EmbedSize];
            dWo = new double[EmbedSize, HeadSize];
        }

        private double[,] InitializeMatrix(int rows, int cols)
        {
            double[,] matrix = new double[rows, cols];
            Random rand = new Random();
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = rand.NextDouble() * 0.02 - 0.01;
            return matrix;
        }

        public double[] Forward(List<double[]> inputs)
        {
            Inputs = inputs;
            int seqLen = inputs.Count;
            Qs = new List<double[]>();
            Ks = new List<double[]>();
            Vs = new List<double[]>();
            Zs = new List<double[]>();
            ScoresList = new List<double[]>();
            Softmaxes = new List<double[]>();

            // Compute Q, K, V
            foreach (var x in inputs)
            {
                Qs.Add(MathOps.MatrixVectorProduct(Wq, x)); // Shape: [HeadSize]
                Ks.Add(MathOps.MatrixVectorProduct(Wk, x)); // Shape: [HeadSize]
                Vs.Add(MathOps.MatrixVectorProduct(Wv, x)); // Shape: [HeadSize]
            }

            // Compute attention scores and apply softmax
            double[] scores = new double[seqLen]; // For the last time step
            for (int t = 0; t < seqLen; t++)
                scores[t] = MathOps.Dot(Qs[seqLen - 1], Ks[t]) / Math.Sqrt(HeadSize);

            double[] softmax = MathOps.Softmax(scores);
            Softmaxes.Add(softmax);

            // Compute weighted sum of Vs
            double[] z = new double[HeadSize];
            for (int t = 0; t < seqLen; t++)
                for (int i = 0; i < HeadSize; i++)
                    z[i] += softmax[t] * Vs[t][i];
            Zs.Add(z);

            // Output projection
            double[] output = MathOps.MatrixVectorProduct(Wo, z);
            return output;
        }

        public List<double[]> Backward(double[] gradOutput)
        {
            int seqLen = Inputs.Count;
            double[] dZ = MathOps.MatrixVectorProductTranspose(Wo, gradOutput);
            // Accumulate gradients for Wo
            double[,] dWo_temp = MathOps.OuterProduct(gradOutput, Zs[0]); // Since we have only one z
            for (int i = 0; i < Wo.GetLength(0); i++)
                for (int j = 0; j < Wo.GetLength(1); j++)
                    dWo[i, j] += dWo_temp[i, j];

            double[] dSoftmax = new double[seqLen];
            double[][] dVs = new double[seqLen][];
            for (int t = 0; t < seqLen; t++)
            {
                dVs[t] = new double[HeadSize];
                for (int i = 0; i < HeadSize; i++)
                    dVs[t][i] = dZ[i] * Softmaxes[0][t];
            }

            // Compute gradient w.r.t. softmax inputs (scores)
            for (int t = 0; t < seqLen; t++)
            {
                double sum = 0;
                for (int i = 0; i < HeadSize; i++)
                    sum += dZ[i] * Vs[t][i];
                dSoftmax[t] = sum;
            }

            double[] dScores = MathOps.SoftmaxGradient(Softmaxes[0], dSoftmax);

            // Initialize gradients for Q and Ks
            double[] dQ = new double[HeadSize];
            double[][] dKs = new double[seqLen][];
            for (int t = 0; t < seqLen; t++)
                dKs[t] = new double[HeadSize];

            // Backprop through attention scores
            for (int t = 0; t < seqLen; t++)
            {
                // dScore/dQ = K_t / sqrt(HeadSize)
                double coeff = 1.0 / Math.Sqrt(HeadSize);
                for (int i = 0; i < HeadSize; i++)
                {
                    dQ[i] += dScores[t] * Ks[t][i] * coeff;
                    dKs[t][i] += dScores[t] * Qs[seqLen - 1][i] * coeff;
                }
            }

            // Backprop through Qs and Ks to inputs
            double[] dQInput = MathOps.MatrixVectorProductTranspose(Wq, dQ);
            double[][] dKInputs = new double[seqLen][];
            for (int t = 0; t < seqLen; t++)
            {
                double[] dK = dKs[t];
                double[] dKInput = MathOps.MatrixVectorProductTranspose(Wk, dK);
                dKInputs[t] = dKInput;
                // Accumulate gradients for Wk
                double[,] dWk_temp = MathOps.OuterProduct(dKs[t], Inputs[t]);
                for (int i = 0; i < Wk.GetLength(0); i++)
                    for (int j = 0; j < Wk.GetLength(1); j++)
                        dWk[i, j] += dWk_temp[i, j];
            }

            // Backprop through Vs to inputs
            double[][] dVInputs = new double[seqLen][];
            for (int t = 0; t < seqLen; t++)
            {
                double[] dV = dVs[t];
                double[] dVInput = MathOps.MatrixVectorProductTranspose(Wv, dV);
                dVInputs[t] = dVInput;
                // Accumulate gradients for Wv
                double[,] dWv_temp = MathOps.OuterProduct(dVs[t], Inputs[t]);
                for (int i = 0; i < Wv.GetLength(0); i++)
                    for (int j = 0; j < Wv.GetLength(1); j++)
                        dWv[i, j] += dWv_temp[i, j];
            }

            // Accumulate gradients for Wq
            double[,] dWq_temp = MathOps.OuterProduct(dQ, Inputs[seqLen - 1]);
            for (int i = 0; i < Wq.GetLength(0); i++)
                for (int j = 0; j < Wq.GetLength(1); j++)
                    dWq[i, j] += dWq_temp[i, j];

            // Backprop through inputs
            List<double[]> dInputs = new List<double[]>();
            for (int t = 0; t < seqLen; t++)
            {
                double[] dx = new double[EmbedSize];

                // Accumulate gradients from Q (only for last input)
                if (t == seqLen - 1)
                    for (int i = 0; i < EmbedSize; i++)
                        dx[i] += Wq.Transpose()[i].Dot(dQ);

                // Accumulate gradients from K
                for (int i = 0; i < EmbedSize; i++)
                    dx[i] += Wk.Transpose()[i].Dot(dKs[t]);

                // Accumulate gradients from V
                for (int i = 0; i < EmbedSize; i++)
                    dx[i] += Wv.Transpose()[i].Dot(dVs[t]);

                dInputs.Add(dx);
            }

            return dInputs;
        }
    }

    // Extension methods for matrix operations
    public static class MatrixExtensions
    {
        public static double[][] Transpose(this double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[][] result = new double[cols][];
            for (int i = 0; i < cols; i++)
            {
                result[i] = new double[rows];
                for (int j = 0; j < rows; j++)
                    result[i][j] = matrix[j, i];
            }
            return result;
        }

        public static double Dot(this double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
                sum += a[i] * b[i];
            return sum;
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

        public FeedForward(int embedSize, int hiddenSize)
        {
            EmbedSize = embedSize;
            HiddenSize = hiddenSize;
            W1 = InitializeMatrix(HiddenSize, EmbedSize);
            B1 = new double[HiddenSize];
            W2 = InitializeMatrix(EmbedSize, HiddenSize);
            B2 = new double[EmbedSize];

            dW1 = new double[HiddenSize, EmbedSize];
            dB1 = new double[HiddenSize];
            dW2 = new double[EmbedSize, HiddenSize];
            dB2 = new double[EmbedSize];
        }

        private double[,] InitializeMatrix(int rows, int cols)
        {
            double[,] matrix = new double[rows, cols];
            Random rand = new Random();
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = rand.NextDouble() * 0.02 - 0.01;
            return matrix;
        }

        public double[] Forward(double[] x)
        {
            x_input = x;
            h_linear = new double[HiddenSize];
            h_relu = new double[HiddenSize];

            // First layer
            for (int i = 0; i < HiddenSize; i++)
            {
                h_linear[i] = B1[i];
                for (int j = 0; j < EmbedSize; j++)
                    h_linear[i] += W1[i, j] * x[j];
                h_relu[i] = Math.Max(0, h_linear[i]);
            }

            // Second layer
            double[] y = new double[EmbedSize];
            for (int i = 0; i < EmbedSize; i++)
            {
                y[i] = B2[i];
                for (int j = 0; j < HiddenSize; j++)
                    y[i] += W2[i, j] * h_relu[j];
            }

            return y;
        }

        public double[] Backward(double[] dOut)
        {
            double[] dh_relu = new double[HiddenSize];

            // Backprop through second layer
            for (int i = 0; i < EmbedSize; i++)
            {
                dB2[i] += dOut[i];
                for (int j = 0; j < HiddenSize; j++)
                {
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
            for (int i = 0; i < HiddenSize; i++)
            {
                dB1[i] += dh_linear[i];
                for (int j = 0; j < EmbedSize; j++)
                {
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
        private double[] saOutput;
        private double[] saAdded;
        private double[] normedSaAdded;
        private double[] ffOutput;
        private double[] ffAdded;

        public TransformerBlock(int embedSize, int hiddenSize, int headSize)
        {
            EmbedSize = embedSize;
            Norm1 = new LayerNorm(embedSize);
            Norm2 = new LayerNorm(embedSize);
            SelfAttention = new SelfAttention(embedSize, headSize);
            FeedForward = new FeedForward(embedSize, hiddenSize);
        }

        public double[] Forward(List<double[]> inputs)
        {
            this.inputs = inputs;
            normedInputs = inputs.Select(x => Norm1.Forward(x)).ToList();
            saOutput = SelfAttention.Forward(normedInputs);
            saAdded = MathOps.Add(inputs.Last(), saOutput);
            normedSaAdded = Norm2.Forward(saAdded);
            ffOutput = FeedForward.Forward(normedSaAdded);
            ffAdded = MathOps.Add(saAdded, ffOutput);
            return ffAdded;
        }

        public List<double[]> Backward(double[] gradOutput)
        {
            // Backward through addition
            double[] dSaAdded = new double[EmbedSize];
            double[] dFfOutput = new double[EmbedSize];
            for (int i = 0; i < EmbedSize; i++)
            {
                dSaAdded[i] += gradOutput[i];
                dFfOutput[i] = gradOutput[i];
            }

            // Backward through FeedForward
            double[] dNormedSaAdded = FeedForward.Backward(dFfOutput);

            // Backward through second LayerNorm
            double[] dSaAdded2 = Norm2.Backward(dNormedSaAdded);

            // Sum gradients from both paths
            for (int i = 0; i < EmbedSize; i++)
                dSaAdded[i] += dSaAdded2[i];

            // Backward through addition
            double[] dSaOutput = new double[EmbedSize];
            double[] dInputLast = new double[EmbedSize];
            for (int i = 0; i < EmbedSize; i++)
            {
                dInputLast[i] = dSaAdded[i];
                dSaOutput[i] = dSaAdded[i];
            }

            // Backward through SelfAttention
            List<double[]> dNormedInputs = SelfAttention.Backward(dSaOutput);

            // Backward through first LayerNorm
            List<double[]> dInputs = new List<double[]>();
            for (int i = 0; i < inputs.Count; i++)
            {
                double[] dNorm = Norm1.Backward(dNormedInputs[i]);
                // Sum gradients from residual connection
                if (i == inputs.Count - 1)
                    for (int j = 0; j < EmbedSize; j++)
                        dNorm[j] += dInputLast[j];
                dInputs.Add(dNorm);
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
        private double[] finalEmbedding;
        private int[] inputTokens;

        public LlamaForCausalLM(int vocabSize, int embedSize, int hiddenSize, int headSize, int numLayers)
        {
            VocabSize = vocabSize;
            EmbedSize = embedSize;
            HiddenSize = hiddenSize;
            HeadSize = headSize;
            NumLayers = numLayers;
            TokenEmbedding = new Embedding(vocabSize, embedSize);
            TransformerBlocks = new List<TransformerBlock>();
            for (int i = 0; i < numLayers; i++)
                TransformerBlocks.Add(new TransformerBlock(embedSize, hiddenSize, headSize));
            FinalLayerNorm = new LayerNorm(embedSize);
            OutputProjection = InitializeMatrix(VocabSize, EmbedSize);
            dOutputProjection = new double[VocabSize, EmbedSize];
        }

        private double[,] InitializeMatrix(int rows, int cols)
        {
            double[,] matrix = new double[rows, cols];
            Random rand = new Random();
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = rand.NextDouble() * 0.02 - 0.01;
            return matrix;
        }

        public double[] Forward(int[] inputTokens)
        {
            this.inputTokens = inputTokens;
            embeddings = inputTokens.Select(token => TokenEmbedding.Forward(token)).ToList();
            foreach (var block in TransformerBlocks)
                embeddings.Add(block.Forward(embeddings));
            finalEmbedding = FinalLayerNorm.Forward(embeddings.Last());

            // Compute logits
            double[] logits = new double[VocabSize];
            for (int i = 0; i < VocabSize; i++)
                logits[i] = MathOps.Dot(OutputProjection.GetRow(i), finalEmbedding);
            return logits;
        }

        public void Backward(double[] dLogits)
        {
            // Backward through OutputProjection
            for (int i = 0; i < VocabSize; i++)
                for (int j = 0; j < EmbedSize; j++)
                {
                    dOutputProjection[i, j] += dLogits[i] * finalEmbedding[j];
                }
            double[] dFinalEmbedding = new double[EmbedSize];
            for (int j = 0; j < EmbedSize; j++)
                for (int i = 0; i < VocabSize; i++)
                    dFinalEmbedding[j] += OutputProjection[i, j] * dLogits[i];

            // Backward through FinalLayerNorm
            double[] dEmbedding = FinalLayerNorm.Backward(dFinalEmbedding);

            // Backward through TransformerBlocks
            for (int i = TransformerBlocks.Count - 1; i >= 0; i--)
            {
                var block = TransformerBlocks[i];
                List<double[]> dEmbeddings = block.Backward(dEmbedding);
                // Sum gradients for embeddings
                for (int j = 0; j < embeddings.Count; j++)
                {
                    if (i == 0)
                    {
                        // Accumulate gradients for the initial embeddings
                        embeddings[j] = dEmbeddings[j];
                    }
                    else
                    {
                        for (int k = 0; k < EmbedSize; k++)
                            embeddings[j][k] += dEmbeddings[j][k];
                    }
                }
                dEmbedding = embeddings[i + 1]; // Next gradient to propagate
            }

            // Backward through TokenEmbedding
            for (int i = 0; i < inputTokens.Length; i++)
            {
                TokenEmbedding.Backward(inputTokens[i], embeddings[i]);
            }
        }
    }

    // Extension method to get a row from a 2D array
    public static class Extensions
    {
        public static double[] GetRow(this double[,] matrix, int row)
        {
            int cols = matrix.GetLength(1);
            double[] result = new double[cols];
            for (int i = 0; i < cols; i++)
                result[i] = matrix[row, i];
            return result;
        }
    }

    // Loss Function with gradient
    public static class LossFunctions
    {
        public static double CrossEntropyLoss(double[] logits, int targetToken, out double[] dLogits)
        {
            double[] probabilities = MathOps.Softmax(logits);
            dLogits = new double[logits.Length];
            for (int i = 0; i < logits.Length; i++)
                dLogits[i] = probabilities[i];
            dLogits[targetToken] -= 1; // Gradient of softmax cross-entropy
            return -Math.Log(probabilities[targetToken] + 1e-9);
        }
    }

    // Optimizer updated to handle all parameters
    public class SGDOptimizer
    {
        public double LearningRate;
        public SGDOptimizer(double learningRate)
        {
            LearningRate = learningRate;
        }

        public void Step(LlamaForCausalLM model)
        {
            // Update TokenEmbedding weights
            for (int i = 0; i < model.TokenEmbedding.Weights.GetLength(0); i++)
                for (int j = 0; j < model.TokenEmbedding.Weights.GetLength(1); j++)
                {
                    model.TokenEmbedding.Weights[i, j] -= LearningRate * model.TokenEmbedding.Gradients[i, j];
                    model.TokenEmbedding.Gradients[i, j] = 0; // Reset gradients
                }

            // Update OutputProjection
            for (int i = 0; i < model.VocabSize; i++)
                for (int j = 0; j < model.EmbedSize; j++)
                {
                    model.OutputProjection[i, j] -= LearningRate * model.dOutputProjection[i, j];
                    model.dOutputProjection[i, j] = 0;
                }

            // Update LayerNorm parameters
            UpdateLayerNorm(model.FinalLayerNorm);

            // Update TransformerBlocks
            foreach (var block in model.TransformerBlocks)
            {
                UpdateLayerNorm(block.Norm1);
                UpdateLayerNorm(block.Norm2);

                // Update SelfAttention parameters
                UpdateMatrix(block.SelfAttention.Wq, block.SelfAttention.dWq);
                UpdateMatrix(block.SelfAttention.Wk, block.SelfAttention.dWk);
                UpdateMatrix(block.SelfAttention.Wv, block.SelfAttention.dWv);
                UpdateMatrix(block.SelfAttention.Wo, block.SelfAttention.dWo);

                // Reset gradients
                ZeroMatrix(block.SelfAttention.dWq);
                ZeroMatrix(block.SelfAttention.dWk);
                ZeroMatrix(block.SelfAttention.dWv);
                ZeroMatrix(block.SelfAttention.dWo);

                // Update FeedForward parameters
                UpdateFeedForward(block.FeedForward);
            }
        }

        private void UpdateLayerNorm(LayerNorm layerNorm)
        {
            for (int i = 0; i < layerNorm.Size; i++)
            {
                layerNorm.Gamma[i] -= LearningRate * layerNorm.dGamma[i];
                layerNorm.Beta[i] -= LearningRate * layerNorm.dBeta[i];
                layerNorm.dGamma[i] = 0;
                layerNorm.dBeta[i] = 0;
            }
        }

        private void UpdateFeedForward(FeedForward feedForward)
        {
            // Update W1 and B1
            for (int i = 0; i < feedForward.HiddenSize; i++)
            {
                feedForward.B1[i] -= LearningRate * feedForward.dB1[i];
                for (int j = 0; j < feedForward.EmbedSize; j++)
                {
                    feedForward.W1[i, j] -= LearningRate * feedForward.dW1[i, j];
                    feedForward.dW1[i, j] = 0;
                }
                feedForward.dB1[i] = 0;
            }

            // Update W2 and B2
            for (int i = 0; i < feedForward.EmbedSize; i++)
            {
                feedForward.B2[i] -= LearningRate * feedForward.dB2[i];
                for (int j = 0; j < feedForward.HiddenSize; j++)
                {
                    feedForward.W2[i, j] -= LearningRate * feedForward.dW2[i, j];
                    feedForward.dW2[i, j] = 0;
                }
                feedForward.dB2[i] = 0;
            }
        }

        private void UpdateMatrix(double[,] weights, double[,] gradients)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    weights[i, j] -= LearningRate * gradients[i, j];
                    gradients[i, j] = 0;
                }
        }

        private void ZeroMatrix(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = 0;
        }
    }

    // Training Code with backward pass and parameter updates
    public class Trainer
    {
        public void Train(LlamaForCausalLM model, List<int[]> inputSequences, List<int[]> targetSequences, int epochs, double learningRate)
        {
            SGDOptimizer optimizer = new SGDOptimizer(learningRate);
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;
                for (int i = 0; i < inputSequences.Count; i++)
                {
                    int[] inputTokens = inputSequences[i];
                    int[] targetTokens = targetSequences[i];

                    // Forward pass
                    double[] logits = model.Forward(inputTokens);
                    int targetToken = targetTokens.Last();

                    // Compute loss and gradient
                    double[] dLogits;
                    double loss = LossFunctions.CrossEntropyLoss(logits, targetToken, out dLogits);
                    totalLoss += loss;

                    // Backward pass
                    model.Backward(dLogits);

                    // Update parameters
                    optimizer.Step(model);
                }
                Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {totalLoss / inputSequences.Count}");
            }
        }
    }
}
