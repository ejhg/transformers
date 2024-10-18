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
    }

    // Embedding layer
    public class Embedding
    {
        public double[,] Weights;
        public Embedding(int vocabSize, int embedSize)
        {
            Weights = new double[vocabSize, embedSize];
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
    }

    // Layer Normalization
    public class LayerNorm
    {
        public double[] Gamma;
        public double[] Beta;
        public int Size;
        private double[] mean;
        private double[] variance;

        public LayerNorm(int size)
        {
            Size = size;
            Gamma = new double[size];
            Beta = new double[size];
            for (int i = 0; i < size; i++)
            {
                Gamma[i] = 1.0;
                Beta[i] = 0.0;
            }
        }

        public double[] Forward(double[] x)
        {
            double mean = x.Average();
            double variance = x.Select(val => Math.Pow(val - mean, 2)).Average();
            double[] normalized = x.Select(val => (val - mean) / Math.Sqrt(variance + 1e-5)).ToArray();
            double[] output = new double[Size];
            for (int i = 0; i < Size; i++)
                output[i] = Gamma[i] * normalized[i] + Beta[i];
            return output;
        }
    }

    // Self-Attention mechanism
    public class SelfAttention
    {
        public double[,] Wq, Wk, Wv, Wo;
        public int EmbedSize, HeadSize;
        public SelfAttention(int embedSize, int headSize)
        {
            EmbedSize = embedSize;
            HeadSize = headSize;
            Wq = InitializeMatrix(embedSize, headSize);
            Wk = InitializeMatrix(embedSize, headSize);
            Wv = InitializeMatrix(embedSize, headSize);
            Wo = InitializeMatrix(headSize, embedSize);
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
            int seqLen = inputs.Count;
            List<double[]> Qs = new List<double[]>();
            List<double[]> Ks = new List<double[]>();
            List<double[]> Vs = new List<double[]>();

            // Compute Q, K, V
            foreach (var x in inputs)
            {
                Qs.Add(MathOps.MatrixVectorProduct(Wq, x));
                Ks.Add(MathOps.MatrixVectorProduct(Wk, x));
                Vs.Add(MathOps.MatrixVectorProduct(Wv, x));
            }

            List<double[]> outputs = new List<double[]>();
            for (int t = 0; t < seqLen; t++)
            {
                double[] q = Qs[t];
                double[] scores = new double[seqLen];
                for (int s = 0; s <= t; s++)
                    scores[s] = MathOps.Dot(q, Ks[s]) / Math.Sqrt(HeadSize);

                // Softmax
                double maxScore = scores.Take(t + 1).Max();
                double sumExp = 0;
                for (int s = 0; s <= t; s++)
                {
                    scores[s] = Math.Exp(scores[s] - maxScore);
                    sumExp += scores[s];
                }
                for (int s = 0; s <= t; s++)
                    scores[s] /= sumExp;

                // Weighted sum of Vs
                double[] z = new double[HeadSize];
                for (int s = 0; s <= t; s++)
                    for (int i = 0; i < HeadSize; i++)
                        z[i] += scores[s] * Vs[s][i];

                // Output projection
                outputs.Add(MathOps.MatrixVectorProduct(Wo, z));
            }

            return outputs.Last();
        }
    }

    // Feed-Forward Network
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
            W1 = InitializeMatrix(hiddenSize, embedSize);
            B1 = new double[hiddenSize];
            W2 = InitializeMatrix(embedSize, hiddenSize);
            B2 = new double[embedSize];
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
            dW1 = new double[HiddenSize, EmbedSize];
            dB1 = new double[HiddenSize];
            dW2 = new double[EmbedSize, HiddenSize];
            dB2 = new double[EmbedSize];
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

    // Transformer Block
    public class TransformerBlock
    {
        public LayerNorm Norm1, Norm2;
        public SelfAttention SelfAttention;
        public FeedForward FeedForward;
        public int EmbedSize;

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
            List<double[]> normedInputs = inputs.Select(x => Norm1.Forward(x)).ToList();
            double[] saOutput = SelfAttention.Forward(normedInputs);
            double[] saAdded = MathOps.Add(inputs.Last(), saOutput);
            double[] normedSaAdded = Norm2.Forward(saAdded);
            double[] ffOutput = FeedForward.Forward(normedSaAdded);
            return MathOps.Add(saAdded, ffOutput);
        }
    }

    // LlamaForCausalLM Model
    public class LlamaForCausalLM
    {
        public Embedding TokenEmbedding;
        public List<TransformerBlock> TransformerBlocks;
        public LayerNorm FinalLayerNorm;
        public double[,] OutputProjection;
        public int VocabSize, EmbedSize, HiddenSize, HeadSize, NumLayers;

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
            OutputProjection = InitializeMatrix(vocabSize, embedSize);
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
            List<double[]> embeddings = inputTokens.Select(token => TokenEmbedding.Forward(token)).ToList();
            foreach (var block in TransformerBlocks)
                embeddings.Add(block.Forward(embeddings));
            double[] finalEmbedding = FinalLayerNorm.Forward(embeddings.Last());

            // Compute logits
            double[] logits = new double[VocabSize];
            for (int i = 0; i < VocabSize; i++)
                logits[i] = MathOps.Dot(OutputProjection.GetRow(i), finalEmbedding);
            return logits;
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

    // Loss Function
    public static class LossFunctions
    {
        public static double CrossEntropyLoss(double[] logits, int targetToken)
        {
            double maxLogit = logits.Max();
            double sumExp = logits.Sum(logit => Math.Exp(logit - maxLogit));
            double[] probabilities = logits.Select(logit => Math.Exp(logit - maxLogit) / sumExp).ToArray();
            return -Math.Log(probabilities[targetToken] + 1e-9);
        }
    }

    // Optimizer
    public class SGDOptimizer
    {
        public double LearningRate;
        public SGDOptimizer(double learningRate)
        {
            LearningRate = learningRate;
        }

        public void Step(FeedForward feedForward)
        {
            // Update W1 and B1
            for (int i = 0; i < feedForward.HiddenSize; i++)
            {
                feedForward.B1[i] -= LearningRate * feedForward.dB1[i];
                for (int j = 0; j < feedForward.EmbedSize; j++)
                    feedForward.W1[i, j] -= LearningRate * feedForward.dW1[i, j];
            }

            // Update W2 and B2
            for (int i = 0; i < feedForward.EmbedSize; i++)
            {
                feedForward.B2[i] -= LearningRate * feedForward.dB2[i];
                for (int j = 0; j < feedForward.HiddenSize; j++)
                    feedForward.W2[i, j] -= LearningRate * feedForward.dW2[i, j];
            }
        }
    }

    // Training Code
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
                    double[] logits = model.Forward(inputTokens);
                    int targetToken = targetTokens.Last();
                    double loss = LossFunctions.CrossEntropyLoss(logits, targetToken);
                    totalLoss += loss;

                    // Backward pass and parameter updates would be implemented here

                    // Example for updating FeedForward layers in TransformerBlocks
                    foreach (var block in model.TransformerBlocks)
                        optimizer.Step(block.FeedForward);
                }
                Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {totalLoss / inputSequences.Count}");
            }
        }
    }
}
