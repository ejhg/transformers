using System;

namespace mingpt6;

public class Vector
{
    public double[] Data;
    public int Length;

    public Vector (int length) {
        Length = length;
        Data = new double[length];
    }

    public Vector (double[] data) {
        Data = data;
        Length = data.Length;
    }

    public static Vector operator + (Vector a, Vector b) {
        Vector result = new Vector (a.Length);
        for (int i = 0; i < a.Length; i++)
            result.Data[i] = a.Data[i] + b.Data[i];
        return result;
    }

    public static Vector operator - (Vector a, Vector b) {
        Vector result = new Vector (a.Length);
        for (int i = 0; i < a.Length; i++)
            result.Data[i] = a.Data[i] - b.Data[i];
        return result;
    }

    public static Vector operator * (double scalar, Vector a) {
        Vector result = new Vector (a.Length);
        for (int i = 0; i < a.Length; i++)
            result.Data[i] = scalar * a.Data[i];
        return result;
    }

    public static double Dot (Vector a, Vector b) {
        double result = 0;
        for (int i = 0; i < a.Length; i++)
            result += a.Data[i] * b.Data[i];
        return result;
    }

    public static Vector ElementWiseMultiply (Vector a, Vector b) {
        Vector result = new Vector (a.Length);
        for (int i = 0; i < a.Length; i++)
            result.Data[i] = a.Data[i] * b.Data[i];
        return result;
    }

    public Vector Clone () {
        return new Vector ((double[])Data.Clone ());
    }
}

public class Matrix
{
    public double[][] Data;
    public int Rows;
    public int Cols;

    public Matrix (int rows, int cols) {
        Rows = rows;
        Cols = cols;
        Data = new double[rows][];
        for (int i = 0; i < rows; i++)
            Data[i] = new double[cols];
    }

    public Matrix (double[][] data) {
        Data = data;
        Rows = data.Length;
        Cols = data[0].Length;
    }

    public static Matrix operator + (Matrix a, Matrix b) {
        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] + b.Data[i][j];
        return result;
    }

    public static Matrix operator - (Matrix a, Matrix b) {
        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] - b.Data[i][j];
        return result;
    }

    public static Matrix operator * (double scalar, Matrix a) {
        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = scalar * a.Data[i][j];
        return result;
    }

    public static Matrix Multiply (Matrix a, Matrix b) {
        if (a.Cols != b.Rows)
            throw new Exception ("Matrix dimensions do not match for multiplication.");

        Matrix result = new Matrix (a.Rows, b.Cols);

        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < b.Cols; j++)
        for (int k = 0; k < a.Cols; k++)
            result.Data[i][j] += a.Data[i][k] * b.Data[k][j];

        return result;
    }

    public Matrix Transpose () {
        Matrix result = new Matrix (Cols, Rows);
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            result.Data[j][i] = Data[i][j];
        return result;
    }

    public Matrix Clone () {
        double[][] newData = new double[Rows][];
        for (int i = 0; i < Rows; i++)
            newData[i] = (double[])Data[i].Clone ();
        return new Matrix (newData);
    }
}

public static class ActivationFunctions
{
    public static double[] Softmax (double[] logits) {
        double maxLogit = double.MinValue;
        foreach (var logit in logits)
            if (logit > maxLogit)
                maxLogit = logit;

        double sumExp = 0;
        double[] exps = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++) {
            exps[i] = Math.Exp (logits[i] - maxLogit);
            sumExp += exps[i];
        }

        double[] softmax = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            softmax[i] = exps[i] / sumExp;

        return softmax;
    }
}

public class LayerNorm
{
    private int hiddenSize;
    private double[] gamma;
    private double[] beta;

    private double[] input;
    private double[] normalizedInput;
    private double mean;
    private double variance;
    private double epsilon = 1e-5;

    public LayerNorm (int hiddenSize) {
        this.hiddenSize = hiddenSize;
        gamma = new double[hiddenSize];
        beta = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
            gamma[i] = 1.0;
        // beta initialized to zeros
    }

    public double[] Forward (double[] input) {
        this.input = input;
        mean = 0;
        for (int i = 0; i < input.Length; i++)
            mean += input[i];
        mean /= input.Length;

        variance = 0;
        for (int i = 0; i < input.Length; i++)
            variance += (input[i] - mean) * (input[i] - mean);
        variance /= input.Length;

        normalizedInput = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
            normalizedInput[i] = (input[i] - mean) / Math.Sqrt (variance + epsilon);

        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
            output[i] = gamma[i] * normalizedInput[i] + beta[i];

        return output;
    }

    public double[] Backward (double[] gradOutput) {
        // Implement backpropagation for layer normalization
        // Compute gradients w.r.t gamma, beta, and input

        int N = input.Length;

        double[] gradGamma = new double[N];
        double[] gradBeta = new double[N];
        double[] gradInput = new double[N];

        for (int i = 0; i < N; i++) {
            gradGamma[i] = gradOutput[i] * normalizedInput[i];
            gradBeta[i] = gradOutput[i];
        }

        // Compute gradInput
        double[] dxhat = new double[N];
        for (int i = 0; i < N; i++)
            dxhat[i] = gradOutput[i] * gamma[i];

        double dvariance = 0;
        for (int i = 0; i < N; i++)
            dvariance += dxhat[i] * (input[i] - mean) * -0.5 * Math.Pow (variance + epsilon, -1.5);

        double dmean = 0;
        for (int i = 0; i < N; i++)
            dmean += dxhat[i] * -1 / Math.Sqrt (variance + epsilon);
        dmean += dvariance * -2 * mean / N;

        for (int i = 0; i < N; i++)
            gradInput[i] = dxhat[i] / Math.Sqrt (variance + epsilon) + dvariance * 2 * (input[i] - mean) / N + dmean / N;

        // Update gamma and beta parameters
        // Here, we should store gradGamma and gradBeta for parameter update in optimizer

        // For simplicity, let's assume we have functions to update gamma and beta

        return gradInput;
    }
}

public class RoPE
{
    private int hiddenSize;

    public RoPE (int hiddenSize, int maxPosition) {
        this.hiddenSize = hiddenSize;
    }

    public double[] ApplyRoPE (double[] x, int position) {
        double[] result = new double[hiddenSize];
        for (int i = 0; i < hiddenSize / 2; i++) {
            double theta = position / Math.Pow (10000, 2.0 * i / hiddenSize);
            double cosTheta = Math.Cos (theta);
            double sinTheta = Math.Sin (theta);

            result[2 * i] = x[2 * i] * cosTheta - x[2 * i + 1] * sinTheta;
            result[2 * i + 1] = x[2 * i] * sinTheta + x[2 * i + 1] * cosTheta;
        }

        return result;
    }
}

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

public class Transformer
{
    private int vocabSize;
    private int hiddenSize;
    private int numHeads;
    private int maxPosition;

    private Matrix embeddingMatrix;
    private MultiHeadSelfAttention attention;
    private LayerNorm layerNorm;
    private Matrix outputProjection;

    public Transformer (int vocabSize, int hiddenSize, int numHeads, int maxPosition) {
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.numHeads = numHeads;
        this.maxPosition = maxPosition;

        embeddingMatrix = new Matrix (vocabSize, hiddenSize);
        InitializeMatrix (embeddingMatrix);

        attention = new MultiHeadSelfAttention (numHeads, hiddenSize, maxPosition);
        layerNorm = new LayerNorm (hiddenSize);
        outputProjection = new Matrix (hiddenSize, vocabSize);
        InitializeMatrix (outputProjection);
    }

    private void InitializeMatrix (Matrix m) {
        Random rand = new Random ();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = rand.NextDouble () * 0.02 - 0.01; // Small random values
    }

    public double[] Forward (int[] inputIds) {
        int seqLength = inputIds.Length;
        double[][] embeddings = new double[seqLength][];
        for (int t = 0; t < seqLength; t++)
            embeddings[t] = embeddingMatrix.Data[inputIds[t]];

        // Apply attention
        double[] attentionOutput = attention.Forward (embeddings, seqLength);

        // Apply layer normalization
        double[] normalizedOutput = layerNorm.Forward (attentionOutput);

        // Compute logits
        double[] logits = MultiplyMatrixVector (outputProjection, normalizedOutput);

        // Apply softmax to get probabilities
        double[] probabilities = ActivationFunctions.Softmax (logits);

        return probabilities;
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

    // Implement backward pass and parameter updates
    public void Backward (int[] inputIds, double[] gradOutput) {
        // Implement backward pass to compute gradients and update parameters
        // For simplicity, this is left as an exercise
    }
}

public class Optimizer
{
    private double learningRate;

    public Optimizer (double learningRate) {
        this.learningRate = learningRate;
    }

    public void Update (Matrix param, Matrix grad) {
        for (int i = 0; i < param.Rows; i++)
        for (int j = 0; j < param.Cols; j++)
            param.Data[i][j] -= learningRate * grad.Data[i][j];
    }

    public void Update (double[] param, double[] grad) {
        for (int i = 0; i < param.Length; i++)
            param[i] -= learningRate * grad[i];
    }
}

public class TrainingLoop
{
    private Transformer model;
    private Optimizer optimizer;
    private int vocabSize;

    public TrainingLoop (Transformer model, Optimizer optimizer, int vocabSize) {
        this.model = model;
        this.optimizer = optimizer;
        this.vocabSize = vocabSize;
    }

    public void Train (int[][] dataset, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            int totalTokens = 0;
            foreach (var sequence in dataset) {
                for (int t = 1; t < sequence.Length; t++) {
                    int[] inputIds = new int[t];
                    Array.Copy (sequence, 0, inputIds, 0, t);
                    int targetId = sequence[t];

                    double[] probabilities = model.Forward (inputIds);

                    // Compute loss (negative log-likelihood)
                    double loss = -Math.Log (probabilities[targetId] + 1e-10);
                    totalLoss += loss;
                    totalTokens++;

                    // Compute gradient w.r.t logits
                    double[] gradOutput = new double[vocabSize];
                    gradOutput[targetId] = -1 / (probabilities[targetId] + 1e-10);

                    // Backward pass
                    model.Backward (inputIds, gradOutput);

                    // Update parameters
                    // For simplicity, parameter updates are assumed to be handled in model.Backward
                }
            }

            double perplexity = Math.Exp (totalLoss / totalTokens);
            Console.WriteLine ($"Epoch {epoch + 1}: Perplexity = {perplexity}");
        }
    }
}

public class Predictor
{
    private Transformer model;

    public Predictor (Transformer model) {
        this.model = model;
    }

    public int PredictNextToken (int[] inputIds) {
        double[] probabilities = model.Forward (inputIds);
        int predictedId = ArgMax (probabilities);
        return predictedId;
    }

    private int ArgMax (double[] array) {
        int index = 0;
        double max = array[0];
        for (int i = 1; i < array.Length; i++)
            if (array[i] > max) {
                max = array[i];
                index = i;
            }

        return index;
    }
}

public class Program
{
    public static void Main (string[] args) {
        int vocabSize = 1000;
        int hiddenSize = 128;
        int numHeads = 8;
        int maxPosition = 512;

        Transformer model = new Transformer (vocabSize, hiddenSize, numHeads, maxPosition);
        Optimizer optimizer = new Optimizer (learningRate: 0.001);
        TrainingLoop trainer = new TrainingLoop (model, optimizer, vocabSize);

        // Example dataset: list of sequences (token IDs)
        int[][] dataset = new int[][] {
            new int[] {
                1,
                2,
                3,
                4,
                5
            },
            new int[] {
                6,
                7,
                8,
                9,
                10
            },
            // Add more sequences
        };

        trainer.Train (dataset, epochs: 10);

        // Prediction
        Predictor predictor = new Predictor (model);
        int[] inputIds = new int[] {
            1,
            2,
            3
        };
        int nextTokenId = predictor.PredictNextToken (inputIds);
        Console.WriteLine ($"Predicted next token ID: {nextTokenId}");
    }
}
