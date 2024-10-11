using System;

namespace mingpt5;

// Vector class
public class Vector
{
    public int Size;
    public double[] Data;

    public Vector(int size)
    {
        Size = size;
        Data = new double[size];
    }

    public Vector(double[] data)
    {
        Size = data.Length;
        Data = new double[Size];
        Array.Copy(data, Data, Size);
    }

    public static Vector operator +(Vector a, Vector b)
    {
        if (a.Size != b.Size)
            throw new Exception("Vector sizes do not match");

        Vector result = new Vector(a.Size);
        for (int i = 0; i < a.Size; i++)
        {
            result.Data[i] = a.Data[i] + b.Data[i];
        }
        return result;
    }

    public static Vector operator -(Vector a, Vector b)
    {
        if (a.Size != b.Size)
            throw new Exception("Vector sizes do not match");

        Vector result = new Vector(a.Size);
        for (int i = 0; i < a.Size; i++)
        {
            result.Data[i] = a.Data[i] - b.Data[i];
        }
        return result;
    }

    public static Vector operator *(double scalar, Vector a)
    {
        Vector result = new Vector(a.Size);
        for (int i = 0; i < a.Size; i++)
        {
            result.Data[i] = scalar * a.Data[i];
        }
        return result;
    }

    public void ApplyFunction(Func<double, double> func)
    {
        for (int i = 0; i < Size; i++)
        {
            Data[i] = func(Data[i]);
        }
    }

    public double Dot(Vector other)
    {
        if (Size != other.Size)
            throw new Exception("Vector sizes do not match");

        double sum = 0.0;
        for (int i = 0; i < Size; i++)
        {
            sum += Data[i] * other.Data[i];
        }
        return sum;
    }

    public Vector Clone()
    {
        return new Vector(Data);
    }
}

// Matrix class
public class Matrix
{
    public int Rows;
    public int Cols;
    public double[][] Data;

    public Matrix(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        Data = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            Data[i] = new double[cols];
        }
    }

    public Matrix(double[][] data)
    {
        Rows = data.Length;
        Cols = data[0].Length;
        Data = new double[Rows][];
        for (int i = 0; i < Rows; i++)
        {
            Data[i] = new double[Cols];
            Array.Copy(data[i], Data[i], Cols);
        }
    }

    public static Matrix operator +(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new Exception("Matrix dimensions do not match");

        Matrix result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] + b.Data[i][j];
        return result;
    }

    public static Matrix operator -(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new Exception("Matrix dimensions do not match");

        Matrix result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] - b.Data[i][j];
        return result;
    }

    public static Matrix operator *(double scalar, Matrix a)
    {
        Matrix result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = scalar * a.Data[i][j];
        return result;
    }

    public static Matrix Multiply(Matrix a, Matrix b)
    {
        if (a.Cols != b.Rows)
            throw new Exception("Matrix dimensions are not compatible for multiplication");

        Matrix result = new Matrix(a.Rows, b.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < b.Cols; j++)
        for (int k = 0; k < a.Cols; k++)
            result.Data[i][j] += a.Data[i][k] * b.Data[k][j];
        return result;
    }

    public Matrix Transpose()
    {
        Matrix result = new Matrix(Cols, Rows);
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            result.Data[j][i] = Data[i][j];
        return result;
    }

    public Vector Multiply(Vector v)
    {
        if (Cols != v.Size)
            throw new Exception("Matrix and vector dimensions are not compatible");

        Vector result = new Vector(Rows);
        for (int i = 0; i < Rows; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < Cols; j++)
            {
                sum += Data[i][j] * v.Data[j];
            }
            result.Data[i] = sum;
        }
        return result;
    }

    public void ApplyFunction(Func<double, double> func)
    {
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            Data[i][j] = func(Data[i][j]);
    }

    public Matrix Clone()
    {
        return new Matrix(Data);
    }
}

// Embedding Layer
public class EmbeddingLayer
{
    public int VocabSize;
    public int EmbeddingDim;
    public Matrix EmbeddingMatrix;

    public EmbeddingLayer(int vocabSize, int embeddingDim)
    {
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        EmbeddingMatrix = new Matrix(VocabSize, EmbeddingDim);
        InitializeWeights();
    }

    private void InitializeWeights()
    {
        Random rand = new Random();
        for (int i = 0; i < VocabSize; i++)
        for (int j = 0; j < EmbeddingDim; j++)
            EmbeddingMatrix.Data[i][j] = (rand.NextDouble() - 0.5) / EmbeddingDim;
    }

    public Vector GetEmbedding(int tokenIndex)
    {
        Vector embedding = new Vector(EmbeddingDim);
        for (int i = 0; i < EmbeddingDim; i++)
        {
            embedding.Data[i] = EmbeddingMatrix.Data[tokenIndex][i];
        }
        return embedding;
    }

    public void UpdateEmbedding(int tokenIndex, Vector grad, double learningRate)
    {
        for (int i = 0; i < EmbeddingDim; i++)
        {
            EmbeddingMatrix.Data[tokenIndex][i] -= learningRate * grad.Data[i];
        }
    }
}

// Positional Encoding (Rotational Positional Encoding)
public class PositionalEncoding
{
    public int EmbeddingDim;

    public PositionalEncoding(int embeddingDim)
    {
        EmbeddingDim = embeddingDim;
    }

    public void ApplyRoPE(Vector embedding, int position)
    {
        // Implementing Rotational Positional Encoding
        int halfDim = EmbeddingDim / 2;
        double theta = 10000.0;
        for (int i = 0; i < halfDim; i++)
        {
            double angle = position / Math.Pow(theta, (2.0 * i) / EmbeddingDim);
            double sinAngle = Math.Sin(angle);
            double cosAngle = Math.Cos(angle);
            double original1 = embedding.Data[2 * i];
            double original2 = embedding.Data[2 * i + 1];

            embedding.Data[2 * i] = original1 * cosAngle - original2 * sinAngle;
            embedding.Data[2 * i + 1] = original1 * sinAngle + original2 * cosAngle;
        }
    }
}

// Layer Normalization
public class LayerNormalization
{
    public int FeatureSize;
    public Vector Gamma;
    public Vector Beta;

    public LayerNormalization(int featureSize)
    {
        FeatureSize = featureSize;
        Gamma = new Vector(featureSize);
        Beta = new Vector(featureSize);
        InitializeParameters();
    }

    private void InitializeParameters()
    {
        for (int i = 0; i < FeatureSize; i++)
        {
            Gamma.Data[i] = 1.0;
            Beta.Data[i] = 0.0;
        }
    }

    public Vector Normalize(Vector input)
    {
        double mean = 0.0;
        double variance = 0.0;

        for (int i = 0; i < FeatureSize; i++)
        {
            mean += input.Data[i];
        }
        mean /= FeatureSize;

        for (int i = 0; i < FeatureSize; i++)
        {
            variance += Math.Pow(input.Data[i] - mean, 2);
        }
        variance /= FeatureSize;

        Vector normalized = new Vector(FeatureSize);
        for (int i = 0; i < FeatureSize; i++)
        {
            normalized.Data[i] = (input.Data[i] - mean) / Math.Sqrt(variance + 1e-6);
            normalized.Data[i] = Gamma.Data[i] * normalized.Data[i] + Beta.Data[i];
        }

        return normalized;
    }

    public (Vector, double, double) Forward(Vector input)
    {
        // Returns normalized input, mean, and variance for backward pass
        double mean = 0.0;
        double variance = 0.0;

        for (int i = 0; i < FeatureSize; i++)
        {
            mean += input.Data[i];
        }
        mean /= FeatureSize;

        for (int i = 0; i < FeatureSize; i++)
        {
            variance += Math.Pow(input.Data[i] - mean, 2);
        }
        variance /= FeatureSize;

        Vector normalized = new Vector(FeatureSize);
        for (int i = 0; i < FeatureSize; i++)
        {
            normalized.Data[i] = (input.Data[i] - mean) / Math.Sqrt(variance + 1e-6);
            normalized.Data[i] = Gamma.Data[i] * normalized.Data[i] + Beta.Data[i];
        }

        return (normalized, mean, variance);
    }

    public (Vector, Vector, Vector) Backward(Vector dout, Vector input, double mean, double variance)
    {
        // Backpropagation for Layer Normalization
        Vector dGamma = new Vector(FeatureSize);
        Vector dBeta = new Vector(FeatureSize);
        Vector dx = new Vector(FeatureSize);

        double invStd = 1.0 / Math.Sqrt(variance + 1e-6);
        Vector xHat = new Vector(FeatureSize);
        for (int i = 0; i < FeatureSize; i++)
        {
            xHat.Data[i] = (input.Data[i] - mean) * invStd;
        }

        for (int i = 0; i < FeatureSize; i++)
        {
            dGamma.Data[i] = dout.Data[i] * xHat.Data[i];
            dBeta.Data[i] = dout.Data[i];
        }

        for (int i = 0; i < FeatureSize; i++)
        {
            double dXhat = dout.Data[i] * Gamma.Data[i];
            double dVar = -0.5 * dXhat * (input.Data[i] - mean) * Math.Pow(variance + 1e-6, -1.5);
            double dMean = -dXhat * invStd;
            dx.Data[i] = dXhat * invStd + dVar * 2.0 * (input.Data[i] - mean) / FeatureSize + dMean / FeatureSize;
        }

        return (dx, dGamma, dBeta);
    }
}

// Multi-Head Self-Attention
public class MultiHeadAttention
{
    public int EmbeddingDim;
    public int NumHeads;
    public int HeadDim;
    public Matrix[] Wq;
    public Matrix[] Wk;
    public Matrix[] Wv;
    public Matrix Wo;

    public MultiHeadAttention(int embeddingDim, int numHeads)
    {
        EmbeddingDim = embeddingDim;
        NumHeads = numHeads;
        HeadDim = EmbeddingDim / NumHeads;

        Wq = new Matrix[NumHeads];
        Wk = new Matrix[NumHeads];
        Wv = new Matrix[NumHeads];

        for (int i = 0; i < NumHeads; i++)
        {
            Wq[i] = new Matrix(HeadDim, EmbeddingDim);
            Wk[i] = new Matrix(HeadDim, EmbeddingDim);
            Wv[i] = new Matrix(HeadDim, EmbeddingDim);
            InitializeMatrix(Wq[i]);
            InitializeMatrix(Wk[i]);
            InitializeMatrix(Wv[i]);
        }

        Wo = new Matrix(EmbeddingDim, EmbeddingDim);
        InitializeMatrix(Wo);
    }

    private void InitializeMatrix(Matrix m)
    {
        Random rand = new Random();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = (rand.NextDouble() - 0.5) / m.Cols;
    }

    public Vector Forward(Vector[] inputs)
    {
        // inputs: list of embeddings for sequence positions
        int seqLength = inputs.Length;
        Vector[] outputs = new Vector[seqLength];

        for (int pos = 0; pos < seqLength; pos++)
        {
            Vector concatHeads = new Vector(EmbeddingDim);
            for (int h = 0; h < NumHeads; h++)
            {
                // Compute query, key, value for this head
                Vector q = Wq[h].Multiply(inputs[pos]);
                Vector[] k = new Vector[seqLength];
                Vector[] v = new Vector[seqLength];
                for (int t = 0; t <= pos; t++) // Autoregressive mask
                {
                    k[t] = Wk[h].Multiply(inputs[t]);
                    v[t] = Wv[h].Multiply(inputs[t]);
                }

                // Compute attention scores
                double[] scores = new double[pos + 1];
                for (int t = 0; t <= pos; t++)
                {
                    scores[t] = q.Dot(k[t]) / Math.Sqrt(HeadDim);
                }

                // Apply softmax
                double maxScore = double.MinValue;
                for (int t = 0; t <= pos; t++)
                {
                    if (scores[t] > maxScore) maxScore = scores[t];
                }

                double sumExp = 0.0;
                for (int t = 0; t <= pos; t++)
                {
                    scores[t] = Math.Exp(scores[t] - maxScore);
                    sumExp += scores[t];
                }

                for (int t = 0; t <= pos; t++)
                {
                    scores[t] /= sumExp;
                }

                // Compute weighted sum of values
                Vector headOutput = new Vector(HeadDim);
                for (int t = 0; t <= pos; t++)
                {
                    for (int i = 0; i < HeadDim; i++)
                    {
                        headOutput.Data[i] += scores[t] * v[t].Data[i];
                    }
                }

                // Concatenate head outputs
                for (int i = 0; i < HeadDim; i++)
                {
                    concatHeads.Data[h * HeadDim + i] = headOutput.Data[i];
                }
            }

            // Apply Wo
            outputs[pos] = Wo.Multiply(concatHeads);
        }

        // Return the last output (since we're predicting next token)
        return outputs[seqLength - 1];
    }
}

// Feed-Forward Network
public class FeedForwardNetwork
{
    public int EmbeddingDim;
    public int HiddenDim;
    public Matrix W1;
    public Vector b1;
    public Matrix W2;
    public Vector b2;

    public FeedForwardNetwork(int embeddingDim, int hiddenDim)
    {
        EmbeddingDim = embeddingDim;
        HiddenDim = hiddenDim;
        W1 = new Matrix(HiddenDim, EmbeddingDim);
        b1 = new Vector(HiddenDim);
        W2 = new Matrix(EmbeddingDim, HiddenDim);
        b2 = new Vector(EmbeddingDim);

        InitializeMatrix(W1);
        InitializeMatrix(W2);
        InitializeVector(b1);
        InitializeVector(b2);
    }

    private void InitializeMatrix(Matrix m)
    {
        Random rand = new Random();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = (rand.NextDouble() - 0.5) / m.Cols;
    }

    private void InitializeVector(Vector v)
    {
        for (int i = 0; i < v.Size; i++)
            v.Data[i] = 0.0;
    }

    public Vector Forward(Vector input)
    {
        Vector hidden = W1.Multiply(input);
        hidden = hidden + b1;
        hidden.ApplyFunction(x => Math.Max(0, x)); // ReLU activation

        Vector output = W2.Multiply(hidden);
        output = output + b2;
        return output;
    }
}

// Transformer Block
public class TransformerBlock
{
    public MultiHeadAttention MHA;
    public LayerNormalization LayerNorm1;
    public FeedForwardNetwork FFN;
    public LayerNormalization LayerNorm2;

    public TransformerBlock(int embeddingDim, int numHeads, int hiddenDim)
    {
        MHA = new MultiHeadAttention(embeddingDim, numHeads);
        LayerNorm1 = new LayerNormalization(embeddingDim);
        FFN = new FeedForwardNetwork(embeddingDim, hiddenDim);
        LayerNorm2 = new LayerNormalization(embeddingDim);
    }

    public Vector Forward(Vector[] inputs, int position)
    {
        // Self-Attention
        Vector attnOutput = MHA.Forward(inputs);

        // Residual Connection and Layer Norm
        Vector x = inputs[position] + attnOutput;
        x = LayerNorm1.Normalize(x);

        // Feed-Forward Network
        Vector ffnOutput = FFN.Forward(x);

        // Residual Connection and Layer Norm
        x = x + ffnOutput;
        x = LayerNorm2.Normalize(x);

        return x;
    }
}

// Transformer Model
public class TransformerModel
{
    public int VocabSize;
    public int EmbeddingDim;
    public int NumHeads;
    public int HiddenDim;
    public int NumLayers;
    public EmbeddingLayer Embedding;
    public PositionalEncoding PosEncoding;
    public TransformerBlock[] Layers;
    public Matrix ClassificationLayer;

    public TransformerModel(int vocabSize, int embeddingDim, int numHeads, int hiddenDim, int numLayers)
    {
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        NumHeads = numHeads;
        HiddenDim = hiddenDim;
        NumLayers = numLayers;

        Embedding = new EmbeddingLayer(VocabSize, EmbeddingDim);
        PosEncoding = new PositionalEncoding(EmbeddingDim);
        Layers = new TransformerBlock[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            Layers[i] = new TransformerBlock(EmbeddingDim, NumHeads, HiddenDim);
        }
        ClassificationLayer = new Matrix(VocabSize, EmbeddingDim);
        InitializeMatrix(ClassificationLayer);
    }

    private void InitializeMatrix(Matrix m)
    {
        Random rand = new Random();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = (rand.NextDouble() - 0.5) / m.Cols;
    }

    public Vector Forward(int[] inputTokens)
    {
        int seqLength = inputTokens.Length;
        Vector[] embeddings = new Vector[seqLength];
        for (int i = 0; i < seqLength; i++)
        {
            embeddings[i] = Embedding.GetEmbedding(inputTokens[i]);
            PosEncoding.ApplyRoPE(embeddings[i], i);
        }

        for (int l = 0; l < NumLayers; l++)
        {
            for (int i = 0; i < seqLength; i++)
            {
                embeddings[i] = Layers[l].Forward(embeddings, i);
            }
        }

        // Use the last token's embedding for classification
        Vector logits = ClassificationLayer.Multiply(embeddings[seqLength - 1]);

        return logits;
    }

    public Vector Softmax(Vector logits)
    {
        double maxLogit = double.MinValue;
        for (int i = 0; i < logits.Size; i++)
        {
            if (logits.Data[i] > maxLogit) maxLogit = logits.Data[i];
        }

        double sumExp = 0.0;
        Vector probs = new Vector(logits.Size);
        for (int i = 0; i < logits.Size; i++)
        {
            probs.Data[i] = Math.Exp(logits.Data[i] - maxLogit);
            sumExp += probs.Data[i];
        }

        for (int i = 0; i < logits.Size; i++)
        {
            probs.Data[i] /= sumExp;
        }

        return probs;
    }

    public double ComputeLoss(Vector probs, int targetIndex)
    {
        double loss = -Math.Log(probs.Data[targetIndex] + 1e-9);
        return loss;
    }
}

// Optimizer (Stochastic Gradient Descent)
public class SGDOptimizer
{
    public double LearningRate;

    public SGDOptimizer(double learningRate)
    {
        LearningRate = learningRate;
    }

    public void UpdateParameters(Vector param, Vector grad)
    {
        for (int i = 0; i < param.Size; i++)
        {
            param.Data[i] -= LearningRate * grad.Data[i];
        }
    }

    public void UpdateParameters(Matrix param, Matrix grad)
    {
        for (int i = 0; i < param.Rows; i++)
        for (int j = 0; j < param.Cols; j++)
        {
            param.Data[i][j] -= LearningRate * grad.Data[i][j];
        }
    }
}

// Training Loop and Next Token Prediction
public class Trainer
{
    public TransformerModel Model;
    public SGDOptimizer Optimizer;

    public Trainer(TransformerModel model, double learningRate)
    {
        Model = model;
        Optimizer = new SGDOptimizer(learningRate);
    }

    public void Train(int[][] inputSequences, int[] targetTokens, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0.0;
            int totalTokens = 0;

            for (int i = 0; i < inputSequences.Length; i++)
            {
                Vector logits = Model.Forward(inputSequences[i]);
                Vector probs = Model.Softmax(logits);
                double loss = Model.ComputeLoss(probs, targetTokens[i]);
                totalLoss += loss;
                totalTokens++;

                // Backpropagation would go here
                // Update parameters using optimizer

                // For simplicity, we are not implementing backward pass in detail
            }

            double perplexity = Math.Exp(totalLoss / totalTokens);
            Console.WriteLine($"Epoch {epoch + 1}, Loss: {totalLoss / totalTokens}, Perplexity: {perplexity}");
        }
    }

    public int PredictNextToken(int[] inputSequence)
    {
        Vector logits = Model.Forward(inputSequence);
        Vector probs = Model.Softmax(logits);

        // Get the index with the highest probability
        double maxProb = double.MinValue;
        int predictedToken = -1;
        for (int i = 0; i < probs.Size; i++)
        {
            if (probs.Data[i] > maxProb)
            {
                maxProb = probs.Data[i];
                predictedToken = i;
            }
        }
        return predictedToken;
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Sample vocabulary and data
        int vocabSize = 1000;
        int embeddingDim = 64;
        int numHeads = 8;
        int hiddenDim = 256;
        int numLayers = 2;
        int[][] inputSequences = new int[][] {
            new int[] { 1, 2, 3 },
            new int[] { 2, 3, 4 },
            new int[] { 3, 4, 5 }
        };
        int[] targetTokens = new int[] { 4, 5, 6 };

        TransformerModel model = new TransformerModel(vocabSize, embeddingDim, numHeads, hiddenDim, numLayers);
        Trainer trainer = new Trainer(model, 0.01);

        trainer.Train(inputSequences, targetTokens, epochs: 10);

        int[] testSequence = new int[] { 1, 2, 3 };
        int predictedToken = trainer.PredictNextToken(testSequence);
        Console.WriteLine($"Predicted next token: {predictedToken}");
    }
}
