using System;

namespace AutoregressiveTransformer
{
    // Vector class for handling vector operations
    class Vector
    {
        public int Size { get; set; }
        public double[] Data { get; set; }

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

        public static Vector Add(Vector a, Vector b)
        {
            if (a.Size != b.Size)
                throw new ArgumentException("Vectors must be the same size.");

            Vector result = new Vector(a.Size);
            for (int i = 0; i < a.Size; i++)
                result.Data[i] = a.Data[i] + b.Data[i];
            return result;
        }

        public static Vector Subtract(Vector a, Vector b)
        {
            if (a.Size != b.Size)
                throw new ArgumentException("Vectors must be the same size.");

            Vector result = new Vector(a.Size);
            for (int i = 0; i < a.Size; i++)
                result.Data[i] = a.Data[i] - b.Data[i];
            return result;
        }

        public static Vector Multiply(Vector a, double scalar)
        {
            Vector result = new Vector(a.Size);
            for (int i = 0; i < a.Size; i++)
                result.Data[i] = a.Data[i] * scalar;
            return result;
        }

        public static double Dot(Vector a, Vector b)
        {
            if (a.Size != b.Size)
                throw new ArgumentException("Vectors must be the same size.");

            double result = 0;
            for (int i = 0; i < a.Size; i++)
                result += a.Data[i] * b.Data[i];
            return result;
        }

        public Vector Clone()
        {
            return new Vector(Data);
        }
    }

    // Matrix class for handling matrix operations
    class Matrix
    {
        public int Rows { get; set; }
        public int Cols { get; set; }
        public double[][] Data { get; set; }

        public Matrix(int rows, int cols)
        {
            Rows = rows;
            Cols = cols;
            Data = new double[rows][];
            for (int i = 0; i < rows; i++)
                Data[i] = new double[cols];
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

        public static Matrix Multiply(Matrix a, Matrix b)
        {
            if (a.Cols != b.Rows)
                throw new ArgumentException("Matrix dimensions are not suitable for multiplication.");

            Matrix result = new Matrix(a.Rows, b.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < b.Cols; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < a.Cols; k++)
                        sum += a.Data[i][k] * b.Data[k][j];
                    result.Data[i][j] = sum;
                }
            }
            return result;
        }

        public static Vector Multiply(Matrix a, Vector b)
        {
            if (a.Cols != b.Size)
                throw new ArgumentException("Matrix and vector dimensions are not suitable for multiplication.");

            Vector result = new Vector(a.Rows);
            for (int i = 0; i < a.Rows; i++)
            {
                double sum = 0;
                for (int j = 0; j < a.Cols; j++)
                    sum += a.Data[i][j] * b.Data[j];
                result.Data[i] = sum;
            }
            return result;
        }

        public static Matrix Transpose(Matrix a)
        {
            Matrix result = new Matrix(a.Cols, a.Rows);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[j][i] = a.Data[i][j];
            return result;
        }

        public static Matrix Add(Matrix a, Matrix b)
        {
            if (a.Rows != b.Rows || a.Cols != b.Cols)
                throw new ArgumentException("Matrices must be the same size.");

            Matrix result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i][j] = a.Data[i][j] + b.Data[i][j];
            return result;
        }

        public static Matrix Subtract(Matrix a, Matrix b)
        {
            if (a.Rows != b.Rows || a.Cols != b.Cols)
                throw new ArgumentException("Matrices must be the same size.");

            Matrix result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i][j] = a.Data[i][j] - b.Data[i][j];
            return result;
        }

        public static Matrix Multiply(Matrix a, double scalar)
        {
            Matrix result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i][j] = a.Data[i][j] * scalar;
            return result;
        }

        public Matrix Clone()
        {
            return new Matrix(Data);
        }
    }

    // Embedding Layer
    class EmbeddingLayer
    {
        public int VocabSize { get; set; }
        public int EmbeddingDim { get; set; }
        public Matrix EmbeddingMatrix { get; set; }
        public Matrix EmbeddingGradients { get; set; }

        public EmbeddingLayer(int vocabSize, int embeddingDim)
        {
            VocabSize = vocabSize;
            EmbeddingDim = embeddingDim;
            EmbeddingMatrix = new Matrix(vocabSize, embeddingDim);
            EmbeddingGradients = new Matrix(vocabSize, embeddingDim);

            Random rand = new Random();
            // Initialize embeddings randomly
            for (int i = 0; i < VocabSize; i++)
                for (int j = 0; j < EmbeddingDim; j++)
                    EmbeddingMatrix.Data[i][j] = rand.NextDouble() * 0.01;
        }

        // Forward pass
        public Matrix Forward(int[] inputTokens)
        {
            int seqLength = inputTokens.Length;
            Matrix embeddings = new Matrix(seqLength, EmbeddingDim);
            for (int i = 0; i < seqLength; i++)
            {
                int tokenId = inputTokens[i];
                for (int j = 0; j < EmbeddingDim; j++)
                    embeddings.Data[i][j] = EmbeddingMatrix.Data[tokenId][j];
            }
            return embeddings;
        }

        // Backward pass
        public void Backward(Matrix gradOutput, int[] inputTokens)
        {
            for (int i = 0; i < inputTokens.Length; i++)
            {
                int tokenId = inputTokens[i];
                for (int j = 0; j < EmbeddingDim; j++)
                {
                    EmbeddingGradients.Data[tokenId][j] += gradOutput.Data[i][j];
                }
            }
        }

        // Parameter update
        public void UpdateParameters(double learningRate)
        {
            for (int i = 0; i < VocabSize; i++)
            {
                for (int j = 0; j < EmbeddingDim; j++)
                {
                    EmbeddingMatrix.Data[i][j] -= learningRate * EmbeddingGradients.Data[i][j];
                    EmbeddingGradients.Data[i][j] = 0; // Reset gradients
                }
            }
        }
    }

    // Positional Encoding
    class PositionalEncoding
    {
        public int MaxSeqLength { get; set; }
        public int EmbeddingDim { get; set; }
        public Matrix PositionalEncodings { get; set; }

        public PositionalEncoding(int maxSeqLength, int embeddingDim)
        {
            MaxSeqLength = maxSeqLength;
            EmbeddingDim = embeddingDim;
            PositionalEncodings = new Matrix(MaxSeqLength, EmbeddingDim);

            for (int pos = 0; pos < MaxSeqLength; pos++)
            {
                for (int i = 0; i < EmbeddingDim; i++)
                {
                    double angle = GetAngle(pos, i);
                    if (i % 2 == 0)
                        PositionalEncodings.Data[pos][i] = Math.Sin(angle);
                    else
                        PositionalEncodings.Data[pos][i] = Math.Cos(angle);
                }
            }
        }

        private double GetAngle(int pos, int i)
        {
            double exponent = (double)(2 * (i / 2)) / EmbeddingDim;
            return pos / Math.Pow(10000, exponent);
        }

        public Matrix AddPositionalEncoding(Matrix embeddings)
        {
            int seqLength = embeddings.Rows;
            Matrix result = new Matrix(seqLength, EmbeddingDim);
            for (int i = 0; i < seqLength; i++)
                for (int j = 0; j < EmbeddingDim; j++)
                    result.Data[i][j] = embeddings.Data[i][j] + PositionalEncodings.Data[i][j];
            return result;
        }
    }

    // Layer Normalization
    class LayerNormalization
    {
        public int EmbeddingDim { get; set; }
        public Vector Gamma { get; set; }
        public Vector Beta { get; set; }
        public Vector dGamma { get; set; }
        public Vector dBeta { get; set; }

        public LayerNormalization(int embeddingDim)
        {
            EmbeddingDim = embeddingDim;
            Gamma = new Vector(embeddingDim);
            Beta = new Vector(embeddingDim);
            dGamma = new Vector(embeddingDim);
            dBeta = new Vector(embeddingDim);

            for (int i = 0; i < EmbeddingDim; i++)
                Gamma.Data[i] = 1.0; // Initialize gamma to 1
        }

        public Matrix Forward(Matrix X)
        {
            int seqLength = X.Rows;
            Matrix normalized = new Matrix(seqLength, EmbeddingDim);

            for (int i = 0; i < seqLength; i++)
            {
                double mean = 0;
                for (int j = 0; j < EmbeddingDim; j++)
                    mean += X.Data[i][j];
                mean /= EmbeddingDim;

                double variance = 0;
                for (int j = 0; j < EmbeddingDim; j++)
                    variance += Math.Pow(X.Data[i][j] - mean, 2);
                variance /= EmbeddingDim;

                double std = Math.Sqrt(variance + 1e-6);

                for (int j = 0; j < EmbeddingDim; j++)
                {
                    double normalizedValue = (X.Data[i][j] - mean) / std;
                    normalized.Data[i][j] = Gamma.Data[j] * normalizedValue + Beta.Data[j];
                }
            }
            return normalized;
        }

        // Backward pass (omitted for brevity)
        // You would compute gradients w.r.t Gamma and Beta here

        public void UpdateParameters(double learningRate)
        {
            for (int i = 0; i < EmbeddingDim; i++)
            {
                Gamma.Data[i] -= learningRate * dGamma.Data[i];
                Beta.Data[i] -= learningRate * dBeta.Data[i];
                dGamma.Data[i] = 0;
                dBeta.Data[i] = 0;
            }
        }
    }

    // Feedforward Network
    class FeedForwardNetwork
    {
        public int EmbeddingDim { get; set; }
        public int HiddenDim { get; set; }
        public Matrix W1 { get; set; }
        public Vector b1 { get; set; }
        public Matrix W2 { get; set; }
        public Vector b2 { get; set; }

        public Matrix dW1 { get; set; }
        public Vector db1 { get; set; }
        public Matrix dW2 { get; set; }
        public Vector db2 { get; set; }

        public FeedForwardNetwork(int embeddingDim, int hiddenDim)
        {
            EmbeddingDim = embeddingDim;
            HiddenDim = hiddenDim;

            W1 = new Matrix(EmbeddingDim, HiddenDim);
            b1 = new Vector(HiddenDim);
            W2 = new Matrix(HiddenDim, EmbeddingDim);
            b2 = new Vector(EmbeddingDim);

            dW1 = new Matrix(EmbeddingDim, HiddenDim);
            db1 = new Vector(HiddenDim);
            dW2 = new Matrix(HiddenDim, EmbeddingDim);
            db2 = new Vector(EmbeddingDim);

            Random rand = new Random();

            // Initialize weights
            for (int i = 0; i < EmbeddingDim; i++)
                for (int j = 0; j < HiddenDim; j++)
                    W1.Data[i][j] = rand.NextDouble() * 0.01;

            for (int i = 0; i < HiddenDim; i++)
                for (int j = 0; j < EmbeddingDim; j++)
                    W2.Data[i][j] = rand.NextDouble() * 0.01;
        }

        public Matrix Forward(Matrix X)
        {
            int seqLength = X.Rows;
            Matrix hidden = new Matrix(seqLength, HiddenDim);
            Matrix output = new Matrix(seqLength, EmbeddingDim);

            // First linear layer + ReLU activation
            for (int i = 0; i < seqLength; i++)
            {
                for (int j = 0; j < HiddenDim; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < EmbeddingDim; k++)
                        sum += X.Data[i][k] * W1.Data[k][j];
                    sum += b1.Data[j];
                    hidden.Data[i][j] = Math.Max(0, sum); // ReLU activation
                }
            }

            // Second linear layer
            for (int i = 0; i < seqLength; i++)
            {
                for (int j = 0; j < EmbeddingDim; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < HiddenDim; k++)
                        sum += hidden.Data[i][k] * W2.Data[k][j];
                    sum += b2.Data[j];
                    output.Data[i][j] = sum;
                }
            }

            return output;
        }

        // Backward pass (omitted for brevity)
        // Compute gradients w.r.t weights and biases

        public void UpdateParameters(double learningRate)
        {
            for (int i = 0; i < W1.Rows; i++)
                for (int j = 0; j < W1.Cols; j++)
                {
                    W1.Data[i][j] -= learningRate * dW1.Data[i][j];
                    dW1.Data[i][j] = 0;
                }

            for (int i = 0; i < b1.Size; i++)
            {
                b1.Data[i] -= learningRate * db1.Data[i];
                db1.Data[i] = 0;
            }

            for (int i = 0; i < W2.Rows; i++)
                for (int j = 0; j < W2.Cols; j++)
                {
                    W2.Data[i][j] -= learningRate * dW2.Data[i][j];
                    dW2.Data[i][j] = 0;
                }

            for (int i = 0; i < b2.Size; i++)
            {
                b2.Data[i] -= learningRate * db2.Data[i];
                db2.Data[i] = 0;
            }
        }
    }

    // Scaled Dot-Product Attention
    class ScaledDotProductAttention
    {
        public static Matrix ComputeAttention(Matrix Q, Matrix K, Matrix V)
        {
            Matrix K_T = Matrix.Transpose(K);
            Matrix scores = Matrix.Multiply(Q, K_T);

            double scale = Math.Sqrt(Q.Cols);
            for (int i = 0; i < scores.Rows; i++)
                for (int j = 0; j < scores.Cols; j++)
                    scores.Data[i][j] /= scale;

            // Apply masking for autoregressive behavior
            for (int i = 0; i < scores.Rows; i++)
                for (int j = i + 1; j < scores.Cols; j++)
                    scores.Data[i][j] = double.NegativeInfinity;

            // Apply softmax to scores
            Matrix attentionWeights = Softmax(scores);

            // Compute weighted sum of values
            Matrix output = Matrix.Multiply(attentionWeights, V);

            return output;
        }

        private static Matrix Softmax(Matrix input)
        {
            Matrix result = new Matrix(input.Rows, input.Cols);
            for (int i = 0; i < input.Rows; i++)
            {
                double max = double.MinValue;
                for (int j = 0; j < input.Cols; j++)
                    if (input.Data[i][j] > max)
                        max = input.Data[i][j];

                double sum = 0;
                double[] expValues = new double[input.Cols];
                for (int j = 0; j < input.Cols; j++)
                {
                    expValues[j] = Math.Exp(input.Data[i][j] - max);
                    sum += expValues[j];
                }
                for (int j = 0; j < input.Cols; j++)
                    result.Data[i][j] = expValues[j] / sum;
            }
            return result;
        }
    }

    // Multi-Head Attention
    class MultiHeadAttention
    {
        public int NumHeads { get; set; }
        public int EmbeddingDim { get; set; }
        public int Dk { get; set; }

        public Matrix[] W_Q { get; set; }
        public Matrix[] W_K { get; set; }
        public Matrix[] W_V { get; set; }
        public Matrix W_O { get; set; }

        public Matrix[] dW_Q { get; set; }
        public Matrix[] dW_K { get; set; }
        public Matrix[] dW_V { get; set; }
        public Matrix dW_O { get; set; }

        public MultiHeadAttention(int embeddingDim, int numHeads)
        {
            EmbeddingDim = embeddingDim;
            NumHeads = numHeads;
            Dk = embeddingDim / numHeads;

            W_Q = new Matrix[NumHeads];
            W_K = new Matrix[NumHeads];
            W_V = new Matrix[NumHeads];

            dW_Q = new Matrix[NumHeads];
            dW_K = new Matrix[NumHeads];
            dW_V = new Matrix[NumHeads];

            Random rand = new Random();

            for (int h = 0; h < NumHeads; h++)
            {
                W_Q[h] = new Matrix(EmbeddingDim, Dk);
                W_K[h] = new Matrix(EmbeddingDim, Dk);
                W_V[h] = new Matrix(EmbeddingDim, Dk);

                dW_Q[h] = new Matrix(EmbeddingDim, Dk);
                dW_K[h] = new Matrix(EmbeddingDim, Dk);
                dW_V[h] = new Matrix(EmbeddingDim, Dk);

                // Initialize weights
                for (int i = 0; i < EmbeddingDim; i++)
                {
                    for (int j = 0; j < Dk; j++)
                    {
                        W_Q[h].Data[i][j] = rand.NextDouble() * 0.01;
                        W_K[h].Data[i][j] = rand.NextDouble() * 0.01;
                        W_V[h].Data[i][j] = rand.NextDouble() * 0.01;
                    }
                }
            }

            W_O = new Matrix(NumHeads * Dk, EmbeddingDim);
            dW_O = new Matrix(NumHeads * Dk, EmbeddingDim);

            // Initialize W_O
            for (int i = 0; i < W_O.Rows; i++)
                for (int j = 0; j < W_O.Cols; j++)
                    W_O.Data[i][j] = rand.NextDouble() * 0.01;
        }

        public Matrix Forward(Matrix X)
        {
            int seqLength = X.Rows;
            Matrix[] heads = new Matrix[NumHeads];

            for (int h = 0; h < NumHeads; h++)
            {
                // Compute Q, K, V
                Matrix Q = Matrix.Multiply(X, W_Q[h]);
                Matrix K = Matrix.Multiply(X, W_K[h]);
                Matrix V = Matrix.Multiply(X, W_V[h]);

                // Compute attention
                Matrix head = ScaledDotProductAttention.ComputeAttention(Q, K, V);
                heads[h] = head;
            }

            // Concatenate heads
            Matrix concatenatedHeads = ConcatenateHeads(heads);

            // Final linear layer
            Matrix output = Matrix.Multiply(concatenatedHeads, W_O);

            return output;
        }

        private Matrix ConcatenateHeads(Matrix[] heads)
        {
            int seqLength = heads[0].Rows;
            int totalDim = NumHeads * Dk;
            Matrix result = new Matrix(seqLength, totalDim);

            for (int i = 0; i < seqLength; i++)
            {
                int colIndex = 0;
                for (int h = 0; h < NumHeads; h++)
                {
                    for (int j = 0; j < Dk; j++)
                    {
                        result.Data[i][colIndex] = heads[h].Data[i][j];
                        colIndex++;
                    }
                }
            }
            return result;
        }

        // Backward pass (omitted for brevity)
        // Compute gradients w.r.t weights

        public void UpdateParameters(double learningRate)
        {
            for (int h = 0; h < NumHeads; h++)
            {
                for (int i = 0; i < W_Q[h].Rows; i++)
                    for (int j = 0; j < W_Q[h].Cols; j++)
                    {
                        W_Q[h].Data[i][j] -= learningRate * dW_Q[h].Data[i][j];
                        W_K[h].Data[i][j] -= learningRate * dW_K[h].Data[i][j];
                        W_V[h].Data[i][j] -= learningRate * dW_V[h].Data[i][j];

                        dW_Q[h].Data[i][j] = 0;
                        dW_K[h].Data[i][j] = 0;
                        dW_V[h].Data[i][j] = 0;
                    }
            }

            for (int i = 0; i < W_O.Rows; i++)
                for (int j = 0; j < W_O.Cols; j++)
                {
                    W_O.Data[i][j] -= learningRate * dW_O.Data[i][j];
                    dW_O.Data[i][j] = 0;
                }
        }
    }

    // Encoder Layer
    class EncoderLayer
    {
        public MultiHeadAttention MultiHeadAttention { get; set; }
        public LayerNormalization LayerNorm1 { get; set; }
        public FeedForwardNetwork FeedForward { get; set; }
        public LayerNormalization LayerNorm2 { get; set; }

        public EncoderLayer(int embeddingDim, int numHeads, int hiddenDim)
        {
            MultiHeadAttention = new MultiHeadAttention(embeddingDim, numHeads);
            LayerNorm1 = new LayerNormalization(embeddingDim);
            FeedForward = new FeedForwardNetwork(embeddingDim, hiddenDim);
            LayerNorm2 = new LayerNormalization(embeddingDim);
        }

        public Matrix Forward(Matrix X)
        {
            // Multi-head attention
            Matrix attnOutput = MultiHeadAttention.Forward(X);

            // Add & Norm
            Matrix attnOutputNorm = LayerNorm1.Forward(Matrix.Add(X, attnOutput));

            // Feedforward
            Matrix ffOutput = FeedForward.Forward(attnOutputNorm);

            // Add & Norm
            Matrix output = LayerNorm2.Forward(Matrix.Add(attnOutputNorm, ffOutput));

            return output;
        }

        // Backward pass (omitted for brevity)

        public void UpdateParameters(double learningRate)
        {
            MultiHeadAttention.UpdateParameters(learningRate);
            FeedForward.UpdateParameters(learningRate);
            LayerNorm1.UpdateParameters(learningRate);
            LayerNorm2.UpdateParameters(learningRate);
        }
    }

    // Transformer Encoder
    class TransformerEncoder
    {
        public int NumLayers { get; set; }
        public EncoderLayer[] EncoderLayers { get; set; }

        public TransformerEncoder(int numLayers, int embeddingDim, int numHeads, int hiddenDim)
        {
            NumLayers = numLayers;
            EncoderLayers = new EncoderLayer[numLayers];
            for (int i = 0; i < numLayers; i++)
                EncoderLayers[i] = new EncoderLayer(embeddingDim, numHeads, hiddenDim);
        }

        public Matrix Forward(Matrix X)
        {
            Matrix output = X;
            for (int i = 0; i < NumLayers; i++)
                output = EncoderLayers[i].Forward(output);
            return output;
        }

        // Backward pass (omitted for brevity)

        public void UpdateParameters(double learningRate)
        {
            for (int i = 0; i < NumLayers; i++)
                EncoderLayers[i].UpdateParameters(learningRate);
        }
    }

    // Output Layer
    class OutputLayer
    {
        public int EmbeddingDim { get; set; }
        public int VocabSize { get; set; }
        public Matrix W_out { get; set; }
        public Matrix dW_out { get; set; }

        public OutputLayer(int embeddingDim, int vocabSize)
        {
            EmbeddingDim = embeddingDim;
            VocabSize = vocabSize;
            W_out = new Matrix(EmbeddingDim, VocabSize);
            dW_out = new Matrix(EmbeddingDim, VocabSize);

            Random rand = new Random();
            for (int i = 0; i < EmbeddingDim; i++)
                for (int j = 0; j < VocabSize; j++)
                    W_out.Data[i][j] = rand.NextDouble() * 0.01;
        }

        public Matrix Forward(Matrix X)
        {
            Matrix logits = Matrix.Multiply(X, W_out);
            return logits;
        }

        // Backward pass (omitted for brevity)

        public void UpdateParameters(double learningRate)
        {
            for (int i = 0; i < W_out.Rows; i++)
                for (int j = 0; j < W_out.Cols; j++)
                {
                    W_out.Data[i][j] -= learningRate * dW_out.Data[i][j];
                    dW_out.Data[i][j] = 0;
                }
        }
    }

    // Loss Function
    class CrossEntropyLoss
    {
        public static double ComputeLoss(Matrix logits, int[] targets)
        {
            int seqLength = logits.Rows;
            int vocabSize = logits.Cols;
            double loss = 0;
            for (int i = 0; i < seqLength; i++)
            {
                double maxLogit = double.MinValue;
                for (int j = 0; j < vocabSize; j++)
                    if (logits.Data[i][j] > maxLogit)
                        maxLogit = logits.Data[i][j];

                double sumExp = 0;
                for (int j = 0; j < vocabSize; j++)
                    sumExp += Math.Exp(logits.Data[i][j] - maxLogit);

                double logProb = logits.Data[i][targets[i]] - maxLogit - Math.Log(sumExp);
                loss -= logProb;
            }
            return loss / seqLength;
        }

        // Backward pass (computes gradients)
        public static Matrix Backward(Matrix logits, int[] targets)
        {
            int seqLength = logits.Rows;
            int vocabSize = logits.Cols;
            Matrix grad = new Matrix(seqLength, vocabSize);

            for (int i = 0; i < seqLength; i++)
            {
                double maxLogit = double.MinValue;
                for (int j = 0; j < vocabSize; j++)
                    if (logits.Data[i][j] > maxLogit)
                        maxLogit = logits.Data[i][j];

                double sumExp = 0;
                double[] expValues = new double[vocabSize];
                for (int j = 0; j < vocabSize; j++)
                {
                    expValues[j] = Math.Exp(logits.Data[i][j] - maxLogit);
                    sumExp += expValues[j];
                }

                for (int j = 0; j < vocabSize; j++)
                {
                    double softmax = expValues[j] / sumExp;
                    grad.Data[i][j] = softmax;
                }
                grad.Data[i][targets[i]] -= 1.0;
            }

            // Average gradients over sequence length
            for (int i = 0; i < seqLength; i++)
                for (int j = 0; j < vocabSize; j++)
                    grad.Data[i][j] /= seqLength;

            return grad;
        }
    }

    // Transformer Model
    class TransformerModel
    {
        public EmbeddingLayer EmbeddingLayer { get; set; }
        public PositionalEncoding PositionalEncoding { get; set; }
        public TransformerEncoder Encoder { get; set; }
        public OutputLayer OutputLayer { get; set; }
        public int MaxSeqLength { get; set; }
        public int VocabSize { get; set; }
        public int EmbeddingDim { get; set; }

        public TransformerModel(int vocabSize, int maxSeqLength, int embeddingDim, int numHeads, int numLayers, int hiddenDim)
        {
            VocabSize = vocabSize;
            MaxSeqLength = maxSeqLength;
            EmbeddingDim = embeddingDim;

            EmbeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);
            PositionalEncoding = new PositionalEncoding(maxSeqLength, embeddingDim);
            Encoder = new TransformerEncoder(numLayers, embeddingDim, numHeads, hiddenDim);
            OutputLayer = new OutputLayer(embeddingDim, vocabSize);
        }

        public Matrix Forward(int[] inputTokens)
        {
            Matrix embeddings = EmbeddingLayer.Forward(inputTokens);
            Matrix positionEncoded = PositionalEncoding.AddPositionalEncoding(embeddings);
            Matrix encoderOutput = Encoder.Forward(positionEncoded);
            Matrix logits = OutputLayer.Forward(encoderOutput);
            return logits;
        }

        // Backward pass (omitted for brevity)

        public void UpdateParameters(double learningRate)
        {
            EmbeddingLayer.UpdateParameters(learningRate);
            Encoder.UpdateParameters(learningRate);
            OutputLayer.UpdateParameters(learningRate);
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Hyperparameters
            int vocabSize = 10000;
            int maxSeqLength = 20;
            int embeddingDim = 128;
            int numHeads = 8;
            int numLayers = 2;
            int hiddenDim = 256;
            double learningRate = 0.001;

            TransformerModel model = new TransformerModel(vocabSize, maxSeqLength, embeddingDim, numHeads, numLayers, hiddenDim);

            // Dummy data for demonstration
            int[] inputTokens = new int[maxSeqLength];
            int[] targetTokens = new int[maxSeqLength];

            Random rand = new Random();
            for (int i = 0; i < maxSeqLength; i++)
            {
                inputTokens[i] = rand.Next(vocabSize);
                targetTokens[i] = rand.Next(vocabSize);
            }

            // Training loop
            for (int epoch = 0; epoch < 10; epoch++)
            {
                // Forward pass
                Matrix logits = model.Forward(inputTokens);

                // Compute loss
                double loss = CrossEntropyLoss.ComputeLoss(logits, targetTokens);

                // Backward pass
                Matrix gradLoss = CrossEntropyLoss.Backward(logits, targetTokens);

                // Update parameters
                model.UpdateParameters(learningRate);

                // Compute perplexity
                double perplexity = Math.Exp(loss);

                Console.WriteLine($"Epoch {epoch + 1}, Loss: {loss}, Perplexity: {perplexity}");
            }

            // Next token prediction
            int[] testInput = new int[maxSeqLength];
            for (int i = 0; i < maxSeqLength; i++)
                testInput[i] = rand.Next(vocabSize);

            Matrix testLogits = model.Forward(testInput);
            int predictedToken = ArgMax(testLogits.Data[maxSeqLength - 1]);

            Console.WriteLine($"Predicted next token: {predictedToken}");
        }

        static int ArgMax(double[] array)
        {
            int index = 0;
            double max = array[0];
            for (int i = 1; i < array.Length; i++)
                if (array[i] > max)
                {
                    max = array[i];
                    index = i;
                }
            return index;
        }
    }
}
