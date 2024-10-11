namespace mingpt5;

// Vector class
public class Vector
{
    public int Size;
    public double[] Data;

    public Vector (int size) {
        Size = size;
        Data = new double[size];
    }

    public Vector (double[] data) {
        Size = data.Length;
        Data = new double[Size];
        Array.Copy (data, Data, Size);
    }

    public static Vector operator + (Vector a, Vector b) {
        if (a.Size != b.Size)
            throw new Exception ("Vector sizes do not match");

        Vector result = new Vector (a.Size);
        for (int i = 0; i < a.Size; i++) {
            result.Data[i] = a.Data[i] + b.Data[i];
        }

        return result;
    }

    public static Vector operator - (Vector a, Vector b) {
        if (a.Size != b.Size)
            throw new Exception ("Vector sizes do not match");

        Vector result = new Vector (a.Size);
        for (int i = 0; i < a.Size; i++) {
            result.Data[i] = a.Data[i] - b.Data[i];
        }

        return result;
    }

    public static Vector operator * (double scalar, Vector a) {
        Vector result = new Vector (a.Size);
        for (int i = 0; i < a.Size; i++) {
            result.Data[i] = scalar * a.Data[i];
        }

        return result;
    }

    public void ApplyFunction (Func<double, double> func) {
        for (int i = 0; i < Size; i++) {
            Data[i] = func (Data[i]);
        }
    }

    public double Dot (Vector other) {
        if (Size != other.Size)
            throw new Exception ("Vector sizes do not match");

        double sum = 0.0;
        for (int i = 0; i < Size; i++) {
            sum += Data[i] * other.Data[i];
        }

        return sum;
    }

    public Vector Clone () {
        return new Vector (Data);
    }
}

// Matrix class
public class Matrix
{
    public int Rows;
    public int Cols;
    public double[][] Data;

    public Matrix (int rows, int cols) {
        Rows = rows;
        Cols = cols;
        Data = new double[rows][];
        for (int i = 0; i < rows; i++) {
            Data[i] = new double[cols];
        }
    }

    public Matrix (double[][] data) {
        Rows = data.Length;
        Cols = data[0].Length;
        Data = new double[Rows][];
        for (int i = 0; i < Rows; i++) {
            Data[i] = new double[Cols];
            Array.Copy (data[i], Data[i], Cols);
        }
    }

    public static Matrix operator + (Matrix a, Matrix b) {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new Exception ("Matrix dimensions do not match");

        Matrix result = new Matrix (a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i][j] = a.Data[i][j] + b.Data[i][j];
        return result;
    }

    public static Matrix operator - (Matrix a, Matrix b) {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new Exception ("Matrix dimensions do not match");

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
            throw new Exception ("Matrix dimensions are not compatible for multiplication");

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

    public Vector Multiply (Vector v) {
        if (Cols != v.Size)
            throw new Exception ("Matrix and vector dimensions are not compatible");

        Vector result = new Vector (Rows);
        for (int i = 0; i < Rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < Cols; j++) {
                sum += Data[i][j] * v.Data[j];
            }

            result.Data[i] = sum;
        }

        return result;
    }

    public void ApplyFunction (Func<double, double> func) {
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            Data[i][j] = func (Data[i][j]);
    }

    public Matrix Clone () {
        return new Matrix (Data);
    }

    public static Matrix OuterProduct (Vector a, Vector b) {
        Matrix result = new Matrix (a.Size, b.Size);
        for (int i = 0; i < a.Size; i++)
        for (int j = 0; j < b.Size; j++)
            result.Data[i][j] = a.Data[i] * b.Data[j];
        return result;
    }
}

// Embedding Layer
public class EmbeddingLayer
{
    public int VocabSize;
    public int EmbeddingDim;
    public Matrix EmbeddingMatrix;
    public Dictionary<int, Vector> Gradients;

    public EmbeddingLayer (int vocabSize, int embeddingDim) {
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        EmbeddingMatrix = new Matrix (VocabSize, EmbeddingDim);
        Gradients = new Dictionary<int, Vector> ();
        InitializeWeights ();
    }

    private void InitializeWeights () {
        Random rand = new Random ();
        for (int i = 0; i < VocabSize; i++)
        for (int j = 0; j < EmbeddingDim; j++)
            EmbeddingMatrix.Data[i][j] = (rand.NextDouble () - 0.5) / EmbeddingDim;
    }

    public Vector GetEmbedding (int tokenIndex) {
        Vector embedding = new Vector (EmbeddingDim);
        for (int i = 0; i < EmbeddingDim; i++) {
            embedding.Data[i] = EmbeddingMatrix.Data[tokenIndex][i];
        }

        return embedding;
    }

    public void Backward (int tokenIndex, Vector grad) {
        if (!Gradients.ContainsKey (tokenIndex))
            Gradients[tokenIndex] = new Vector (EmbeddingDim);

        for (int i = 0; i < EmbeddingDim; i++) {
            Gradients[tokenIndex].Data[i] += grad.Data[i];
        }
    }

    public void UpdateParameters (double learningRate) {
        foreach (var kvp in Gradients) {
            int tokenIndex = kvp.Key;
            Vector grad = kvp.Value;
            for (int i = 0; i < EmbeddingDim; i++) {
                EmbeddingMatrix.Data[tokenIndex][i] -= learningRate * grad.Data[i];
            }
        }

        Gradients.Clear ();
    }
}

// Positional Encoding (RoPE)
public class PositionalEncoding
{
    public int EmbeddingDim;

    public PositionalEncoding (int embeddingDim) {
        EmbeddingDim = embeddingDim;
    }

    public Vector ApplyRoPE (Vector embedding, int position) {
        int halfDim = EmbeddingDim / 2;
        double theta = 10000.0;
        Vector output = embedding.Clone ();
        for (int i = 0; i < halfDim; i++) {
            double angle = position / Math.Pow (theta, (2.0 * i) / EmbeddingDim);
            double sinAngle = Math.Sin (angle);
            double cosAngle = Math.Cos (angle);
            double original1 = embedding.Data[2 * i];
            double original2 = embedding.Data[2 * i + 1];

            output.Data[2 * i] = original1 * cosAngle - original2 * sinAngle;
            output.Data[2 * i + 1] = original1 * sinAngle + original2 * cosAngle;
        }

        return output;
    }
}

// Layer Normalization
public class LayerNormalization
{
    public int FeatureSize;
    public Vector Gamma;
    public Vector Beta;
    public Vector Input;
    public Vector Normalized;
    public double Mean;
    public double Variance;

    public LayerNormalization (int featureSize) {
        FeatureSize = featureSize;
        Gamma = new Vector (featureSize);
        Beta = new Vector (featureSize);
        InitializeParameters ();
    }

    private void InitializeParameters () {
        for (int i = 0; i < FeatureSize; i++) {
            Gamma.Data[i] = 1.0;
            Beta.Data[i] = 0.0;
        }
    }

    public Vector Forward (Vector input) {
        Input = input.Clone ();
        Mean = 0.0;
        Variance = 0.0;

        for (int i = 0; i < FeatureSize; i++) {
            Mean += Input.Data[i];
        }

        Mean /= FeatureSize;

        for (int i = 0; i < FeatureSize; i++) {
            Variance += Math.Pow (Input.Data[i] - Mean, 2);
        }

        Variance /= FeatureSize;

        Normalized = new Vector (FeatureSize);
        for (int i = 0; i < FeatureSize; i++) {
            Normalized.Data[i] = (Input.Data[i] - Mean) / Math.Sqrt (Variance + 1e-6);
            Normalized.Data[i] = Gamma.Data[i] * Normalized.Data[i] + Beta.Data[i];
        }

        return Normalized;
    }

    public Vector Backward (Vector dout) {
        Vector dGamma = new Vector (FeatureSize);
        Vector dBeta = new Vector (FeatureSize);
        Vector dx = new Vector (FeatureSize);

        double invStd = 1.0 / Math.Sqrt (Variance + 1e-6);
        Vector xHat = new Vector (FeatureSize);
        for (int i = 0; i < FeatureSize; i++) {
            xHat.Data[i] = (Input.Data[i] - Mean) * invStd;
        }

        for (int i = 0; i < FeatureSize; i++) {
            dGamma.Data[i] += dout.Data[i] * xHat.Data[i];
            dBeta.Data[i] += dout.Data[i];
        }

        for (int i = 0; i < FeatureSize; i++) {
            double dXhat = dout.Data[i] * Gamma.Data[i];
            double dVar = -0.5 * dXhat * (Input.Data[i] - Mean) * Math.Pow (Variance + 1e-6, -1.5);
            double dMean = -dXhat * invStd;
            dx.Data[i] += dXhat * invStd + dVar * 2.0 * (Input.Data[i] - Mean) / FeatureSize + dMean / FeatureSize;
        }

        // Update parameters
        for (int i = 0; i < FeatureSize; i++) {
            Gamma.Data[i] -= dout.Data[i] * xHat.Data[i];
            Beta.Data[i] -= dout.Data[i];
        }

        return dx;
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

    // Cache for backward pass
    public Vector[] Q;
    public Vector[][] K;
    public Vector[][] V;
    public double[][] AttentionWeights;
    public Vector[] Inputs;

    public MultiHeadAttention (int embeddingDim, int numHeads) {
        EmbeddingDim = embeddingDim;
        NumHeads = numHeads;
        HeadDim = EmbeddingDim / NumHeads;

        Wq = new Matrix[NumHeads];
        Wk = new Matrix[NumHeads];
        Wv = new Matrix[NumHeads];

        for (int i = 0; i < NumHeads; i++) {
            Wq[i] = new Matrix (HeadDim, EmbeddingDim);
            Wk[i] = new Matrix (HeadDim, EmbeddingDim);
            Wv[i] = new Matrix (HeadDim, EmbeddingDim);
            InitializeMatrix (Wq[i]);
            InitializeMatrix (Wk[i]);
            InitializeMatrix (Wv[i]);
        }

        Wo = new Matrix (EmbeddingDim, EmbeddingDim);
        InitializeMatrix (Wo);
    }

    private void InitializeMatrix (Matrix m) {
        Random rand = new Random ();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = (rand.NextDouble () - 0.5) / m.Cols;
    }

    public Vector Forward (Vector[] inputs, int position) {
        Inputs = inputs;
        int seqLength = inputs.Length;
        Q = new Vector[NumHeads];
        K = new Vector[NumHeads][];
        V = new Vector[NumHeads][];
        AttentionWeights = new double[NumHeads][];
        Vector concatHeads = new Vector (EmbeddingDim);

        for (int h = 0; h < NumHeads; h++) {
            // Compute query, key, value for this head
            Q[h] = Wq[h].Multiply (inputs[position]);
            K[h] = new Vector[position + 1];
            V[h] = new Vector[position + 1];
            AttentionWeights[h] = new double[position + 1];
            for (int t = 0; t <= position; t++) // Autoregressive mask
            {
                K[h][t] = Wk[h].Multiply (inputs[t]);
                V[h][t] = Wv[h].Multiply (inputs[t]);
            }

            // Compute attention scores
            double[] scores = new double[position + 1];
            for (int t = 0; t <= position; t++) {
                scores[t] = Q[h].Dot (K[h][t]) / Math.Sqrt (HeadDim);
            }

            // Apply softmax
            double maxScore = double.MinValue;
            for (int t = 0; t <= position; t++) {
                if (scores[t] > maxScore) maxScore = scores[t];
            }

            double sumExp = 0.0;
            for (int t = 0; t <= position; t++) {
                scores[t] = Math.Exp (scores[t] - maxScore);
                sumExp += scores[t];
            }

            for (int t = 0; t <= position; t++) {
                scores[t] /= sumExp;
            }

            AttentionWeights[h] = scores;

            // Compute weighted sum of values
            Vector headOutput = new Vector (HeadDim);
            for (int t = 0; t <= position; t++) {
                for (int i = 0; i < HeadDim; i++) {
                    headOutput.Data[i] += scores[t] * V[h][t].Data[i];
                }
            }

            // Concatenate head outputs
            for (int i = 0; i < HeadDim; i++) {
                concatHeads.Data[h * HeadDim + i] = headOutput.Data[i];
            }
        }

        // Apply Wo
        Vector output = Wo.Multiply (concatHeads);
        return output;
    }

    public Vector Backward (Vector dout, int position) {
        Vector dConcatHeads = Wo.Transpose ().Multiply (dout);
        // Gradients for Wo
        Matrix dWo = Matrix.OuterProduct (dout, dConcatHeads);

        // Initialize gradients
        Vector[] dInputs = new Vector[Inputs.Length];
        for (int i = 0; i < Inputs.Length; i++) {
            dInputs[i] = new Vector (EmbeddingDim);
        }

        // Backpropagate through heads
        for (int h = 0; h < NumHeads; h++) {
            Vector dHeadOutput = new Vector (HeadDim);
            for (int i = 0; i < HeadDim; i++) {
                dHeadOutput.Data[i] = dConcatHeads.Data[h * HeadDim + i];
            }

            // Backprop through weighted sum of values
            Vector[] dV = new Vector[position + 1];
            double[] dScores = new double[position + 1];
            for (int t = 0; t <= position; t++) {
                dV[t] = new Vector (HeadDim);
            }

            for (int t = 0; t <= position; t++) {
                double attnWeight = AttentionWeights[h][t];
                for (int i = 0; i < HeadDim; i++) {
                    dV[t].Data[i] += dHeadOutput.Data[i] * attnWeight;
                }
            }

            // Backprop through attention weights
            for (int t = 0; t <= position; t++) {
                double attnWeight = AttentionWeights[h][t];
                dScores[t] = 0.0;
                for (int i = 0; i < HeadDim; i++) {
                    dScores[t] += dHeadOutput.Data[i] * V[h][t].Data[i];
                }

                dScores[t] *= attnWeight;
            }

            // Backprop through scores to Q, K
            Vector dQ = new Vector (HeadDim);
            Vector[] dK = new Vector[position + 1];
            for (int t = 0; t <= position; t++) {
                dK[t] = new Vector (HeadDim);
            }

            for (int t = 0; t <= position; t++) {
                double scale = 1.0 / Math.Sqrt (HeadDim);
                for (int i = 0; i < HeadDim; i++) {
                    double temp = dScores[t] * scale;
                    dQ.Data[i] += temp * K[h][t].Data[i];
                    dK[t].Data[i] += temp * Q[h].Data[i];
                }
            }

            // Backprop through linear layers Wq, Wk, Wv
            Vector dInputQ = Wq[h].Transpose ().Multiply (dQ);
            Matrix dWq = Matrix.OuterProduct (dQ, Inputs[position]);

            for (int t = 0; t <= position; t++) {
                Vector dInputK = Wk[h].Transpose ().Multiply (dK[t]);
                Vector dInputV = Wv[h].Transpose ().Multiply (dV[t]);
                Matrix dWk = Matrix.OuterProduct (dK[t], Inputs[t]);
                Matrix dWv = Matrix.OuterProduct (dV[t], Inputs[t]);

                // Accumulate gradients
                dInputs[t] = dInputs[t] + dInputK + dInputV;
                // Update Wk[h] and Wv[h]
                for (int r = 0; r < Wk[h].Rows; r++)
                for (int c = 0; c < Wk[h].Cols; c++) {
                    Wk[h].Data[r][c] -= dWk.Data[r][c];
                    Wv[h].Data[r][c] -= dWv.Data[r][c];
                }
            }

            dInputs[position] = dInputs[position] + dInputQ;
            // Update Wq[h]
            for (int r = 0; r < Wq[h].Rows; r++)
            for (int c = 0; c < Wq[h].Cols; c++) {
                Wq[h].Data[r][c] -= dWq.Data[r][c];
            }
        }

        // Update Wo
        for (int r = 0; r < Wo.Rows; r++)
        for (int c = 0; c < Wo.Cols; c++) {
            Wo.Data[r][c] -= dWo.Data[r][c];
        }

        return dInputs[position];
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
    public Vector Hidden;
    public Vector Input;

    public FeedForwardNetwork (int embeddingDim, int hiddenDim) {
        EmbeddingDim = embeddingDim;
        HiddenDim = hiddenDim;
        W1 = new Matrix (HiddenDim, EmbeddingDim);
        b1 = new Vector (HiddenDim);
        W2 = new Matrix (EmbeddingDim, HiddenDim);
        b2 = new Vector (EmbeddingDim);

        InitializeMatrix (W1);
        InitializeMatrix (W2);
        InitializeVector (b1);
        InitializeVector (b2);
    }

    private void InitializeMatrix (Matrix m) {
        Random rand = new Random ();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = (rand.NextDouble () - 0.5) / m.Cols;
    }

    private void InitializeVector (Vector v) {
        for (int i = 0; i < v.Size; i++)
            v.Data[i] = 0.0;
    }

    public Vector Forward (Vector input) {
        Input = input.Clone ();
        Hidden = W1.Multiply (Input);
        Hidden = Hidden + b1;
        // ReLU activation
        for (int i = 0; i < Hidden.Size; i++) {
            Hidden.Data[i] = Math.Max (0, Hidden.Data[i]);
        }

        Vector output = W2.Multiply (Hidden);
        output = output + b2;
        return output;
    }

    public Vector Backward (Vector dout) {
        Vector dHidden = W2.Transpose ().Multiply (dout);
        // Gradients for W2 and b2
        Matrix dW2 = Matrix.OuterProduct (dout, Hidden);
        Vector db2 = dout.Clone ();

        // Backprop through ReLU
        for (int i = 0; i < dHidden.Size; i++) {
            if (Hidden.Data[i] <= 0) {
                dHidden.Data[i] = 0;
            }
        }

        // Gradients for W1 and b1
        Matrix dW1 = Matrix.OuterProduct (dHidden, Input);
        Vector db1 = dHidden.Clone ();
        Vector dx = W1.Transpose ().Multiply (dHidden);

        // Update parameters
        for (int r = 0; r < W1.Rows; r++)
        for (int c = 0; c < W1.Cols; c++) {
            W1.Data[r][c] -= dW1.Data[r][c];
        }

        for (int i = 0; i < b1.Size; i++) {
            b1.Data[i] -= db1.Data[i];
        }

        for (int r = 0; r < W2.Rows; r++)
        for (int c = 0; c < W2.Cols; c++) {
            W2.Data[r][c] -= dW2.Data[r][c];
        }

        for (int i = 0; i < b2.Size; i++) {
            b2.Data[i] -= db2.Data[i];
        }

        return dx;
    }
}

// Transformer Block
public class TransformerBlock
{
    public MultiHeadAttention MHA;
    public LayerNormalization LayerNorm1;
    public FeedForwardNetwork FFN;
    public LayerNormalization LayerNorm2;
    public Vector Input;
    public Vector AttnOutput;
    public Vector FFNOutput;
    public int Position;

    public TransformerBlock (int embeddingDim, int numHeads, int hiddenDim) {
        MHA = new MultiHeadAttention (embeddingDim, numHeads);
        LayerNorm1 = new LayerNormalization (embeddingDim);
        FFN = new FeedForwardNetwork (embeddingDim, hiddenDim);
        LayerNorm2 = new LayerNormalization (embeddingDim);
    }

    public Vector Forward (Vector[] inputs, int position) {
        Input = inputs[position].Clone ();
        Position = position;
        // Self-Attention
        AttnOutput = MHA.Forward (inputs, position);

        // Residual Connection and Layer Norm
        Vector x = Input + AttnOutput;
        x = LayerNorm1.Forward (x);

        // Feed-Forward Network
        FFNOutput = FFN.Forward (x);

        // Residual Connection and Layer Norm
        x = x + FFNOutput;
        x = LayerNorm2.Forward (x);

        return x;
    }

    public Vector Backward (Vector dout) {
        // Backward through Layer Norm 2
        Vector dLN2 = LayerNorm2.Backward (dout);

        // Backward through residual connection
        Vector dFFNOutput = dLN2.Clone ();
        Vector dResidual = dLN2.Clone ();

        // Backward through Feed-Forward Network
        Vector dFFNInput = FFN.Backward (dFFNOutput);

        // Backward through Layer Norm 1
        Vector dLN1 = LayerNorm1.Backward (dFFNInput + dResidual);

        // Backward through residual connection
        Vector dAttnOutput = dLN1.Clone ();
        Vector dInput = dLN1.Clone ();

        // Backward through Multi-Head Attention
        Vector dMHAInput = MHA.Backward (dAttnOutput, Position);

        // Sum gradients
        dInput = dInput + dMHAInput;

        return dInput;
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
    public Vector[] Embeddings;

    public TransformerModel (int vocabSize, int embeddingDim, int numHeads, int hiddenDim, int numLayers) {
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        NumHeads = numHeads;
        HiddenDim = hiddenDim;
        NumLayers = numLayers;

        Embedding = new EmbeddingLayer (VocabSize, EmbeddingDim);
        PosEncoding = new PositionalEncoding (EmbeddingDim);
        Layers = new TransformerBlock[NumLayers];
        for (int i = 0; i < NumLayers; i++) {
            Layers[i] = new TransformerBlock (EmbeddingDim, NumHeads, HiddenDim);
        }

        ClassificationLayer = new Matrix (VocabSize, EmbeddingDim);
        InitializeMatrix (ClassificationLayer);
    }

    private void InitializeMatrix (Matrix m) {
        Random rand = new Random ();
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m.Data[i][j] = (rand.NextDouble () - 0.5) / m.Cols;
    }

    public Vector Forward (int[] inputTokens) {
        int seqLength = inputTokens.Length;
        Embeddings = new Vector[seqLength];
        for (int i = 0; i < seqLength; i++) {
            Vector embedding = Embedding.GetEmbedding (inputTokens[i]);
            embedding = PosEncoding.ApplyRoPE (embedding, i);
            Embeddings[i] = embedding;
        }

        for (int l = 0; l < NumLayers; l++) {
            for (int i = 0; i < seqLength; i++) {
                Embeddings[i] = Layers[l].Forward (Embeddings, i);
            }
        }

        // Use the last token's embedding for classification
        Vector logits = ClassificationLayer.Multiply (Embeddings[seqLength - 1]);

        return logits;
    }

    public Vector Backward (Vector dLogits, int[] inputTokens) {
        // Backprop through classification layer
        Vector dLastEmbedding = ClassificationLayer.Transpose ().Multiply (dLogits);

        // Gradient for classification layer
        Matrix dClassificationLayer = Matrix.OuterProduct (dLogits, Embeddings[inputTokens.Length - 1]);

        // Update classification layer
        for (int r = 0; r < ClassificationLayer.Rows; r++)
        for (int c = 0; c < ClassificationLayer.Cols; c++) {
            ClassificationLayer.Data[r][c] -= dClassificationLayer.Data[r][c];
        }

        // Backprop through transformer layers
        Vector[] dEmbeddings = new Vector[inputTokens.Length];
        for (int i = 0; i < inputTokens.Length; i++) {
            dEmbeddings[i] = new Vector (EmbeddingDim);
        }

        dEmbeddings[inputTokens.Length - 1] = dLastEmbedding;

        for (int l = NumLayers - 1; l >= 0; l--) {
            for (int i = inputTokens.Length - 1; i >= 0; i--) {
                dEmbeddings[i] = Layers[l].Backward (dEmbeddings[i]);
            }
        }

        // Backprop through embedding layer
        for (int i = 0; i < inputTokens.Length; i++) {
            Embedding.Backward (inputTokens[i], dEmbeddings[i]);
        }

        return null; // No need to return anything
    }

    public Vector Softmax (Vector logits) {
        double maxLogit = double.MinValue;
        for (int i = 0; i < logits.Size; i++) {
            if (logits.Data[i] > maxLogit) maxLogit = logits.Data[i];
        }

        double sumExp = 0.0;
        Vector probs = new Vector (logits.Size);
        for (int i = 0; i < logits.Size; i++) {
            probs.Data[i] = Math.Exp (logits.Data[i] - maxLogit);
            sumExp += probs.Data[i];
        }

        for (int i = 0; i < logits.Size; i++) {
            probs.Data[i] /= sumExp;
        }

        return probs;
    }

    public double ComputeLoss (Vector probs, int targetIndex) {
        double loss = -Math.Log (probs.Data[targetIndex] + 1e-9);
        return loss;
    }

    public Vector ComputeLossGradient (Vector probs, int targetIndex) {
        Vector grad = probs.Clone ();
        grad.Data[targetIndex] -= 1.0;
        return grad;
    }
}

// Training Loop and Next Token Prediction
public class Trainer
{
    public TransformerModel Model;
    public double LearningRate;

    public Trainer (TransformerModel model, double learningRate) {
        Model = model;
        LearningRate = learningRate;
    }

    public void Train (Func<(int[], int)> getSample, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            int totalTokens = 0;

            var (inputSequence, targetToken) = getSample ();

            // Forward pass
            Vector logits = Model.Forward (inputSequence);
            Vector probs = Model.Softmax (logits);
            double loss = Model.ComputeLoss (probs, targetToken);
            totalLoss += loss;
            totalTokens++;

            // Backward pass
            Vector dLoss = Model.ComputeLossGradient (probs, targetToken);
            Model.Backward (dLoss, inputSequence);

            // Update parameters
            Model.Embedding.UpdateParameters (LearningRate);
            // Other parameters are updated within their respective backward methods

            double perplexity = Math.Exp (totalLoss / totalTokens);
            Console.WriteLine ($"Epoch {epoch + 1}, Loss: {totalLoss / totalTokens}, Perplexity: {perplexity}");
        }
    }

    public int PredictNextToken (int[] inputSequence) {
        Vector logits = Model.Forward (inputSequence);
        Vector probs = Model.Softmax (logits);

        // Get the index with the highest probability
        double maxProb = double.MinValue;
        int predictedToken = -1;
        for (int i = 0; i < probs.Size; i++) {
            if (probs.Data[i] > maxProb) {
                maxProb = probs.Data[i];
                predictedToken = i;
            }
        }

        return predictedToken;
    }
}

class MinGPT5Test
{
    public static void run () {
        int embeddingDim = 96;
        int numHeads = 6;
        int hiddenDim = 256;
        int numLayers = 4;

        var data = LoadData (
            sequenceLength: 8,
            out var vocabSize,
            out var vocabulary);

        TransformerModel model = new TransformerModel (vocabSize, embeddingDim, numHeads, hiddenDim, numLayers);
        Trainer trainer = new Trainer (model, 0.0005);

        trainer.Train (data, epochs: 100);

        int[] testSequence = new int[] {
            1,
            2,
            3
        };
        int predictedToken = trainer.PredictNextToken (testSequence);
        Console.WriteLine ($"Predicted next token: {predictedToken}");
    }

    static Func<(int[], int)> LoadData (int sequenceLength, out int vocabularySize, out char[] vocabulary) {
        var text = File.ReadAllText ("resources/tinyshakespeare.txt");
        var sourceCharacters = text.ToArray ();

        vocabulary = sourceCharacters
            .Distinct ()
            .Order ()
            .ToArray ();
        vocabularySize = vocabulary.Length;

        Console.WriteLine ($"vocabulary: {string.Join ("", vocabulary)}");

        var charToIndex = vocabulary
            .Select ((c, index) => (c, index))
            .ToDictionary ();

        var data = sourceCharacters.Select (c => charToIndex[c]).ToArray ();

        var rnd = new Random ();

        return () => {
            var sample = data
                .Skip (rnd.Next (0, data.Length - sequenceLength - 1))
                .Take (sequenceLength + 1)
                .ToArray ();

            return (sample.Take (sequenceLength).ToArray (), sample[^1]);
        };
    }
}
