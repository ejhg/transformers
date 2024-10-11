using System;

public class GPTModel
{
    public int VocabSize, EmbeddingSize, NumHeads, NumLayers, MaxSeqLen;
    public EmbeddingLayer TokenEmbedding;
    public EmbeddingLayer PositionalEmbedding;
    public TransformerBlock[] Layers;
    public LinearLayer FinalLayer;

    public GPTModel(int vocabSize, int embeddingSize, int numHeads, int numLayers, int maxSeqLen)
    {
        VocabSize = vocabSize;
        EmbeddingSize = embeddingSize;
        NumHeads = numHeads;
        NumLayers = numLayers;
        MaxSeqLen = maxSeqLen;

        TokenEmbedding = new EmbeddingLayer(vocabSize, embeddingSize);
        PositionalEmbedding = new EmbeddingLayer(maxSeqLen, embeddingSize);
        Layers = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++)
            Layers[i] = new TransformerBlock(embeddingSize, numHeads);
        FinalLayer = new LinearLayer(embeddingSize, vocabSize);
    }

    public Matrix[] Forward(int[][] batchInputIds)
    {
        int batchSize = batchInputIds.Length;
        Matrix[] x = new Matrix[batchSize];

        // Token and positional embeddings
        for (int b = 0; b < batchSize; b++)
        {
            var tokenEmb = TokenEmbedding.Forward(batchInputIds[b]);
            var positions = new int[batchInputIds[b].Length];
            for (int i = 0; i < positions.Length; i++)
                positions[i] = i;
            var posEmb = PositionalEmbedding.Forward(positions);
            x[b] = tokenEmb + posEmb;
        }

        // Transformer layers
        foreach (var layer in Layers)
        {
            for (int b = 0; b < batchSize; b++)
                x[b] = layer.Forward(x[b]);
        }

        // Final linear layer
        for (int b = 0; b < batchSize; b++)
        {
            x[b] = FinalLayer.Forward(x[b]);
        }

        return x; // Returns an array of matrices representing logits for each batch item
    }

    public void Backward(Matrix[] dLogits, int[][] batchInputIds)
    {
        int batchSize = dLogits.Length;
        Matrix[] dX = new Matrix[batchSize];

        // Backward through final linear layer
        for (int b = 0; b < batchSize; b++)
            dX[b] = FinalLayer.Backward(dLogits[b]);

        // Backward through transformer layers
        for (int i = Layers.Length - 1; i >= 0; i--)
        {
            for (int b = 0; b < batchSize; b++)
                dX[b] = Layers[i].Backward(dX[b]);
        }

        // Backward through embeddings
        for (int b = 0; b < batchSize; b++)
        {
            Matrix dPosEmb = dX[b];
            TokenEmbedding.Backward(dX[b], batchInputIds[b]);
            PositionalEmbedding.Backward(dPosEmb, GetPositions(batchInputIds[b].Length));
        }
    }

    private int[] GetPositions(int length)
    {
        var positions = new int[length];
        for (int i = 0; i < length; i++)
            positions[i] = i;
        return positions;
    }
}

public class EmbeddingLayer
{
    public int VocabSize, EmbeddingSize;
    public Matrix Weights;
    public Matrix GradWeights;

    public EmbeddingLayer(int vocabSize, int embeddingSize)
    {
        VocabSize = vocabSize; EmbeddingSize = embeddingSize;
        Weights = Matrix.Random(vocabSize, embeddingSize);
        GradWeights = new Matrix(vocabSize, embeddingSize);
    }

    public Matrix Forward(int[] inputIds)
    {
        var result = new Matrix(inputIds.Length, EmbeddingSize);
        for (int i = 0; i < inputIds.Length; i++)
            for (int j = 0; j < EmbeddingSize; j++)
                result.Data[i, j] = Weights.Data[inputIds[i], j];
        return result;
    }

    public void Backward(Matrix dOutput, int[] inputIds)
    {
        for (int i = 0; i < inputIds.Length; i++)
            for (int j = 0; j < EmbeddingSize; j++)
                GradWeights.Data[inputIds[i], j] += dOutput.Data[i, j];
    }
}

public class TransformerBlock
{
    public int EmbeddingSize, NumHeads;
    public MultiHeadSelfAttention SelfAttention;
    public LayerNorm LayerNorm1;
    public FeedForward FFN;
    public LayerNorm LayerNorm2;
    private Matrix Residual1;
    private Matrix Residual2;

    public TransformerBlock(int embeddingSize, int numHeads)
    {
        EmbeddingSize = embeddingSize;
        NumHeads = numHeads;
        SelfAttention = new MultiHeadSelfAttention(embeddingSize, numHeads);
        LayerNorm1 = new LayerNorm(embeddingSize);
        FFN = new FeedForward(embeddingSize);
        LayerNorm2 = new LayerNorm(embeddingSize);
    }

    public Matrix Forward(Matrix x)
    {
        Residual1 = x;
        var attnOutput = SelfAttention.Forward(x);
        x = x + attnOutput;
        x = LayerNorm1.Forward(x);

        Residual2 = x;
        var ffnOutput = FFN.Forward(x);
        x = x + ffnOutput;
        x = LayerNorm2.Forward(x);
        return x;
    }

    public Matrix Backward(Matrix dOutput)
    {
        dOutput = LayerNorm2.Backward(dOutput);
        var dFFN = dOutput;
        var dResidual2 = dOutput;

        dFFN = FFN.Backward(dFFN);
        dResidual2 += dFFN;

        dOutput = dResidual2;
        dOutput = LayerNorm1.Backward(dOutput);
        var dAttn = dOutput;
        var dResidual1 = dOutput;

        dAttn = SelfAttention.Backward(dAttn);
        dResidual1 += dAttn;

        return dResidual1;
    }
}

public class MultiHeadSelfAttention
{
    public int EmbeddingSize, NumHeads, HeadSize;
    public Matrix Wq, Wk, Wv, Wo;
    public Matrix GradWq, GradWk, GradWv, GradWo;

    private Matrix Q, K, V, AttentionOutput;
    private Matrix[] Q_heads, K_heads, V_heads, AttnWeights, HeadOutputs;

    public MultiHeadSelfAttention(int embeddingSize, int numHeads)
    {
        EmbeddingSize = embeddingSize;
        NumHeads = numHeads;
        HeadSize = embeddingSize / numHeads;

        Wq = Matrix.Random(embeddingSize, embeddingSize);
        Wk = Matrix.Random(embeddingSize, embeddingSize);
        Wv = Matrix.Random(embeddingSize, embeddingSize);
        Wo = Matrix.Random(embeddingSize, embeddingSize);

        GradWq = new Matrix(embeddingSize, embeddingSize);
        GradWk = new Matrix(embeddingSize, embeddingSize);
        GradWv = new Matrix(embeddingSize, embeddingSize);
        GradWo = new Matrix(embeddingSize, embeddingSize);
    }

    public Matrix Forward(Matrix x)
    {
        Q = x * Wq;
        K = x * Wk;
        V = x * Wv;

        Q_heads = SplitHeads(Q);
        K_heads = SplitHeads(K);
        V_heads = SplitHeads(V);

        HeadOutputs = new Matrix[NumHeads];
        AttnWeights = new Matrix[NumHeads];

        for (int i = 0; i < NumHeads; i++)
        {
            var scores = (Q_heads[i] * K_heads[i].Transpose()) / Math.Sqrt(HeadSize);
            var attn_weights = Softmax(scores);
            var attn_output = attn_weights * V_heads[i];

            AttnWeights[i] = attn_weights;
            HeadOutputs[i] = attn_output;
        }

        var concat = ConcatenateHeads(HeadOutputs);
        AttentionOutput = concat;
        var output = concat * Wo;
        return output;
    }

    public Matrix Backward(Matrix dOutput)
    {
        // Backprop through Wo
        var dConcat = dOutput * Wo.Transpose();
        GradWo += AttentionOutput.Transpose() * dOutput;

        // Split gradients for heads
        var dHeadOutputs = SplitHeads(dConcat);

        var dQ = new Matrix(Q.Rows, Q.Cols);
        var dK = new Matrix(K.Rows, K.Cols);
        var dV = new Matrix(V.Rows, V.Cols);

        for (int i = 0; i < NumHeads; i++)
        {
            var dAttnOutput = dHeadOutputs[i];
            var dAttnWeights = dAttnOutput * V_heads[i].Transpose();
            var dV_head = AttnWeights[i].Transpose() * dAttnOutput;

            // Backprop through Softmax
            var dScores = SoftmaxBackward(AttnWeights[i], dAttnWeights);

            var dQ_head = dScores * K_heads[i];
            var dK_head = dScores.Transpose() * Q_heads[i];

            // Aggregate gradients
            AddToMatrix(dV, dV_head, i);
            AddToMatrix(dQ, dQ_head, i);
            AddToMatrix(dK, dK_head, i);
        }

        // Backprop through Wq, Wk, Wv
        GradWq += Q.Transpose() * dQ;
        GradWk += K.Transpose() * dK;
        GradWv += V.Transpose() * dV;

        var dInput = dQ * Wq.Transpose() + dK * Wk.Transpose() + dV * Wv.Transpose();

        return dInput;
    }

    private void AddToMatrix(Matrix fullMatrix, Matrix dHeadMatrix, int headIndex)
    {
        int offset = headIndex * HeadSize;
        for (int i = 0; i < fullMatrix.Rows; i++)
            for (int j = 0; j < HeadSize; j++)
                fullMatrix.Data[i, offset + j] += dHeadMatrix.Data[i, j];
    }

    private Matrix[] SplitHeads(Matrix x)
    {
        var heads = new Matrix[NumHeads];
        for (int i = 0; i < NumHeads; i++)
        {
            heads[i] = new Matrix(x.Rows, HeadSize);
            for (int j = 0; j < x.Rows; j++)
                for (int k = 0; k < HeadSize; k++)
                    heads[i].Data[j, k] = x.Data[j, i * HeadSize + k];
        }
        return heads;
    }

    private Matrix ConcatenateHeads(Matrix[] heads)
    {
        var seq_len = heads[0].Rows;
        var concat = new Matrix(seq_len, EmbeddingSize);
        for (int i = 0; i < NumHeads; i++)
            for (int j = 0; j < seq_len; j++)
                for (int k = 0; k < HeadSize; k++)
                    concat.Data[j, i * HeadSize + k] = heads[i].Data[j, k];
        return concat;
    }

    private Matrix Softmax(Matrix x)
    {
        var result = new Matrix(x.Rows, x.Cols);
        for (int i = 0; i < x.Rows; i++)
        {
            double max = double.NegativeInfinity;
            for (int j = 0; j < x.Cols; j++)
                if (x.Data[i, j] > max)
                    max = x.Data[i, j];
            double sum = 0.0;
            for (int j = 0; j < x.Cols; j++)
            {
                result.Data[i, j] = Math.Exp(x.Data[i, j] - max);
                sum += result.Data[i, j];
            }
            for (int j = 0; j < x.Cols; j++)
                result.Data[i, j] /= sum;
        }
        return result;
    }

    private Matrix SoftmaxBackward(Matrix softmaxOutput, Matrix dOutput)
    {
        var result = new Matrix(softmaxOutput.Rows, softmaxOutput.Cols);
        for (int i = 0; i < softmaxOutput.Rows; i++)
        {
            for (int j = 0; j < softmaxOutput.Cols; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < softmaxOutput.Cols; k++)
                {
                    double delta = (j == k) ? 1.0 : 0.0;
                    sum += dOutput.Data[i, k] * softmaxOutput.Data[i, k] * (delta - softmaxOutput.Data[i, j]);
                }
                result.Data[i, j] = sum;
            }
        }
        return result;
    }
}

public class FeedForward
{
    public int EmbeddingSize;
    public LinearLayer Linear1;
    public LinearLayer Linear2;
    private Matrix Input;
    private Matrix Hidden;

    public FeedForward(int embeddingSize)
    {
        EmbeddingSize = embeddingSize;
        Linear1 = new LinearLayer(embeddingSize, embeddingSize * 4);
        Linear2 = new LinearLayer(embeddingSize * 4, embeddingSize);
    }

    public Matrix Forward(Matrix x)
    {
        Input = x;
        Hidden = Linear1.Forward(x);
        Hidden = Relu(Hidden);
        var output = Linear2.Forward(Hidden);
        return output;
    }

    public Matrix Backward(Matrix dOutput)
    {
        var dHidden = Linear2.Backward(dOutput);
        dHidden = ReluBackward(Hidden, dHidden);
        var dInput = Linear1.Backward(dHidden);
        return dInput;
    }

    private Matrix Relu(Matrix x)
    {
        var result = new Matrix(x.Rows, x.Cols);
        for (int i = 0; i < x.Rows; i++)
            for (int j = 0; j < x.Cols; j++)
                result.Data[i, j] = Math.Max(0, x.Data[i, j]);
        return result;
    }

    private Matrix ReluBackward(Matrix x, Matrix dOutput)
    {
        var result = new Matrix(dOutput.Rows, dOutput.Cols);
        for (int i = 0; i < x.Rows; i++)
            for (int j = 0; j < x.Cols; j++)
                result.Data[i, j] = x.Data[i, j] > 0 ? dOutput.Data[i, j] : 0;
        return result;
    }
}

public class LayerNorm
{
    public int EmbeddingSize;
    public double[] Gamma;
    public double[] Beta;
    public double[] GradGamma;
    public double[] GradBeta;
    private double[] Mean;
    private double[] Variance;
    private Matrix Input;

    public LayerNorm(int embeddingSize)
    {
        EmbeddingSize = embeddingSize;
        Gamma = new double[embeddingSize];
        Beta = new double[embeddingSize];
        GradGamma = new double[embeddingSize];
        GradBeta = new double[embeddingSize];
        for (int i = 0; i < embeddingSize; i++)
            Gamma[i] = 1.0;
    }

    public Matrix Forward(Matrix x)
    {
        Input = x;
        int N = x.Rows;
        int D = x.Cols;
        Mean = new double[N];
        Variance = new double[N];
        var result = new Matrix(N, D);
        double epsilon = 1e-5;

        for (int i = 0; i < N; i++)
        {
            double mean = 0.0;
            for (int j = 0; j < D; j++)
                mean += x.Data[i, j];
            mean /= D;
            Mean[i] = mean;

            double variance = 0.0;
            for (int j = 0; j < D; j++)
                variance += (x.Data[i, j] - mean) * (x.Data[i, j] - mean);
            variance /= D;
            Variance[i] = variance;

            for (int j = 0; j < D; j++)
            {
                double normalized = (x.Data[i, j] - mean) / Math.Sqrt(variance + epsilon);
                result.Data[i, j] = Gamma[j] * normalized + Beta[j];
            }
        }
        return result;
    }

    public Matrix Backward(Matrix dOutput)
    {
        int N = dOutput.Rows;
        int D = dOutput.Cols;
        var dx = new Matrix(N, D);
        double epsilon = 1e-5;

        for (int i = 0; i < N; i++)
        {
            double mean = Mean[i];
            double variance = Variance[i];
            double stdInv = 1.0 / Math.Sqrt(variance + epsilon);

            double[] dXHat = new double[D];
            double dVar = 0.0;
            double dMean = 0.0;

            for (int j = 0; j < D; j++)
            {
                double xHat = (Input.Data[i, j] - mean) * stdInv;
                GradGamma[j] += dOutput.Data[i, j] * xHat;
                GradBeta[j] += dOutput.Data[i, j];
                dXHat[j] = dOutput.Data[i, j] * Gamma[j];
            }

            for (int j = 0; j < D; j++)
            {
                dVar += dXHat[j] * (Input.Data[i, j] - mean) * -0.5 * Math.Pow(variance + epsilon, -1.5);
            }

            for (int j = 0; j < D; j++)
            {
                dMean += dXHat[j] * -stdInv + dVar * -2.0 * (Input.Data[i, j] - mean) / D;
            }

            for (int j = 0; j < D; j++)
            {
                dx.Data[i, j] = dXHat[j] * stdInv + dVar * 2.0 * (Input.Data[i, j] - mean) / D + dMean / D;
            }
        }
        return dx;
    }
}

public class LinearLayer
{
    public int InputSize, OutputSize;
    public Matrix Weights;
    public Matrix Bias;
    public Matrix GradWeights;
    public Matrix GradBias;
    private Matrix Input;

    public LinearLayer(int inputSize, int outputSize)
    {
        InputSize = inputSize; OutputSize = outputSize;
        Weights = Matrix.Random(inputSize, outputSize);
        Bias = new Matrix(1, outputSize);
        GradWeights = new Matrix(inputSize, outputSize);
        GradBias = new Matrix(1, outputSize);
    }

    public Matrix Forward(Matrix input)
    {
        Input = input;
        return (input * Weights) + Bias;
    }

    public Matrix Backward(Matrix dOutput)
    {
        GradWeights += Input.Transpose() * dOutput;
        GradBias += dOutput.SumRows();
        var dInput = dOutput * Weights.Transpose();
        return dInput;
    }
}

public class Matrix
{
    public int Rows, Cols;
    public double[,] Data;

    public Matrix(int rows, int cols)
    {
        Rows = rows; Cols = cols;
        Data = new double[rows, cols];
    }

    public static Matrix Random(int rows, int cols)
    {
        var m = new Matrix(rows, cols);
        var rand = new Random();
        double std = 1.0 / Math.Sqrt(cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m.Data[i, j] = rand.NextDouble() * 2 * std - std;
        return m;
    }

    public Matrix Transpose()
    {
        var result = new Matrix(Cols, Rows);
        for (int i = 0; i < Rows; i++)
            for (int j = 0; j < Cols; j++)
                result.Data[j, i] = Data[i, j];
        return result;
    }

    public static Matrix operator +(Matrix a, Matrix b)
    {
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Cols; j++)
                result.Data[i, j] = a.Data[i, j] + b.Data[i, j];
        return result;
    }

    public static Matrix operator -(Matrix a, Matrix b)
    {
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Cols; j++)
                result.Data[i, j] = a.Data[i, j] - b.Data[i, j];
        return result;
    }

    public static Matrix operator *(Matrix a, Matrix b)
    {
        var result = new Matrix(a.Rows, b.Cols);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < b.Cols; j++)
                for (int k = 0; k < a.Cols; k++)
                    result.Data[i, j] += a.Data[i, k] * b.Data[k, j];
        return result;
    }

    public static Matrix operator *(double scalar, Matrix a)
    {
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Cols; j++)
                result.Data[i, j] = scalar * a.Data[i, j];
        return result;
    }

    public static Matrix operator /(Matrix a, double scalar)
    {
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        for (int j = 0; j < a.Cols; j++)
            result.Data[i, j] = a.Data[i, j] / scalar;
        return result;
    }

    public Matrix SumRows()
    {
        var result = new Matrix(1, Cols);
        for (int j = 0; j < Cols; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < Rows; i++)
                sum += Data[i, j];
            result.Data[0, j] = sum;
        }
        return result;
    }

    public void Clear()
    {
        for (int i = 0; i < Rows; i++)
            for (int j = 0; j < Cols; j++)
                Data[i, j] = 0.0;
    }

    public static Matrix operator +(Matrix a)
    {
        return a;
    }

    public static Matrix operator -(Matrix a)
    {
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Cols; j++)
                result.Data[i, j] = -a.Data[i, j];
        return result;
    }
}

public class Optimizer
{
    public double LearningRate;

    public Optimizer(double learningRate)
    {
        LearningRate = learningRate;
    }

    public void Step(GPTModel model)
    {
        // Update token embeddings
        model.TokenEmbedding.Weights -= LearningRate * model.TokenEmbedding.GradWeights;
        model.TokenEmbedding.GradWeights.Clear();

        // Update positional embeddings
        model.PositionalEmbedding.Weights -= LearningRate * model.PositionalEmbedding.GradWeights;
        model.PositionalEmbedding.GradWeights.Clear();

        // Update final linear layer
        model.FinalLayer.Weights -= LearningRate * model.FinalLayer.GradWeights;
        model.FinalLayer.Bias -= LearningRate * model.FinalLayer.GradBias;
        model.FinalLayer.GradWeights.Clear();
        model.FinalLayer.GradBias.Clear();

        // Update transformer layers
        foreach (var layer in model.Layers)
        {
            // Update self-attention parameters
            layer.SelfAttention.Wq -= LearningRate * layer.SelfAttention.GradWq;
            layer.SelfAttention.Wk -= LearningRate * layer.SelfAttention.GradWk;
            layer.SelfAttention.Wv -= LearningRate * layer.SelfAttention.GradWv;
            layer.SelfAttention.Wo -= LearningRate * layer.SelfAttention.GradWo;

            layer.SelfAttention.GradWq.Clear();
            layer.SelfAttention.GradWk.Clear();
            layer.SelfAttention.GradWv.Clear();
            layer.SelfAttention.GradWo.Clear();

            // Update feed-forward network parameters
            layer.FFN.Linear1.Weights -= LearningRate * layer.FFN.Linear1.GradWeights;
            layer.FFN.Linear1.Bias -= LearningRate * layer.FFN.Linear1.GradBias;
            layer.FFN.Linear2.Weights -= LearningRate * layer.FFN.Linear2.GradWeights;
            layer.FFN.Linear2.Bias -= LearningRate * layer.FFN.Linear2.GradBias;

            layer.FFN.Linear1.GradWeights.Clear();
            layer.FFN.Linear1.GradBias.Clear();
            layer.FFN.Linear2.GradWeights.Clear();
            layer.FFN.Linear2.GradBias.Clear();

            // Update layer norm parameters
            for (int i = 0; i < layer.LayerNorm1.EmbeddingSize; i++)
            {
                layer.LayerNorm1.Gamma[i] -= LearningRate * layer.LayerNorm1.GradGamma[i];
                layer.LayerNorm1.Beta[i] -= LearningRate * layer.LayerNorm1.GradBeta[i];
                layer.LayerNorm1.GradGamma[i] = 0.0;
                layer.LayerNorm1.GradBeta[i] = 0.0;

                layer.LayerNorm2.Gamma[i] -= LearningRate * layer.LayerNorm2.GradGamma[i];
                layer.LayerNorm2.Beta[i] -= LearningRate * layer.LayerNorm2.GradBeta[i];
                layer.LayerNorm2.GradGamma[i] = 0.0;
                layer.LayerNorm2.GradBeta[i] = 0.0;
            }
        }
    }
}

public class MinGPT3Test
{
    public static void Main()
    {
        int vocabSize = 1000;
        int embeddingSize = 64;
        int numHeads = 4;
        int numLayers = 2;
        int maxSeqLen = 128;
        int batchSize = 4; // Define batch size

        var model = new GPTModel(vocabSize, embeddingSize, numHeads, numLayers, maxSeqLen);
        var optimizer = new Optimizer(learningRate: 0.001);

        for (int epoch = 0; epoch < 10; epoch++)
        {
            int[][] batchInputIds = GetBatchTrainingData(batchSize);
            var logitsBatch = model.Forward(batchInputIds);

            double loss = ComputeLoss(logitsBatch, batchInputIds, out Matrix[] dLogitsBatch);

            model.Backward(dLogitsBatch, batchInputIds);
            optimizer.Step(model);

            Console.WriteLine($"Epoch {epoch}, Loss: {loss:F4}");
        }
    }

    static int[][] GetBatchTrainingData(int batchSize)
    {
        var rand = new Random();
        int seqLen = 10;
        var batchInputIds = new int[batchSize][];
        for (int b = 0; b < batchSize; b++)
        {
            batchInputIds[b] = new int[seqLen];
            for (int i = 0; i < seqLen; i++)
                batchInputIds[b][i] = rand.Next(0, 1000);
        }
        return batchInputIds;
    }

    static double ComputeLoss(Matrix[] logitsBatch, int[][] batchTargetIds, out Matrix[] dLogitsBatch)
    {
        int batchSize = logitsBatch.Length;
        double totalLoss = 0.0;
        dLogitsBatch = new Matrix[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            Matrix logits = logitsBatch[b];
            int[] targetIds = batchTargetIds[b];
            int N = targetIds.Length;
            int V = logits.Cols;
            var dLogits = new Matrix(N, V);
            double loss = 0.0;

            for (int i = 0; i < N; i++)
            {
                double maxLogit = double.NegativeInfinity;
                for (int j = 0; j < V; j++)
                    if (logits.Data[i, j] > maxLogit)
                        maxLogit = logits.Data[i, j];

                double sumExp = 0.0;
                for (int j = 0; j < V; j++)
                    sumExp += Math.Exp(logits.Data[i, j] - maxLogit);
                double logSumExp = maxLogit + Math.Log(sumExp);

                double logProb = logits.Data[i, targetIds[i]] - logSumExp;
                loss -= logProb;

                // Compute gradient
                for (int j = 0; j < V; j++)
                {
                    double softmax = Math.Exp(logits.Data[i, j] - logSumExp) / sumExp;
                    dLogits.Data[i, j] = softmax;
                }
                dLogits.Data[i, targetIds[i]] -= 1.0;
            }
            loss /= N;
            totalLoss += loss;

            // Normalize gradient
            for (int i = 0; i < N; i++)
                for (int j = 0; j < V; j++)
                    dLogits.Data[i, j] /= N;

            dLogitsBatch[b] = dLogits;
        }

        totalLoss /= batchSize;
        return totalLoss;
    }
}
