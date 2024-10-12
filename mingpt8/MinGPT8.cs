using System;

namespace MiniGPT
{
    // Helper class for parameters with gradients
    public class Parameter
    {
        public double[] Value;
        public double[] Grad;

        public Parameter(int size)
        {
            Value = new double[size];
            Grad = new double[size];
        }
    }

    // Multi-head self-attention layer
    public class MultiHeadAttention
    {
        int numHeads;
        int headDim;
        int embedSize;
        double dropoutRate;
        Random rand = new Random();

        // Parameters
        public Parameter Wq, Wk, Wv, Wo;

        // Intermediate variables for backpropagation
        double[][] Q, K, V, Attn, Mask, dQ, dK, dV;
        double[][][] Q_split, K_split, V_split, Attn_split;
        double[][][] dQ_split, dK_split, dV_split;
        double[][] concatenated;

        public MultiHeadAttention(int embedSize, int numHeads, double dropoutRate)
        {
            this.embedSize = embedSize;
            this.numHeads = numHeads;
            this.headDim = embedSize / numHeads;
            this.dropoutRate = dropoutRate;

            Wq = new Parameter(embedSize * embedSize);
            Wk = new Parameter(embedSize * embedSize);
            Wv = new Parameter(embedSize * embedSize);
            Wo = new Parameter(embedSize * embedSize);
            InitializeParameters();
        }

        void InitializeParameters()
        {
            int size = embedSize * embedSize;
            for (int i = 0; i < size; i++)
            {
                Wq.Value[i] = rand.NextDouble() * 0.02 - 0.01;
                Wk.Value[i] = rand.NextDouble() * 0.02 - 0.01;
                Wv.Value[i] = rand.NextDouble() * 0.02 - 0.01;
                Wo.Value[i] = rand.NextDouble() * 0.02 - 0.01;
            }
        }

        public double[][] Forward(double[][] X)
        {
            int seqLen = X.Length;

            // Linear projections
            Q = Linear(X, Wq);
            K = Linear(X, Wk);
            V = Linear(X, Wv);

            // Reshape for multi-head
            Q_split = SplitHeads(Q);
            K_split = SplitHeads(K);
            V_split = SplitHeads(V);

            // Scaled dot-product attention
            Attn_split = ScaledDotProductAttention(Q_split, K_split, V_split);

            // Concatenate heads
            concatenated = CombineHeads(Attn_split);

            // Final linear layer
            double[][] output = Linear(concatenated, Wo);

            // Apply dropout (residual dropout)
            output = Dropout(output, dropoutRate);

            return output;
        }

        public double[][] Backward(double[][] dOut)
        {
            int seqLen = dOut.Length;

            // Backprop through dropout
            dOut = DropoutBackward(dOut, dropoutRate);

            // Gradients w.r.t Wo
            double[][] dConcatenated = LinearBackward(dOut, concatenated, Wo, Wo.Grad);

            // Split gradients to heads
            double[][][] dAttn_split = SplitHeadsBackward(dConcatenated);

            // Backprop through attention
            (dQ_split, dK_split, dV_split) = ScaledDotProductAttentionBackward(dAttn_split, Q_split, K_split, V_split);

            // Merge head gradients
            double[][] dQ = MergeHeads(dQ_split);
            double[][] dK = MergeHeads(dK_split);
            double[][] dV = MergeHeads(dV_split);

            // Gradients w.r.t Wq, Wk, Wv
            double[][] dX_q = LinearBackward(dQ, Q, Wq, Wq.Grad);
            double[][] dX_k = LinearBackward(dK, K, Wk, Wk.Grad);
            double[][] dX_v = LinearBackward(dV, V, Wv, Wv.Grad);

            // Sum gradients from the three paths
            double[][] dX = new double[seqLen][];
            for (int i = 0; i < seqLen; i++)
            {
                dX[i] = new double[embedSize];
                for (int j = 0; j < embedSize; j++)
                {
                    dX[i][j] = dX_q[i][j] + dX_k[i][j] + dX_v[i][j];
                }
            }

            return dX;
        }

        double[][] Linear(double[][] X, Parameter W)
        {
            int seqLen = X.Length;
            int inputDim = X[0].Length;
            int outputDim = embedSize;
            double[][] output = new double[seqLen][];
            for (int i = 0; i < seqLen; i++)
            {
                output[i] = new double[outputDim];
                for (int j = 0; j < outputDim; j++)
                {
                    for (int k = 0; k < inputDim; k++)
                    {
                        output[i][j] += X[i][k] * W.Value[k * outputDim + j];
                    }
                }
            }
            return output;
        }

        double[][] LinearBackward(double[][] dOut, double[][] X, Parameter W, double[] dW)
        {
            int seqLen = X.Length;
            int inputDim = X[0].Length;
            int outputDim = embedSize;
            double[][] dX = new double[seqLen][];
            for (int i = 0; i < seqLen; i++)
            {
                dX[i] = new double[inputDim];
                for (int j = 0; j < outputDim; j++)
                {
                    for (int k = 0; k < inputDim; k++)
                    {
                        dX[i][k] += dOut[i][j] * W.Value[k * outputDim + j];
                        dW[k * outputDim + j] += X[i][k] * dOut[i][j];
                    }
                }
            }
            return dX;
        }

        double[][][] SplitHeads(double[][] X)
        {
            int seqLen = X.Length;
            double[][][] output = new double[numHeads][][];
            for (int h = 0; h < numHeads; h++)
            {
                output[h] = new double[seqLen][];
                for (int i = 0; i < seqLen; i++)
                {
                    output[h][i] = new double[headDim];
                    Array.Copy(X[i], h * headDim, output[h][i], 0, headDim);
                }
            }
            return output;
        }

        double[][][] SplitHeadsBackward(double[][] dConcatenated)
        {
            int seqLen = dConcatenated.Length;
            double[][][] dX = new double[numHeads][][];
            for (int h = 0; h < numHeads; h++)
            {
                dX[h] = new double[seqLen][];
                for (int i = 0; i < seqLen; i++)
                {
                    dX[h][i] = new double[headDim];
                    Array.Copy(dConcatenated[i], h * headDim, dX[h][i], 0, headDim);
                }
            }
            return dX;
        }

        double[][] CombineHeads(double[][][] X)
        {
            int seqLen = X[0].Length;
            double[][] output = new double[seqLen][];
            for (int i = 0; i < seqLen; i++)
            {
                output[i] = new double[embedSize];
                for (int h = 0; h < numHeads; h++)
                {
                    Array.Copy(X[h][i], 0, output[i], h * headDim, headDim);
                }
            }
            return output;
        }

        double[][] MergeHeads(double[][][] dX)
        {
            int seqLen = dX[0].Length;
            double[][] output = new double[seqLen][];
            for (int i = 0; i < seqLen; i++)
            {
                output[i] = new double[embedSize];
                for (int h = 0; h < numHeads; h++)
                {
                    Array.Copy(dX[h][i], 0, output[i], h * headDim, headDim);
                }
            }
            return output;
        }

        double[][][] ScaledDotProductAttention(double[][][] Q, double[][][] K, double[][][] V)
        {
            int seqLen = Q[0].Length;
            double[][][] output = new double[numHeads][][];
            for (int h = 0; h < numHeads; h++)
            {
                output[h] = new double[seqLen][];
                for (int i = 0; i < seqLen; i++)
                {
                    double[] scores = new double[seqLen];
                    for (int j = 0; j <= i; j++)
                    {
                        scores[j] = DotProduct(Q[h][i], K[h][j]) / Math.Sqrt(headDim);
                    }
                    for (int j = i + 1; j < seqLen; j++)
                    {
                        scores[j] = double.NegativeInfinity;
                    }
                    double[] attnWeights = Softmax(scores);
                    output[h][i] = new double[headDim];
                    for (int j = 0; j <= i; j++)
                    {
                        for (int d = 0; d < headDim; d++)
                        {
                            output[h][i][d] += attnWeights[j] * V[h][j][d];
                        }
                    }
                }
            }
            return output;
        }

        (double[][][], double[][][], double[][][]) ScaledDotProductAttentionBackward(double[][][] dOut, double[][][] Q, double[][][] K, double[][][] V)
        {
            int seqLen = Q[0].Length;
            double[][][] dQ = new double[numHeads][][];
            double[][][] dK = new double[numHeads][][];
            double[][][] dV = new double[numHeads][][];
            for (int h = 0; h < numHeads; h++)
            {
                dQ[h] = new double[seqLen][];
                dK[h] = new double[seqLen][];
                dV[h] = new double[seqLen][];
                for (int i = 0; i < seqLen; i++)
                {
                    dQ[h][i] = new double[headDim];
                    for (int j = 0; j < seqLen; j++)
                    {
                        dK[h][j] = new double[headDim];
                        dV[h][j] = new double[headDim];
                    }
                    double[] scores = new double[seqLen];
                    for (int j = 0; j <= i; j++)
                    {
                        scores[j] = DotProduct(Q[h][i], K[h][j]) / Math.Sqrt(headDim);
                    }
                    for (int j = i + 1; j < seqLen; j++)
                    {
                        scores[j] = double.NegativeInfinity;
                    }
                    double[] attnWeights = Softmax(scores);

                    // Compute gradient w.r.t V
                    for (int j = 0; j <= i; j++)
                    {
                        for (int d = 0; d < headDim; d++)
                        {
                            dV[h][j][d] += attnWeights[j] * dOut[h][i][d];
                        }
                    }

                    // Compute gradient w.r.t attention weights
                    double[] dAttnWeights = new double[seqLen];
                    for (int d = 0; d < headDim; d++)
                    {
                        for (int j = 0; j <= i; j++)
                        {
                            dAttnWeights[j] += V[h][j][d] * dOut[h][i][d];
                        }
                    }

                    // Backprop through softmax
                    double[] dScores = SoftmaxBackward(dAttnWeights, attnWeights);

                    // Compute gradient w.r.t Q and K
                    for (int j = 0; j <= i; j++)
                    {
                        double scale = 1.0 / Math.Sqrt(headDim);
                        for (int d = 0; d < headDim; d++)
                        {
                            dQ[h][i][d] += scale * dScores[j] * K[h][j][d];
                            dK[h][j][d] += scale * dScores[j] * Q[h][i][d];
                        }
                    }
                }
            }
            return (dQ, dK, dV);
        }

        double[] Softmax(double[] x)
        {
            double max = double.NegativeInfinity;
            foreach (var val in x)
            {
                if (val > max) max = val;
            }
            double sum = 0;
            double[] exp = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                exp[i] = Math.Exp(x[i] - max);
                sum += exp[i];
            }
            for (int i = 0; i < x.Length; i++)
            {
                exp[i] /= sum;
            }
            return exp;
        }

        double[] SoftmaxBackward(double[] dOut, double[] softmax)
        {
            double[] dX = new double[dOut.Length];
            for (int i = 0; i < dOut.Length; i++)
            {
                for (int j = 0; j < dOut.Length; j++)
                {
                    double delta = (i == j) ? 1 : 0;
                    dX[i] += softmax[j] * (delta - softmax[i]) * dOut[j];
                }
            }
            return dX;
        }

        double DotProduct(double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }

        double[][] Dropout(double[][] X, double rate)
        {
            int seqLen = X.Length;
            int dim = X[0].Length;
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    if (rand.NextDouble() < rate)
                    {
                        X[i][j] = 0;
                    }
                }
            }
            return X;
        }

        double[][] DropoutBackward(double[][] dOut, double rate)
        {
            return dOut;
        }
    }

    // Feed-forward network
    public class FeedForward
    {
        int embedSize;
        int hiddenSize;
        double dropoutRate;
        Random rand = new Random();

        // Parameters
        public Parameter W1, b1, W2, b2;

        // Intermediate variables
        double[][] hidden;
        double[][] X_input;

        public FeedForward(int embedSize, int hiddenSize, double dropoutRate)
        {
            this.embedSize = embedSize;
            this.hiddenSize = hiddenSize;
            this.dropoutRate = dropoutRate;

            W1 = new Parameter(embedSize * hiddenSize);
            b1 = new Parameter(hiddenSize);
            W2 = new Parameter(hiddenSize * embedSize);
            b2 = new Parameter(embedSize);
            InitializeParameters();
        }

        void InitializeParameters()
        {
            int size1 = embedSize * hiddenSize;
            int size2 = hiddenSize * embedSize;
            for (int i = 0; i < size1; i++)
            {
                W1.Value[i] = rand.NextDouble() * 0.02 - 0.01;
            }
            for (int i = 0; i < size2; i++)
            {
                W2.Value[i] = rand.NextDouble() * 0.02 - 0.01;
            }
            for (int i = 0; i < hiddenSize; i++)
            {
                b1.Value[i] = 0;
            }
            for (int i = 0; i < embedSize; i++)
            {
                b2.Value[i] = 0;
            }
        }

        public double[][] Forward(double[][] X)
        {
            int seqLen = X.Length;
            X_input = X;
            hidden = new double[seqLen][];
            double[][] output = new double[seqLen][];
            for (int i = 0; i < seqLen; i++)
            {
                // Hidden layer
                hidden[i] = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++)
                {
                    for (int k = 0; k < embedSize; k++)
                    {
                        hidden[i][j] += X[i][k] * W1.Value[k * hiddenSize + j];
                    }
                    hidden[i][j] += b1.Value[j];
                    hidden[i][j] = Math.Max(0, hidden[i][j]); // ReLU activation
                }
                // Dropout
                for (int j = 0; j < hiddenSize; j++)
                {
                    if (rand.NextDouble() < dropoutRate)
                    {
                        hidden[i][j] = 0;
                    }
                }
                // Output layer
                output[i] = new double[embedSize];
                for (int j = 0; j < embedSize; j++)
                {
                    for (int k = 0; k < hiddenSize; k++)
                    {
                        output[i][j] += hidden[i][k] * W2.Value[k * embedSize + j];
                    }
                    output[i][j] += b2.Value[j];
                }
            }
            return output;
        }

        public double[][] Backward(double[][] dOut)
        {
            int seqLen = dOut.Length;
            double[][] dHidden = new double[seqLen][];
            double[][] dX = new double[seqLen][];
            for (int i = 0; i < seqLen; i++)
            {
                dHidden[i] = new double[hiddenSize];
                dX[i] = new double[embedSize];
                // Gradients w.r.t W2 and b2
                for (int j = 0; j < embedSize; j++)
                {
                    for (int k = 0; k < hiddenSize; k++)
                    {
                        W2.Grad[k * embedSize + j] += hidden[i][k] * dOut[i][j];
                        dHidden[i][k] += W2.Value[k * embedSize + j] * dOut[i][j];
                    }
                    b2.Grad[j] += dOut[i][j];
                }
                // Backprop through ReLU
                for (int j = 0; j < hiddenSize; j++)
                {
                    if (hidden[i][j] <= 0)
                    {
                        dHidden[i][j] = 0;
                    }
                }
                // Gradients w.r.t W1 and b1
                for (int j = 0; j < hiddenSize; j++)
                {
                    for (int k = 0; k < embedSize; k++)
                    {
                        W1.Grad[k * hiddenSize + j] += X_input[i][k] * dHidden[i][j];
                        dX[i][k] += W1.Value[k * hiddenSize + j] * dHidden[i][j];
                    }
                    b1.Grad[j] += dHidden[i][j];
                }
            }
            return dX;
        }
    }

    // Transformer block
    public class TransformerBlock
    {
        int embedSize;
        double dropoutRate;

        // Layers
        public MultiHeadAttention attention;
        public FeedForward feedForward;

        // Intermediate variables
        double[][] X_input;
        double[][] attnOutput;

        public TransformerBlock(int embedSize, int numHeads, int hiddenSize, double dropoutRate)
        {
            this.embedSize = embedSize;
            this.dropoutRate = dropoutRate;

            attention = new MultiHeadAttention(embedSize, numHeads, dropoutRate);
            feedForward = new FeedForward(embedSize, hiddenSize, dropoutRate);
        }

        public double[][] Forward(double[][] X)
        {
            X_input = X;
            // Self-attention
            attnOutput = attention.Forward(X);
            // Residual connection can be added here (skipped for simplicity)
            double[][] attnResidual = attnOutput; // Assuming no residuals
            // Feed-forward
            double[][] ffOutput = feedForward.Forward(attnResidual);
            // Residual connection can be added here (skipped for simplicity)
            return ffOutput;
        }

        public double[][] Backward(double[][] dOut)
        {
            // Backprop through feed-forward
            double[][] dAttnOutput = feedForward.Backward(dOut);
            // Backprop through attention
            double[][] dX = attention.Backward(dAttnOutput);
            // Assuming residual connections, sum gradients
            // For simplicity, assuming no residuals
            return dX;
        }
    }

    // Transformer model
    public class Transformer
    {
        int vocabSize;
        int embedSize;
        int numLayers;
        double dropoutRate;
        int maxSeqLen;
        Random rand = new Random();

        // Embeddings
        public Parameter tokenEmbedding;
        public Parameter positionEmbedding;

        // Layers
        public TransformerBlock[] layers;

        // Output layer
        public Parameter outputWeight;
        public Parameter outputBias;

        // Intermediate variables
        int[] inputIds;
        double[][] X_embedded;
        double[][][] layerOutputs;

        public Transformer(int vocabSize, int embedSize, int numLayers, int numHeads, int hiddenSize, int maxSeqLen, double dropoutRate)
        {
            this.vocabSize = vocabSize;
            this.embedSize = embedSize;
            this.numLayers = numLayers;
            this.maxSeqLen = maxSeqLen;
            this.dropoutRate = dropoutRate;

            tokenEmbedding = new Parameter(vocabSize * embedSize);
            positionEmbedding = new Parameter(maxSeqLen * embedSize);
            InitializeEmbeddings();

            layers = new TransformerBlock[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                layers[i] = new TransformerBlock(embedSize, numHeads, hiddenSize, dropoutRate);
            }

            outputWeight = new Parameter(embedSize * vocabSize);
            outputBias = new Parameter(vocabSize);
            InitializeOutputLayer();
        }

        void InitializeEmbeddings()
        {
            int size = vocabSize * embedSize;
            for (int i = 0; i < size; i++)
            {
                tokenEmbedding.Value[i] = rand.NextDouble() * 0.02 - 0.01;
            }
            size = maxSeqLen * embedSize;
            for (int i = 0; i < size; i++)
            {
                positionEmbedding.Value[i] = rand.NextDouble() * 0.02 - 0.01;
            }
        }

        void InitializeOutputLayer()
        {
            int size = embedSize * vocabSize;
            for (int i = 0; i < size; i++)
            {
                outputWeight.Value[i] = rand.NextDouble() * 0.02 - 0.01;
            }
            for (int i = 0; i < vocabSize; i++)
            {
                outputBias.Value[i] = 0;
            }
        }

        public double[][] Forward(int[] inputIds)
        {
            this.inputIds = inputIds;
            int seqLen = inputIds.Length;
            double[][] X = new double[seqLen][];

            // Token embeddings
            for (int i = 0; i < seqLen; i++)
            {
                X[i] = new double[embedSize];
                int tokenId = inputIds[i];
                Array.Copy(tokenEmbedding.Value, tokenId * embedSize, X[i], 0, embedSize);
            }

            // Position embeddings
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embedSize; j++)
                {
                    X[i][j] += positionEmbedding.Value[i * embedSize + j];
                }
            }

            // Dropout
            X = Dropout(X, dropoutRate);
            X_embedded = X;

            // Transformer layers
            layerOutputs = new double[numLayers + 1][][];
            layerOutputs[0] = X;
            for (int l = 0; l < numLayers; l++)
            {
                X = layers[l].Forward(X);
                layerOutputs[l + 1] = X;
            }

            // Output layer
            double[][] logits = new double[seqLen][];
            for (int i = 0; i < seqLen; i++)
            {
                logits[i] = new double[vocabSize];
                for (int j = 0; j < vocabSize; j++)
                {
                    for (int k = 0; k < embedSize; k++)
                    {
                        logits[i][j] += X[i][k] * outputWeight.Value[k * vocabSize + j];
                    }
                    logits[i][j] += outputBias.Value[j];
                }
            }
            return logits;
        }

        public void Backward(double[][] dLoss)
        {
            int seqLen = dLoss.Length;
            double[][] dX = new double[seqLen][];
            // Gradients w.r.t output layer weights and biases
            double[][] lastLayerOutput = layerOutputs[numLayers];
            for (int i = 0; i < seqLen; i++)
            {
                dX[i] = new double[embedSize];
                for (int j = 0; j < vocabSize; j++)
                {
                    for (int k = 0; k < embedSize; k++)
                    {
                        outputWeight.Grad[k * vocabSize + j] += lastLayerOutput[i][k] * dLoss[i][j];
                        dX[i][k] += outputWeight.Value[k * vocabSize + j] * dLoss[i][j];
                    }
                    outputBias.Grad[j] += dLoss[i][j];
                }
            }
            // Backprop through transformer layers
            for (int l = numLayers - 1; l >= 0; l--)
            {
                dX = layers[l].Backward(dX);
            }
            // Backprop through embeddings
            dX = DropoutBackward(dX, dropoutRate);
            for (int i = 0; i < seqLen; i++)
            {
                int tokenId = inputIds[i];
                for (int j = 0; j < embedSize; j++)
                {
                    tokenEmbedding.Grad[tokenId * embedSize + j] += dX[i][j];
                    positionEmbedding.Grad[i * embedSize + j] += dX[i][j];
                }
            }
        }

        double[][] Dropout(double[][] X, double rate)
        {
            int seqLen = X.Length;
            int dim = X[0].Length;
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    if (rand.NextDouble() < rate)
                    {
                        X[i][j] = 0;
                    }
                }
            }
            return X;
        }

        double[][] DropoutBackward(double[][] dOut, double rate)
        {
            return dOut;
        }

        public Parameter[] GetParameters()
        {
            var parameters = new System.Collections.Generic.List<Parameter>();
            parameters.Add(tokenEmbedding);
            parameters.Add(positionEmbedding);
            parameters.Add(outputWeight);
            parameters.Add(outputBias);
            foreach (var layer in layers)
            {
                parameters.Add(layer.attention.Wq);
                parameters.Add(layer.attention.Wk);
                parameters.Add(layer.attention.Wv);
                parameters.Add(layer.attention.Wo);
                parameters.Add(layer.feedForward.W1);
                parameters.Add(layer.feedForward.b1);
                parameters.Add(layer.feedForward.W2);
                parameters.Add(layer.feedForward.b2);
            }
            return parameters.ToArray();
        }
    }

    // Optimizer
    public class Optimizer
    {
        double learningRate;

        public Optimizer(double learningRate)
        {
            this.learningRate = learningRate;
        }

        public void Step(Parameter[] parameters)
        {
            foreach (var param in parameters)
            {
                for (int i = 0; i < param.Value.Length; i++)
                {
                    param.Value[i] -= learningRate * param.Grad[i];
                    param.Grad[i] = 0; // Reset gradient after update
                }
            }
        }
    }

    // Training loop
    public class Trainer
    {
        Transformer model;
        Optimizer optimizer;
        int vocabSize;

        public Trainer(Transformer model, Optimizer optimizer, int vocabSize)
        {
            this.model = model;
            this.optimizer = optimizer;
            this.vocabSize = vocabSize;
        }

        public void Train(int[][] inputs, int[][] targets, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    double[][] logits = model.Forward(inputs[i]);
                    double loss = ComputeLoss(logits, targets[i]);
                    totalLoss += loss;

                    // Backward pass
                    double[][] dLoss = ComputeLossGradient(logits, targets[i]);
                    model.Backward(dLoss);

                    // Update parameters
                    optimizer.Step(model.GetParameters());
                }
                Console.WriteLine($"Epoch {epoch + 1}, Loss: {totalLoss / inputs.Length}");
            }
        }

        double ComputeLoss(double[][] logits, int[] targets)
        {
            double loss = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                double[] probs = Softmax(logits[i]);
                loss -= Math.Log(probs[targets[i]] + 1e-9);
            }
            return loss / logits.Length;
        }

        double[][] ComputeLossGradient(double[][] logits, int[] targets)
        {
            double[][] dLoss = new double[logits.Length][];
            for (int i = 0; i < logits.Length; i++)
            {
                double[] probs = Softmax(logits[i]);
                probs[targets[i]] -= 1;
                dLoss[i] = probs;
            }
            return dLoss;
        }

        double[] Softmax(double[] x)
        {
            double max = double.NegativeInfinity;
            foreach (var val in x)
            {
                if (val > max) max = val;
            }
            double sum = 0;
            double[] exp = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                exp[i] = Math.Exp(x[i] - max);
                sum += exp[i];
            }
            for (int i = 0; i < x.Length; i++)
            {
                exp[i] /= sum;
            }
            return exp;
        }
    }

    // Next token prediction
    public class Predictor
    {
        Transformer model;

        public Predictor(Transformer model)
        {
            this.model = model;
        }

        public int PredictNextToken(int[] inputIds)
        {
            double[][] logits = model.Forward(inputIds);
            double[] lastLogits = logits[logits.Length - 1];
            int nextToken = ArgMax(lastLogits);
            return nextToken;
        }

        int ArgMax(double[] x)
        {
            int maxIndex = 0;
            double maxValue = x[0];
            for (int i = 1; i < x.Length; i++)
            {
                if (x[i] > maxValue)
                {
                    maxValue = x[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
    }

    // Main program
    class Program
    {
        static void Main(string[] args)
        {
            int vocabSize = 10000;
            int embedSize = 128;
            int numLayers = 2;
            int numHeads = 4;
            int hiddenSize = 256;
            int maxSeqLen = 50;
            double dropoutRate = 0.1;
            double learningRate = 0.001;
            int epochs = 10;

            Transformer model = new Transformer(vocabSize, embedSize, numLayers, numHeads, hiddenSize, maxSeqLen, dropoutRate);
            Optimizer optimizer = new Optimizer(learningRate);
            Trainer trainer = new Trainer(model, optimizer, vocabSize);

            // Example training data
            int[][] inputs = new int[][] {
                new int[] {1, 2, 3, 4},
                new int[] {2, 3, 4, 5},
            };
            int[][] targets = new int[][] {
                new int[] {2, 3, 4, 5},
                new int[] {3, 4, 5, 6},
            };

            trainer.Train(inputs, targets, epochs);

            Predictor predictor = new Predictor(model);
            int[] inputSequence = new int[] {1, 2, 3};
            int nextToken = predictor.PredictNextToken(inputSequence);
            Console.WriteLine($"Next token: {nextToken}");
        }
    }
}
