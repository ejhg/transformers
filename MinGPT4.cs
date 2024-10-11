using System;

namespace mingpt4;

class MinGPT4Test
{
    static void run () {
        // Hyperparameters
        int vocabSize = 1000;
        int seqLength = 20;
        int embedDim = 64;
        int numHeads = 4;
        int numLayers = 2;
        int numEpochs = 10;
        double learningRate = 0.001;

        // Initialize model
        GPTModel model = new GPTModel (vocabSize, seqLength, embedDim, numHeads, numLayers, learningRate);

        // Dummy data for training (Replace with actual data)
        int[][] inputs = new int[100][];
        int[][] targets = new int[100][];
        Random rand = new Random ();
        for (int i = 0; i < 100; i++) {
            inputs[i] = new int[seqLength];
            targets[i] = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                inputs[i][j] = rand.Next (vocabSize);
                targets[i][j] = rand.Next (vocabSize);
            }
        }

        // Training loop
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            double totalLoss = 0.0;
            for (int i = 0; i < inputs.Length; i++) {
                double[][] logits = model.Forward (inputs[i]);
                double loss = model.Loss (logits, targets[i]);
                model.Backward ();
                model.UpdateParameters ();
                totalLoss += loss;
            }

            Console.WriteLine ($"Epoch {epoch + 1}/{numEpochs}, Loss: {totalLoss / inputs.Length}");
        }
    }
}

public class GPTModel
{
    int vocabSize;
    int seqLength;
    int embedDim;
    int numHeads;
    int numLayers;
    double learningRate;

    // Model parameters
    double[][] tokenEmbedding;
    double[][] positionEmbedding;
    TransformerBlock[] layers;
    double[][] outputWeights;

    // Optimizer parameters
    double[][] tokenEmbeddingGrad;
    double[][] positionEmbeddingGrad;
    double[][] outputWeightsGrad;

    public GPTModel (int vocabSize, int seqLength, int embedDim, int numHeads, int numLayers, double learningRate) {
        this.vocabSize = vocabSize;
        this.seqLength = seqLength;
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.learningRate = learningRate;

        // Initialize embeddings
        tokenEmbedding = RandomMatrix (vocabSize, embedDim);
        positionEmbedding = RandomMatrix (seqLength, embedDim);

        // Initialize transformer layers
        layers = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new TransformerBlock (embedDim, numHeads);
        }

        // Initialize output weights
        outputWeights = RandomMatrix (embedDim, vocabSize);
    }

    double[][] inputEmbedding;
    double[][] logits;
    int[] targets;

    public double[][] Forward (int[] inputs) {
        // Embedding lookup
        inputEmbedding = new double[seqLength][];
        for (int i = 0; i < seqLength; i++) {
            inputEmbedding[i] = AddVectors (tokenEmbedding[inputs[i]], positionEmbedding[i]);
        }

        // Pass through transformer layers
        double[][] x = inputEmbedding;
        for (int i = 0; i < numLayers; i++) {
            x = layers[i].Forward (x);
        }

        // Output logits
        logits = new double[seqLength][];
        for (int i = 0; i < seqLength; i++) {
            logits[i] = MatVecMul (outputWeights, x[i]);
        }

        return logits;
    }

    public double Loss (double[][] logits, int[] targets) {
        this.targets = targets;
        double loss = 0.0;
        for (int i = 0; i < seqLength; i++) {
            double[] probs = Softmax (logits[i]);
            loss -= Math.Log (probs[targets[i]] + 1e-9);
        }

        return loss / seqLength;
    }

    public void Backward () {
        // Initialize gradients
        outputWeightsGrad = ZeroMatrix (embedDim, vocabSize);
        double[][] gradOutput = new double[seqLength][];

        // Compute gradients w.r.t output weights and activations
        for (int i = 0; i < seqLength; i++) {
            double[] probs = Softmax (logits[i]);
            probs[targets[i]] -= 1.0;
            gradOutput[i] = probs;

            // Update output weights gradient
            for (int j = 0; j < embedDim; j++) {
                for (int k = 0; k < vocabSize; k++) {
                    outputWeightsGrad[j][k] += inputEmbedding[i][j] * gradOutput[i][k];
                }
            }
        }

        // Backpropagate through transformer layers
        double[][] gradInput = gradOutput;
        for (int i = numLayers - 1; i >= 0; i--) {
            gradInput = layers[i].Backward (gradInput);
        }

        // Gradients w.r.t embeddings
        tokenEmbeddingGrad = ZeroMatrix (vocabSize, embedDim);
        positionEmbeddingGrad = ZeroMatrix (seqLength, embedDim);
        for (int i = 0; i < seqLength; i++) {
            int tokenIdx = targets[i];
            for (int j = 0; j < embedDim; j++) {
                tokenEmbeddingGrad[tokenIdx][j] += gradInput[i][j];
                positionEmbeddingGrad[i][j] += gradInput[i][j];
            }
        }
    }

    public void UpdateParameters () {
        // Update token embeddings
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embedDim; j++) {
                tokenEmbedding[i][j] -= learningRate * tokenEmbeddingGrad[i][j];
            }
        }

        // Update position embeddings
        for (int i = 0; i < seqLength; i++) {
            for (int j = 0; j < embedDim; j++) {
                positionEmbedding[i][j] -= learningRate * positionEmbeddingGrad[i][j];
            }
        }

        // Update output weights
        for (int i = 0; i < embedDim; i++) {
            for (int j = 0; j < vocabSize; j++) {
                outputWeights[i][j] -= learningRate * outputWeightsGrad[i][j];
            }
        }

        // Update transformer layers
        for (int i = 0; i < numLayers; i++) {
            layers[i].UpdateParameters (learningRate);
        }
    }

    // Utility functions
    double[] AddVectors (double[] a, double[] b) {
        double[] result = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] + b[i];
        return result;
    }

    double[] MatVecMul (double[][] matrix, double[] vector) {
        int rows = matrix.Length;
        int cols = matrix[0].Length;
        double[] result = new double[cols];
        for (int i = 0; i < cols; i++) {
            result[i] = 0.0;
            for (int j = 0; j < rows; j++) {
                result[i] += vector[j] * matrix[j][i];
            }
        }

        return result;
    }

    double[] Softmax (double[] x) {
        double max = double.NegativeInfinity;
        for (int i = 0; i < x.Length; i++)
            if (x[i] > max)
                max = x[i];
        double sum = 0.0;
        double[] exp = new double[x.Length];
        for (int i = 0; i < x.Length; i++) {
            exp[i] = Math.Exp (x[i] - max);
            sum += exp[i];
        }

        for (int i = 0; i < x.Length; i++)
            exp[i] /= sum;
        return exp;
    }

    double[][] RandomMatrix (int rows, int cols) {
        Random rand = new Random ();
        double[][] matrix = new double[rows][];
        for (int i = 0; i < rows; i++) {
            matrix[i] = new double[cols];
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = rand.NextDouble () * 0.02 - 0.01;
            }
        }

        return matrix;
    }

    double[][] ZeroMatrix (int rows, int cols) {
        double[][] matrix = new double[rows][];
        for (int i = 0; i < rows; i++)
            matrix[i] = new double[cols];
        return matrix;
    }
}

public class TransformerBlock
{
    int embedDim;
    int numHeads;

    // Parameters for self-attention
    double[][] Wq;
    double[][] Wk;
    double[][] Wv;
    double[][] Wo;

    // Parameters for feed-forward network
    double[][] W1;
    double[][] W2;

    // Layer normalization parameters
    double[] ln1Gamma;
    double[] ln1Beta;
    double[] ln2Gamma;
    double[] ln2Beta;

    // Gradients
    double[][] WqGrad;
    double[][] WkGrad;
    double[][] WvGrad;
    double[][] WoGrad;
    double[][] W1Grad;
    double[][] W2Grad;
    double[] ln1GammaGrad;
    double[] ln1BetaGrad;
    double[] ln2GammaGrad;
    double[] ln2BetaGrad;

    // Cache for backward pass
    double[][] input;
    double[][] attnOutput;
    double[][] ffOutput;
    double[][] norm1Output;
    double[][] norm2Output;

    public TransformerBlock (int embedDim, int numHeads) {
        this.embedDim = embedDim;
        this.numHeads = numHeads;

        // Initialize weights
        Wq = RandomMatrix (embedDim, embedDim);
        Wk = RandomMatrix (embedDim, embedDim);
        Wv = RandomMatrix (embedDim, embedDim);
        Wo = RandomMatrix (embedDim, embedDim);
        W1 = RandomMatrix (embedDim, embedDim * 4);
        W2 = RandomMatrix (embedDim * 4, embedDim);

        // Initialize layer normalization parameters
        ln1Gamma = new double[embedDim];
        ln1Beta = new double[embedDim];
        ln2Gamma = new double[embedDim];
        ln2Beta = new double[embedDim];
        for (int i = 0; i < embedDim; i++) {
            ln1Gamma[i] = 1.0;
            ln2Gamma[i] = 1.0;
        }
    }

    public double[][] Forward (double[][] x) {
        input = x;

        // Layer normalization 1
        norm1Output = LayerNorm (x, ln1Gamma, ln1Beta);

        // Multi-head self-attention
        attnOutput = SelfAttention (norm1Output);

        // Residual connection
        double[][] attnResidual = AddMatrices (x, attnOutput);

        // Layer normalization 2
        norm2Output = LayerNorm (attnResidual, ln2Gamma, ln2Beta);

        // Feed-forward network
        ffOutput = FeedForward (norm2Output);

        // Output with residual connection
        double[][] output = AddMatrices (attnResidual, ffOutput);

        return output;
    }

    public double[][] Backward (double[][] gradOutput) {
        // Backprop through residual connection
        double[][] gradFF = gradOutput;
        double[][] gradAttnResidual = gradOutput;

        // Backprop through feed-forward network
        double[][] gradNorm2 = FeedForwardBackward (gradFF);

        // Backprop through layer normalization 2
        double[][] gradAttn = LayerNormBackward (gradNorm2, norm2Output, ln2Gamma, ln2GammaGrad, ln2BetaGrad);

        // Add gradients from residual connection
        for (int i = 0; i < gradAttn.Length; i++)
        for (int j = 0; j < gradAttn[0].Length; j++)
            gradAttnResidual[i][j] += gradAttn[i][j];

        // Backprop through self-attention
        double[][] gradNorm1 = SelfAttentionBackward (gradAttnResidual);

        // Backprop through layer normalization 1
        double[][] gradInput = LayerNormBackward (gradNorm1, norm1Output, ln1Gamma, ln1GammaGrad, ln1BetaGrad);

        // Add gradients from residual connection
        for (int i = 0; i < gradInput.Length; i++)
        for (int j = 0; j < gradInput[0].Length; j++)
            gradInput[i][j] += gradAttnResidual[i][j];

        return gradInput;
    }

    public void UpdateParameters (double lr) {
        // Update weights
        UpdateMatrix (Wq, WqGrad, lr);
        UpdateMatrix (Wk, WkGrad, lr);
        UpdateMatrix (Wv, WvGrad, lr);
        UpdateMatrix (Wo, WoGrad, lr);
        UpdateMatrix (W1, W1Grad, lr);
        UpdateMatrix (W2, W2Grad, lr);

        // Update layer norm parameters
        for (int i = 0; i < embedDim; i++) {
            ln1Gamma[i] -= lr * ln1GammaGrad[i];
            ln1Beta[i] -= lr * ln1BetaGrad[i];
            ln2Gamma[i] -= lr * ln2GammaGrad[i];
            ln2Beta[i] -= lr * ln2BetaGrad[i];
        }
    }

    // Multi-head self-attention implementation
    double[][] SelfAttention (double[][] x) {
        int seqLen = x.Length;
        int headDim = embedDim / numHeads;

        // Project inputs to Q, K, V
        double[][] Q = MatMul (x, Wq);
        double[][] K = MatMul (x, Wk);
        double[][] V = MatMul (x, Wv);

        // Split into heads
        double[][][] Q_heads = SplitHeads (Q, numHeads);
        double[][][] K_heads = SplitHeads (K, numHeads);
        double[][][] V_heads = SplitHeads (V, numHeads);

        // Scaled dot-product attention for each head
        double[][][] attnHeads = new double[numHeads][][];
        for (int h = 0; h < numHeads; h++) {
            attnHeads[h] = ScaledDotProductAttention (Q_heads[h], K_heads[h], V_heads[h]);
        }

        // Concatenate heads
        double[][] attnOutput = CombineHeads (attnHeads);

        // Final linear projection
        attnOutput = MatMul (attnOutput, Wo);

        return attnOutput;
    }

    double[][] SelfAttentionBackward (double[][] gradOutput) {
        // Not implemented for brevity
        // Should compute gradients w.r.t Wq, Wk, Wv, Wo and inputs
        return gradOutput;
    }

    // Feed-forward network implementation
    double[][] FeedForward (double[][] x) {
        double[][] hidden = MatMul (x, W1);
        hidden = ReLU (hidden);
        double[][] output = MatMul (hidden, W2);
        return output;
    }

    double[][] FeedForwardBackward (double[][] gradOutput) {
        // Not implemented for brevity
        // Should compute gradients w.r.t W1, W2 and inputs
        return gradOutput;
    }

    // Layer normalization implementation
    double[][] LayerNorm (double[][] x, double[] gamma, double[] beta) {
        int seqLen = x.Length;
        int dim = x[0].Length;
        double[][] output = new double[seqLen][];
        for (int i = 0; i < seqLen; i++) {
            double mean = 0.0;
            double variance = 0.0;
            output[i] = new double[dim];
            for (int j = 0; j < dim; j++) {
                mean += x[i][j];
            }

            mean /= dim;
            for (int j = 0; j < dim; j++) {
                variance += Math.Pow (x[i][j] - mean, 2);
            }

            variance /= dim;
            double std = Math.Sqrt (variance + 1e-5);
            for (int j = 0; j < dim; j++) {
                output[i][j] = gamma[j] * ((x[i][j] - mean) / std) + beta[j];
            }
        }

        return output;
    }

    double[][] LayerNormBackward (double[][] gradOutput, double[][] normOutput, double[] gamma, double[] gammaGrad, double[] betaGrad) {
        // Not implemented for brevity
        // Should compute gradients w.r.t gamma, beta, and inputs
        return gradOutput;
    }

    // Utility functions
    double[][] MatMul (double[][] a, double[][] b) {
        int rows = a.Length;
        int cols = b[0].Length;
        int innerDim = a[0].Length;
        double[][] result = new double[rows][];
        for (int i = 0; i < rows; i++) {
            result[i] = new double[cols];
            for (int j = 0; j < cols; j++) {
                result[i][j] = 0.0;
                for (int k = 0; k < innerDim; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        return result;
    }

    double[][] AddMatrices (double[][] a, double[][] b) {
        int rows = a.Length;
        int cols = a[0].Length;
        double[][] result = new double[rows][];
        for (int i = 0; i < rows; i++) {
            result[i] = new double[cols];
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }

        return result;
    }

    double[][] ReLU (double[][] x) {
        int rows = x.Length;
        int cols = x[0].Length;
        double[][] result = new double[rows][];
        for (int i = 0; i < rows; i++) {
            result[i] = new double[cols];
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.Max (0.0, x[i][j]);
            }
        }

        return result;
    }

    double[][][] SplitHeads (double[][] x, int numHeads) {
        int seqLen = x.Length;
        int dim = x[0].Length;
        int headDim = dim / numHeads;
        double[][][] result = new double[numHeads][][];
        for (int h = 0; h < numHeads; h++) {
            result[h] = new double[seqLen][];
            for (int i = 0; i < seqLen; i++) {
                result[h][i] = new double[headDim];
                Array.Copy (x[i], h * headDim, result[h][i], 0, headDim);
            }
        }

        return result;
    }

    double[][] CombineHeads (double[][][] heads) {
        int numHeads = heads.Length;
        int seqLen = heads[0].Length;
        int headDim = heads[0][0].Length;
        int dim = numHeads * headDim;
        double[][] result = new double[seqLen][];
        for (int i = 0; i < seqLen; i++) {
            result[i] = new double[dim];
            for (int h = 0; h < numHeads; h++) {
                Array.Copy (heads[h][i], 0, result[i], h * headDim, headDim);
            }
        }

        return result;
    }

    double[][] ScaledDotProductAttention (double[][] Q, double[][] K, double[][] V) {
        int seqLen = Q.Length;
        int dim = Q[0].Length;
        double[][] scores = new double[seqLen][];
        for (int i = 0; i < seqLen; i++) {
            scores[i] = new double[seqLen];
            for (int j = 0; j < seqLen; j++) {
                double dot = 0.0;
                for (int k = 0; k < dim; k++) {
                    dot += Q[i][k] * K[j][k];
                }

                scores[i][j] = dot / Math.Sqrt (dim);
            }
        }

        // Apply softmax to scores
        double[][] attnWeights = new double[seqLen][];
        for (int i = 0; i < seqLen; i++) {
            attnWeights[i] = Softmax (scores[i]);
        }

        // Compute attention output
        double[][] output = new double[seqLen][];
        for (int i = 0; i < seqLen; i++) {
            output[i] = new double[dim];
            for (int j = 0; j < seqLen; j++) {
                for (int k = 0; k < dim; k++) {
                    output[i][k] += attnWeights[i][j] * V[j][k];
                }
            }
        }

        return output;
    }

    double[] Softmax (double[] x) {
        double max = double.NegativeInfinity;
        for (int i = 0; i < x.Length; i++)
            if (x[i] > max)
                max = x[i];
        double sum = 0.0;
        double[] exp = new double[x.Length];
        for (int i = 0; i < x.Length; i++) {
            exp[i] = Math.Exp (x[i] - max);
            sum += exp[i];
        }

        for (int i = 0; i < x.Length; i++)
            exp[i] /= sum;
        return exp;
    }

    void UpdateMatrix (double[][] W, double[][] gradW, double lr) {
        int rows = W.Length;
        int cols = W[0].Length;
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            W[i][j] -= lr * gradW[i][j];
    }

    double[][] RandomMatrix (int rows, int cols) {
        Random rand = new Random ();
        double[][] matrix = new double[rows][];
        for (int i = 0; i < rows; i++) {
            matrix[i] = new double[cols];
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = rand.NextDouble () * 0.02 - 0.01;
            }
        }

        return matrix;
    }
}
