using System;
using System.Collections.Generic;
using System.Linq;

namespace MiniGPT2;

// Tensor class to handle multi-dimensional arrays and operations
public class Tensor
{
    public float[] Data;
    public int[] Shape;
    public int Size;
    public Tensor Grad;
    public List<Action> BackwardOps;

    public Tensor (float[] data, int[] shape) {
        Data = data;
        Shape = shape;
        Size = data.Length;
        BackwardOps = new List<Action> ();
    }

    public Tensor (int[] shape) {
        Shape = shape;
        Size = shape.Aggregate (1, (a, b) => a * b);
        Data = new float[Size];
        BackwardOps = new List<Action> ();
    }

    public void ZeroGrad () {
        if (Grad != null) {
            Array.Clear (Grad.Data, 0, Grad.Size);
        } else {
            Grad = new Tensor (new float[Size], Shape);
        }
    }

    public Tensor Clone () {
        return new Tensor ((float[])Data.Clone (), (int[])Shape.Clone ());
    }

    // Initialize weights
    public void InitWeights (Random rand) {
        for (int i = 0; i < Size; i++)
            Data[i] = (float)(rand.NextDouble () * 0.02 - 0.01);
    }

    public Tensor Transpose (int[] axes = null) {
        if (axes == null) {
            axes = Enumerable.Range (0, Shape.Length).Reverse ().ToArray ();
        }

        if (axes.Length != Shape.Length)
            throw new Exception ("Axes length must match tensor shape length.");

        int[] newShape = new int[Shape.Length];
        for (int i = 0; i < Shape.Length; i++) {
            newShape[i] = Shape[axes[i]];
        }

        Tensor result = new Tensor (newShape);
        int[] oldStrides = GetStrides (Shape);
        int[] newStrides = GetStrides (newShape);

        for (int i = 0; i < Size; i++) {
            int oldIndex = i;
            int[] oldCoords = new int[Shape.Length];
            for (int j = 0; j < Shape.Length; j++) {
                oldCoords[j] = oldIndex / oldStrides[j];
                oldIndex %= oldStrides[j];
            }

            int[] newCoords = new int[Shape.Length];
            for (int j = 0; j < Shape.Length; j++) {
                newCoords[j] = oldCoords[axes[j]];
            }

            int newIndex = 0;
            for (int j = 0; j < Shape.Length; j++) {
                newIndex += newCoords[j] * newStrides[j];
            }

            result.Data[newIndex] = Data[i];
        }

        void Backward () {
            if (Grad == null) ZeroGrad ();
            if (result.Grad == null) result.ZeroGrad ();

            for (int i = 0; i < Size; i++) {
                int oldIndex = i;
                int[] oldCoords = new int[Shape.Length];
                for (int j = 0; j < Shape.Length; j++) {
                    oldCoords[j] = oldIndex / oldStrides[j];
                    oldIndex %= oldStrides[j];
                }

                int[] newCoords = new int[Shape.Length];
                for (int j = 0; j < Shape.Length; j++) {
                    newCoords[j] = oldCoords[axes[j]];
                }

                int newIndex = 0;
                for (int j = 0; j < Shape.Length; j++) {
                    newIndex += newCoords[j] * newStrides[j];
                }

                Grad.Data[i] += result.Grad.Data[newIndex];
            }
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    private int[] GetStrides (int[] shape) {
        int[] strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    // Variance along a specified axis
    public Tensor Var (int axis) {
        Tensor mean = Mean (axis);
        Tensor diff = this - mean;
        Tensor sqDiff = diff * diff;
        Tensor variance = sqDiff.Mean (axis);
        return variance;
    }

    // GELU activation function
    public Tensor GELU () {
        return Apply (
            x => 0.5f * x * (1 + (float)Math.Tanh (Math.Sqrt (2 / Math.PI) * (x + 0.044715f * x * x * x))),
            x => {
                float tanhTerm = (float)Math.Tanh (Math.Sqrt (2 / Math.PI) * (x + 0.044715f * x * x * x));
                float derivative = 0.5f * (1 + tanhTerm) + 0.5f * x * (1 - tanhTerm * tanhTerm) *
                    (float)Math.Sqrt (2 / Math.PI) * (1 + 0.134145f * x * x);
                return derivative;
            });
    }

    // Element-wise addition
    public static Tensor operator + (Tensor a, Tensor b) {
        if (!a.Shape.SequenceEqual (b.Shape))
            throw new Exception ("Shapes do not match for addition.");

        Tensor result = new Tensor (a.Shape);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] + b.Data[i];

        void Backward () {
            if (a.Grad == null) a.ZeroGrad ();
            if (b.Grad == null) b.ZeroGrad ();

            for (int i = 0; i < a.Size; i++) {
                a.Grad.Data[i] += result.Grad.Data[i];
                b.Grad.Data[i] += result.Grad.Data[i];
            }
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    public static Tensor operator + (Tensor a, float b) {
        Tensor result = new Tensor (a.Shape);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] + b;

        void Backward () {
            if (a.Grad == null) a.ZeroGrad ();

            for (int i = 0; i < a.Size; i++) {
                a.Grad.Data[i] += result.Grad.Data[i];
            }
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    // Element-wise subtraction
    public static Tensor operator - (Tensor a, Tensor b) {
        if (!a.Shape.SequenceEqual (b.Shape))
            throw new Exception ("Shapes do not match for subtraction.");

        Tensor result = new Tensor (a.Shape);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] - b.Data[i];

        void Backward () {
            if (a.Grad == null) a.ZeroGrad ();
            if (b.Grad == null) b.ZeroGrad ();

            for (int i = 0; i < a.Size; i++) {
                a.Grad.Data[i] += result.Grad.Data[i];
                b.Grad.Data[i] -= result.Grad.Data[i];
            }
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    // Scalar multiplication
    public static Tensor operator * (Tensor a, float scalar) {
        Tensor result = new Tensor (a.Shape);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] * scalar;

        void Backward () {
            if (a.Grad == null) a.ZeroGrad ();
            for (int i = 0; i < a.Size; i++)
                a.Grad.Data[i] += result.Grad.Data[i] * scalar;
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    // Element-wise multiplication
    public static Tensor operator * (Tensor a, Tensor b) {
        if (!a.Shape.SequenceEqual (b.Shape))
            throw new Exception ("Shapes do not match for multiplication.");

        Tensor result = new Tensor (a.Shape);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] * b.Data[i];

        void Backward () {
            if (a.Grad == null) a.ZeroGrad ();
            if (b.Grad == null) b.ZeroGrad ();

            for (int i = 0; i < a.Size; i++) {
                a.Grad.Data[i] += b.Data[i] * result.Grad.Data[i];
                b.Grad.Data[i] += a.Data[i] * result.Grad.Data[i];
            }
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    // Matrix multiplication
    public Tensor MatMul (Tensor other) {
        if (Shape.Length != 2 || other.Shape.Length != 2 || Shape[1] != other.Shape[0])
            throw new Exception ("Shapes are not compatible for matrix multiplication.");

        int m = Shape[0], n = Shape[1], p = other.Shape[1];
        Tensor result = new Tensor (new int[] {
            m,
            p
        });

        for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += Data[i * n + k] * other.Data[k * p + j];
            }

            result.Data[i * p + j] = sum;
        }

        void Backward () {
            if (Grad == null) ZeroGrad ();
            if (other.Grad == null) other.ZeroGrad ();

            for (int i = 0; i < m; i++)
            for (int j = 0; j < p; j++) {
                float grad = result.Grad.Data[i * p + j];
                for (int k = 0; k < n; k++) {
                    Grad.Data[i * n + k] += grad * other.Data[k * p + j];
                    other.Grad.Data[k * p + j] += grad * Data[i * n + k];
                }
            }
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    // Reshape tensor
    public Tensor Reshape (int[] newShape) {
        if (Size != newShape.Aggregate (1, (a, b) => a * b))
            throw new Exception ("Cannot reshape tensor to new shape.");

        Tensor result = new Tensor (Data, newShape);
        result.BackwardOps = BackwardOps;
        return result;
    }

    // Apply function element-wise
    public Tensor Apply (Func<float, float> func, Func<float, float> gradFunc) {
        Tensor result = new Tensor (Shape);
        for (int i = 0; i < Size; i++)
            result.Data[i] = func (Data[i]);

        void Backward () {
            if (Grad == null) ZeroGrad ();
            for (int i = 0; i < Size; i++)
                Grad.Data[i] += result.Grad.Data[i] * gradFunc (Data[i]);
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    // Sum over specified axis
    public Tensor Sum (int axis) {
        int[] newShape = Shape.ToArray ();
        newShape[axis] = 1;
        int newSize = newShape.Aggregate (1, (a, b) => a * b);
        Tensor result = new Tensor (newShape);

        int stride = Shape.Skip (axis + 1).Aggregate (1, (a, b) => a * b);
        int jump = Shape[axis] * stride;

        for (int i = 0; i < Size; i += jump)
        for (int j = 0; j < stride; j++) {
            float sum = 0;
            for (int k = 0; k < Shape[axis]; k++)
                sum += Data[i + k * stride + j];
            result.Data[(i / jump) * stride + j] = sum;
        }

        void Backward () {
            if (Grad == null) ZeroGrad ();
            for (int i = 0; i < Size; i += jump)
            for (int j = 0; j < stride; j++) {
                float grad = result.Grad.Data[(i / jump) * stride + j];
                for (int k = 0; k < Shape[axis]; k++)
                    Grad.Data[i + k * stride + j] += grad;
            }
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    // Mean over specified axis
    public Tensor Mean (int axis) {
        Tensor sum = Sum (axis);
        float factor = Shape[axis];
        return sum * (1.0f / factor);
    }

    // Element-wise division
    public static Tensor operator / (Tensor a, Tensor b) {
        if (!a.Shape.SequenceEqual (b.Shape))
            throw new Exception ("Shapes do not match for division.");

        Tensor result = new Tensor (a.Shape);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] / b.Data[i];

        void Backward () {
            if (a.Grad == null) a.ZeroGrad ();
            if (b.Grad == null) b.ZeroGrad ();

            for (int i = 0; i < a.Size; i++) {
                a.Grad.Data[i] += result.Grad.Data[i] / b.Data[i];
                b.Grad.Data[i] -= result.Grad.Data[i] * a.Data[i] / (b.Data[i] * b.Data[i]);
            }
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    // Square root
    public Tensor Sqrt () {
        return Apply (x => (float)Math.Sqrt (x), x => 0.5f / (float)Math.Sqrt (x));
    }

    // Power
    public Tensor Pow (float exponent) {
        return Apply (x => (float)Math.Pow (x, exponent), x => exponent * (float)Math.Pow (x, exponent - 1));
    }

    // Exponential
    public Tensor Exp () {
        return Apply (x => (float)Math.Exp (x), x => (float)Math.Exp (x));
    }

    // Logarithm
    public Tensor Log () {
        return Apply (x => (float)Math.Log (x), x => 1 / x);
    }

    // Softmax along last axis
    public Tensor Softmax () {
        int lastAxis = Shape.Length - 1;
        int outerSize = Size / Shape[lastAxis];
        int innerSize = Shape[lastAxis];

        Tensor result = new Tensor (Shape);

        for (int i = 0; i < outerSize; i++) {
            float max = Data.Skip (i * innerSize).Take (innerSize).Max ();
            float sum = 0;
            for (int j = 0; j < innerSize; j++) {
                result.Data[i * innerSize + j] = (float)Math.Exp (Data[i * innerSize + j] - max);
                sum += result.Data[i * innerSize + j];
            }

            for (int j = 0; j < innerSize; j++) {
                result.Data[i * innerSize + j] /= sum;
            }
        }

        void Backward () {
            if (Grad == null) ZeroGrad ();
            for (int i = 0; i < outerSize; i++) {
                for (int j = 0; j < innerSize; j++) {
                    float grad = 0;
                    for (int k = 0; k < innerSize; k++) {
                        float delta = (j == k) ? 1 : 0;
                        grad += result.Data[i * innerSize + k] * (delta - result.Data[i * innerSize + j]) * result.Grad.Data[i * innerSize + k];
                    }

                    Grad.Data[i * innerSize + j] += grad;
                }
            }
        }

        result.BackwardOps.Add (Backward);
        return result;
    }

    // Run backward pass
    public void Backward () {
        Grad = new Tensor (Shape);
        for (int i = 0; i < Size; i++)
            Grad.Data[i] = 1;

        for (int i = BackwardOps.Count - 1; i >= 0; i--)
            BackwardOps[i] ();
    }

    // Utility methods
    public void Fill (float value) {
        for (int i = 0; i < Size; i++)
            Data[i] = value;
    }

    public void ApplyMask (bool[,] mask) {
        int batchSize = Shape[0];
        int seqLen = Shape[1];
        int seqLen2 = Shape[2];
        for (int b = 0; b < batchSize; b++)
        for (int i = 0; i < seqLen; i++)
        for (int j = 0; j < seqLen2; j++) {
            if (!mask[i, j])
                Data[b * seqLen * seqLen2 + i * seqLen2 + j] = float.NegativeInfinity;
        }
    }
}

/// <summary>
/// Uses Tensors, with batching
/// </summary>
class MinGPT2
{
    // Hyperparameters
    const int vocabSize = 1000;
    const int embedSize = 64;
    const int numHeads = 4;
    const int numLayers = 2;
    const int blockSize = 128;

    // Model parameters
    public Tensor tokenEmbedding; // [vocabSize, embedSize]
    public Tensor positionEmbedding; // [blockSize, embedSize]
    public LayerNorm[] layerNorm1;
    public LayerNorm[] layerNorm2;
    public MultiHeadAttention[] attnLayers;
    public MLP[] mlpLayers;

    public MinGPT2 () {
        Random rand = new Random ();

        tokenEmbedding = new Tensor (new int[] {
            vocabSize,
            embedSize
        });
        positionEmbedding = new Tensor (new int[] {
            blockSize,
            embedSize
        });

        for (int i = 0; i < tokenEmbedding.Size; i++)
            tokenEmbedding.Data[i] = (float)(rand.NextDouble () * 0.02 - 0.01);

        for (int i = 0; i < positionEmbedding.Size; i++)
            positionEmbedding.Data[i] = (float)(rand.NextDouble () * 0.02 - 0.01);

        layerNorm1 = new LayerNorm[numLayers];
        layerNorm2 = new LayerNorm[numLayers];
        attnLayers = new MultiHeadAttention[numLayers];
        mlpLayers = new MLP[numLayers];

        for (int i = 0; i < numLayers; i++) {
            layerNorm1[i] = new LayerNorm (embedSize);
            layerNorm2[i] = new LayerNorm (embedSize);
            attnLayers[i] = new MultiHeadAttention (embedSize, numHeads);
            mlpLayers[i] = new MLP (embedSize);
        }
    }

    public Tensor Forward (int[][] x) {
        // x: [batchSize][seqLen]
        int batchSize = x.Length;
        int seqLen = x[0].Length;
        Tensor xTensor = new Tensor (new int[] {
            batchSize,
            seqLen,
            embedSize
        });

        // Token and position embeddings
        for (int b = 0; b < batchSize; b++)
        for (int t = 0; t < seqLen; t++) {
            int token = x[b][t];
            for (int k = 0; k < embedSize; k++) {
                xTensor.Data[b * seqLen * embedSize + t * embedSize + k] =
                    tokenEmbedding.Data[token * embedSize + k] +
                    positionEmbedding.Data[t * embedSize + k];
            }
        }

        // Transformer blocks
        for (int l = 0; l < numLayers; l++) {
            Tensor residual = xTensor;
            xTensor = layerNorm1[l].Forward (xTensor);
            xTensor = attnLayers[l].Forward (xTensor, seqLen);
            xTensor = xTensor + residual;

            residual = xTensor;
            xTensor = layerNorm2[l].Forward (xTensor);
            xTensor = mlpLayers[l].Forward (xTensor);
            xTensor = xTensor + residual;
        }

        return xTensor;
    }

    public void ZeroGrad () {
        // Zero gradients for all parameters
        foreach (var ln in layerNorm1)
            ln.ZeroGrad ();
        foreach (var ln in layerNorm2)
            ln.ZeroGrad ();
        foreach (var attn in attnLayers)
            attn.ZeroGrad ();
        foreach (var mlp in mlpLayers)
            mlp.ZeroGrad ();

        tokenEmbedding.ZeroGrad ();
        positionEmbedding.ZeroGrad ();
    }

    public void UpdateParameters (float lr) {
        // Update model parameters
        foreach (var ln in layerNorm1)
            ln.UpdateParameters (lr);
        foreach (var ln in layerNorm2)
            ln.UpdateParameters (lr);
        foreach (var attn in attnLayers)
            attn.UpdateParameters (lr);
        foreach (var mlp in mlpLayers)
            mlp.UpdateParameters (lr);

        for (int i = 0; i < tokenEmbedding.Size; i++)
            tokenEmbedding.Data[i] -= lr * tokenEmbedding.Grad.Data[i];

        for (int i = 0; i < positionEmbedding.Size; i++)
            positionEmbedding.Data[i] -= lr * positionEmbedding.Grad.Data[i];
    }
}

// Multi-head attention
class MultiHeadAttention
{
    int embedSize;
    int numHeads;
    int headDim;

    public Tensor Wq; // [embedSize, embedSize]
    public Tensor Wk; // [embedSize, embedSize]
    public Tensor Wv; // [embedSize, embedSize]
    public Tensor Wo; // [embedSize, embedSize]

    public MultiHeadAttention (int embedSize, int numHeads) {
        this.embedSize = embedSize;
        this.numHeads = numHeads;
        headDim = embedSize / numHeads;

        Random rand = new Random ();

        Wq = new Tensor (new int[] {
            embedSize,
            embedSize
        });
        Wk = new Tensor (new int[] {
            embedSize,
            embedSize
        });
        Wv = new Tensor (new int[] {
            embedSize,
            embedSize
        });
        Wo = new Tensor (new int[] {
            embedSize,
            embedSize
        });

        Wq.InitWeights (rand);
        Wk.InitWeights (rand);
        Wv.InitWeights (rand);
        Wo.InitWeights (rand);
    }

    public Tensor Forward (Tensor x, int seqLen) {
        // x: [batchSize, seqLen, embedSize]
        int batchSize = x.Shape[0];

        Tensor Q = x.Reshape (new int[] {
            batchSize * seqLen,
            embedSize
        }).MatMul (Wq).Reshape (new int[] {
            batchSize,
            seqLen,
            numHeads,
            headDim
        });
        Tensor K = x.Reshape (new int[] {
            batchSize * seqLen,
            embedSize
        }).MatMul (Wk).Reshape (new int[] {
            batchSize,
            seqLen,
            numHeads,
            headDim
        });
        Tensor V = x.Reshape (new int[] {
            batchSize * seqLen,
            embedSize
        }).MatMul (Wv).Reshape (new int[] {
            batchSize,
            seqLen,
            numHeads,
            headDim
        });

        // Transpose to [batchSize, numHeads, seqLen, headDim]
        Q = Q.Transpose (new int[] {
            0,
            2,
            1,
            3
        });
        K = K.Transpose (new int[] {
            0,
            2,
            1,
            3
        });
        V = V.Transpose (new int[] {
            0,
            2,
            1,
            3
        });

        // Scaled dot-product attention
        Tensor scores = Q.MatMul (K.Transpose (new int[] {
            0,
            1,
            3,
            2
        })) * (1.0f / (float)Math.Sqrt (headDim));
        // Apply mask to prevent attending to future positions
        scores.ApplyMask (CreateAttentionMask (seqLen));

        Tensor attn = scores.Softmax ();
        Tensor context = attn.MatMul (V);

        // Transpose back and concatenate heads
        context = context.Transpose (new int[] {
            0,
            2,
            1,
            3
        }).Reshape (new int[] {
            batchSize,
            seqLen,
            embedSize
        });

        // Final linear layer
        Tensor output = context.Reshape (new int[] {
            batchSize * seqLen,
            embedSize
        }).MatMul (Wo).Reshape (new int[] {
            batchSize,
            seqLen,
            embedSize
        });

        return output;
    }

    public void ZeroGrad () {
        Wq.ZeroGrad ();
        Wk.ZeroGrad ();
        Wv.ZeroGrad ();
        Wo.ZeroGrad ();
    }

    public void UpdateParameters (float lr) {
        for (int i = 0; i < Wq.Size; i++)
            Wq.Data[i] -= lr * Wq.Grad.Data[i];
        for (int i = 0; i < Wk.Size; i++)
            Wk.Data[i] -= lr * Wk.Grad.Data[i];
        for (int i = 0; i < Wv.Size; i++)
            Wv.Data[i] -= lr * Wv.Grad.Data[i];
        for (int i = 0; i < Wo.Size; i++)
            Wo.Data[i] -= lr * Wo.Grad.Data[i];
    }

    // Helper methods
    bool[,] CreateAttentionMask (int seqLen) {
        bool[,] mask = new bool[seqLen, seqLen];
        for (int i = 0; i < seqLen; i++)
        for (int j = 0; j <= i; j++)
            mask[i, j] = true;
        return mask;
    }
}

// MLP layer
class MLP
{
    int embedSize;
    int hiddenSize;
    public Tensor W1; // [embedSize, hiddenSize]
    public Tensor W2; // [hiddenSize, embedSize]

    public MLP (int embedSize) {
        this.embedSize = embedSize;
        hiddenSize = 4 * embedSize;
        Random rand = new Random ();

        W1 = new Tensor (new int[] {
            embedSize,
            hiddenSize
        });
        W2 = new Tensor (new int[] {
            hiddenSize,
            embedSize
        });

        W1.InitWeights (rand);
        W2.InitWeights (rand);
    }

    public Tensor Forward (Tensor x) {
        // x: [batchSize, seqLen, embedSize]
        int batchSize = x.Shape[0];
        int seqLen = x.Shape[1];

        Tensor x1 = x.Reshape (new int[] {
            batchSize * seqLen,
            embedSize
        }).MatMul (W1);
        x1 = x1.GELU ();
        Tensor x2 = x1.MatMul (W2);
        x2 = x2.Reshape (new int[] {
            batchSize,
            seqLen,
            embedSize
        });

        return x2;
    }

    public void ZeroGrad () {
        W1.ZeroGrad ();
        W2.ZeroGrad ();
    }

    public void UpdateParameters (float lr) {
        for (int i = 0; i < W1.Size; i++)
            W1.Data[i] -= lr * W1.Grad.Data[i];
        for (int i = 0; i < W2.Size; i++)
            W2.Data[i] -= lr * W2.Grad.Data[i];
    }
}

// Layer normalization
class LayerNorm
{
    int embedSize;
    public Tensor gamma;
    public Tensor beta;

    public LayerNorm (int embedSize) {
        this.embedSize = embedSize;
        gamma = new Tensor (new int[] { embedSize });
        beta = new Tensor (new int[] { embedSize });

        gamma.Fill (1);
        beta.Fill (0);
    }

    public Tensor Forward (Tensor x) {
        // x: [batchSize, seqLen, embedSize]
        int batchSize = x.Shape[0];
        int seqLen = x.Shape[1];

        Tensor mean = x.Mean (axis: 2); // [batchSize, seqLen, 1]
        Tensor variance = x.Var (axis: 2); // [batchSize, seqLen, 1]

        Tensor xNorm = (x - mean) / (variance + 1e-5f).Sqrt ();
        Tensor y = gamma * xNorm + beta;

        return y;
    }

    public void ZeroGrad () {
        gamma.ZeroGrad ();
        beta.ZeroGrad ();
    }

    public void UpdateParameters (float lr) {
        for (int i = 0; i < gamma.Size; i++)
            gamma.Data[i] -= lr * gamma.Grad.Data[i];
        for (int i = 0; i < beta.Size; i++)
            beta.Data[i] -= lr * beta.Grad.Data[i];
    }
}

// Trainer class
class Trainer
{
    MinGPT2 model;
    int[][] data;
    int epochs;
    int batchSize;
    float learningRate;

    public Trainer (MinGPT2 model, int[][] data, int epochs = 10, int batchSize = 32, float learningRate = 1e-3f) {
        this.model = model;
        this.data = data;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
    }

    public void Train () {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Shuffle (data);
            for (int i = 0; i < data.Length; i += batchSize) {
                int[][] batch = GetBatch (data, i, batchSize);
                int maxSeqLen = GetMaxSequenceLength (batch);
                int[][] inputs = PadSequences (batch, maxSeqLen);
                int[][] targets = GetTargets (batch, maxSeqLen);

                // Forward pass
                Tensor logits = model.Forward (inputs); // [batchSize, seqLen, embedSize]

                // Compute loss and gradients
                float loss = ComputeLoss (logits, targets);
                model.ZeroGrad ();
                logits.Backward ();

                // Update parameters
                model.UpdateParameters (learningRate);
            }

            Console.WriteLine ($"Epoch {epoch + 1} completed.");
        }
    }

    int GetMaxSequenceLength (int[][] batch) {
        int maxLen = 0;
        foreach (var seq in batch)
            if (seq.Length > maxLen)
                maxLen = seq.Length;
        return maxLen;
    }

    int[][] PadSequences (int[][] sequences, int maxLen) {
        int[][] padded = new int[sequences.Length][];
        for (int i = 0; i < sequences.Length; i++) {
            int[] seq = sequences[i];
            padded[i] = new int[maxLen];
            Array.Copy (seq, padded[i], seq.Length);
            for (int j = seq.Length; j < maxLen; j++)
                padded[i][j] = 0; // Assume 0 is the padding token
        }

        return padded;
    }

    int[][] GetTargets (int[][] sequences, int maxLen) {
        int[][] targets = new int[sequences.Length][];
        for (int i = 0; i < sequences.Length; i++) {
            int[] seq = sequences[i];
            targets[i] = new int[maxLen];
            for (int j = 0; j < seq.Length - 1; j++)
                targets[i][j] = seq[j + 1];
            targets[i][seq.Length - 1] = 0; // Next token for the last token
            for (int j = seq.Length; j < maxLen; j++)
                targets[i][j] = 0;
        }

        return targets;
    }

    float ComputeLoss (Tensor logits, int[][] targets) {
        // logits: [batchSize, seqLen, embedSize]
        // targets: [batchSize][seqLen]
        int batchSize = logits.Shape[0];
        int seqLen = logits.Shape[1];
        int vocabSize = model.tokenEmbedding.Shape[0];

        float loss = 0;
        int count = 0;

        for (int b = 0; b < batchSize; b++)
        for (int t = 0; t < seqLen; t++) {
            int target = targets[b][t];
            if (target == 0) continue; // Skip padding tokens

            // Get logits for this position
            float[] logitSlice = new float[vocabSize];
            for (int v = 0; v < vocabSize; v++)
                logitSlice[v] = model.tokenEmbedding.Data[v * model.tokenEmbedding.Shape[1] + t];

            // Compute softmax
            float maxLogit = logitSlice.Max ();
            float sumExp = 0;
            for (int v = 0; v < vocabSize; v++)
                sumExp += (float)Math.Exp (logitSlice[v] - maxLogit);

            float logProb = logitSlice[target] - maxLogit - (float)Math.Log (sumExp);
            loss -= logProb;
            count++;
        }

        return loss / count;
    }

    void Shuffle<T> (T[] array) {
        Random rng = new Random ();
        int n = array.Length;
        while (n > 1) {
            int k = rng.Next (n--);
            T temp = array[n];
            array[n] = array[k];
            array[k] = temp;
        }
    }

    int[][] GetBatch (int[][] data, int start, int batchSize) {
        int end = Math.Min (start + batchSize, data.Length);
        int[][] batch = new int[end - start][];
        Array.Copy (data, start, batch, 0, end - start);
        return batch;
    }
}

class MinGPT2Trainer
{
    public static void run () {
        MinGPT2 gpt = new MinGPT2 ();
        int[][] trainingData = LoadData ();
        Trainer trainer = new Trainer (gpt, trainingData, epochs: 10, batchSize: 32, learningRate: 1e-3f);
        trainer.Train ();
    }

    static int[][] LoadData () {
        var text = File.ReadAllText ("resources/tinyshakespeare.txt");
        var data = text.Select (_ => (int)_).ToArray ();

        var sequenceLength = 128;

        int batches = data.Length / sequenceLength;

        // Load and preprocess your training data here
        // Placeholder for data loading
        // Return an array of sequences (arrays of token IDs)
        return Enumerable
            .Range (0, batches)
            .Select (i => data
                .Skip (i * sequenceLength)
                .Take (sequenceLength)
                .ToArray ())
            .ToArray ();
    }
}
