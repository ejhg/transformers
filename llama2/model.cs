using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics; // For Complex numbers

namespace TransformerModel
{
    public class ModelArgs
    {
        public int dim { get; set; } = 4096;
        public int n_layers { get; set; } = 32;
        public int n_heads { get; set; } = 32;
        public int? n_kv_heads { get; set; } = null;
        public int vocab_size { get; set; } = -1; // defined later by tokenizer
        public int multiple_of { get; set; } = 256; // make SwiGLU hidden layer size multiple of large power of 2
        public double? ffn_dim_multiplier { get; set; } = null;
        public double norm_eps { get; set; } = 1e-5;
        public int max_batch_size { get; set; } = 32;
        public int max_seq_len { get; set; } = 2048;
    }

    public abstract class Module
    {
        public abstract Tensor Forward(Tensor x);
    }

    public class RMSNorm : Module
    {
        private double eps;
        private Tensor weight;

        public RMSNorm(int dim, double eps = 1e-6)
        {
            this.eps = eps;
            this.weight = Tensor.Ones(dim);
        }

        private Tensor _norm(Tensor x)
        {
            var x2 = x.Pow(2);
            var mean = x2.Mean(-1, keepdim: true);
            var denom = (mean + this.eps).Rsqrt();
            return x * denom;
        }

        public override Tensor Forward(Tensor x)
        {
            var output = _norm(x.Float()).TypeAs(x);
            return output * this.weight;
        }
    }

    public static class Functions
    {
        public static Tensor PrecomputeFreqsCis(int dim, int end, double theta = 10000.0)
        {
            var freqs = Tensor.Arange(0, dim, 2).Slice(0, dim / 2).Float();
            freqs = 1.0 / Tensor.Pow(theta, freqs / dim);
            var t = Tensor.Arange(end);
            freqs = Tensor.Outer(t, freqs).Float();
            var freqs_cis = Tensor.Polar(Tensor.OnesLike(freqs), freqs);
            return freqs_cis;
        }

        public static Tensor ReshapeForBroadcast(Tensor freqs_cis, Tensor x)
        {
            var ndim = x.Dimensions;
            var shape = x.Shape.Select((d, i) => (i == 1 || i == ndim - 1) ? d : 1).ToArray();
            return freqs_cis.View(shape);
        }

        public static (Tensor, Tensor) ApplyRotaryEmb(Tensor xq, Tensor xk, Tensor freqs_cis)
        {
            var xq_ = xq.Float().Reshape(xq.Shape.Take(xq.Dimensions - 1).Concat(new[] { -1, 2 }).ToArray()).ViewAsComplex();
            var xk_ = xk.Float().Reshape(xk.Shape.Take(xk.Dimensions - 1).Concat(new[] { -1, 2 }).ToArray()).ViewAsComplex();
            freqs_cis = ReshapeForBroadcast(freqs_cis, xq_);
            var xq_out = (xq_ * freqs_cis).ViewAsReal().Flatten(-2, -1);
            var xk_out = (xk_ * freqs_cis).ViewAsReal().Flatten(-2, -1);
            return (xq_out.TypeAs(xq), xk_out.TypeAs(xk));
        }

        public static Tensor RepeatKv(Tensor x, int n_rep)
        {
            var shape = x.Shape;
            if (n_rep == 1) return x;
            var expandedShape = new int[] { shape[0], shape[1], shape[2], n_rep, shape[3] };
            var xExpanded = x.Unsqueeze(3).Expand(expandedShape);
            var newShape = new int[] { shape[0], shape[1], shape[2] * n_rep, shape[3] };
            return xExpanded.Contiguous().Reshape(newShape);
        }
    }

    public class Attention : Module
    {
        private int n_kv_heads;
        private int n_local_heads;
        private int n_local_kv_heads;
        private int n_rep;
        private int head_dim;
        private Linear wq;
        private Linear wk;
        private Linear wv;
        private Linear wo;
        private Tensor cache_k;
        private Tensor cache_v;
        private ModelArgs args;

        public Attention(ModelArgs args)
        {
            this.args = args;
            this.n_kv_heads = args.n_kv_heads ?? args.n_heads;
            this.n_local_heads = args.n_heads;
            this.n_local_kv_heads = this.n_kv_heads;
            this.n_rep = this.n_local_heads / this.n_local_kv_heads;
            this.head_dim = args.dim / args.n_heads;

            this.wq = new Linear(args.dim, args.n_heads * this.head_dim, bias: false);
            this.wk = new Linear(args.dim, this.n_kv_heads * this.head_dim, bias: false);
            this.wv = new Linear(args.dim, this.n_kv_heads * this.head_dim, bias: false);
            this.wo = new Linear(args.n_heads * this.head_dim, args.dim, bias: false);

            this.cache_k = Tensor.Zeros(args.max_batch_size, args.max_seq_len, this.n_local_kv_heads, this.head_dim);
            this.cache_v = Tensor.Zeros(args.max_batch_size, args.max_seq_len, this.n_local_kv_heads, this.head_dim);
        }

        public Tensor Forward(Tensor x, int start_pos, Tensor freqs_cis, Tensor mask = null)
        {
            int bsz = x.Shape[0];
            int seqlen = x.Shape[1];

            var xq = this.wq.Forward(x);
            var xk = this.wk.Forward(x);
            var xv = this.wv.Forward(x);

            xq = xq.View(bsz, seqlen, this.n_local_heads, this.head_dim);
            xk = xk.View(bsz, seqlen, this.n_local_kv_heads, this.head_dim);
            xv = xv.View(bsz, seqlen, this.n_local_kv_heads, this.head_dim);

            var (xq_rot, xk_rot) = Functions.ApplyRotaryEmb(xq, xk, freqs_cis);

            this.cache_k.SliceAssign(xk_rot, bsz, start_pos, seqlen);
            this.cache_v.SliceAssign(xv, bsz, start_pos, seqlen);

            var keys = this.cache_k.Slice(bsz, 0, start_pos + seqlen);
            var values = this.cache_v.Slice(bsz, 0, start_pos + seqlen);

            keys = Functions.RepeatKv(keys, this.n_rep);
            values = Functions.RepeatKv(values, this.n_rep);

            xq_rot = xq_rot.Transpose(1, 2); // (bsz, n_local_heads, seqlen, head_dim)
            keys = keys.Transpose(1, 2);     // (bsz, n_local_heads, cache_len, head_dim)
            values = values.Transpose(1, 2); // (bsz, n_local_heads, cache_len, head_dim)

            var scores = xq_rot.MatMul(keys.Transpose(-2, -1)) / Math.Sqrt(this.head_dim);

            if (mask != null)
            {
                scores = scores + mask;
            }

            var probs = scores.Softmax(-1);

            var output = probs.MatMul(values); // (bsz, n_local_heads, seqlen, head_dim)
            output = output.Transpose(1, 2).Contiguous().Reshape(bsz, seqlen, -1);

            return this.wo.Forward(output);
        }

        public override Tensor Forward(Tensor x)
        {
            throw new NotImplementedException();
        }
    }

    public class FeedForward : Module
    {
        private Linear w1;
        private Linear w2;
        private Linear w3;

        public FeedForward(int dim, int hidden_dim, int multiple_of, double? ffn_dim_multiplier)
        {
            hidden_dim = (2 * hidden_dim) / 3;
            if (ffn_dim_multiplier.HasValue)
            {
                hidden_dim = (int)(ffn_dim_multiplier.Value * hidden_dim);
            }
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);

            this.w1 = new Linear(dim, hidden_dim, bias: false);
            this.w2 = new Linear(hidden_dim, dim, bias: false);
            this.w3 = new Linear(dim, hidden_dim, bias: false);
        }

        public override Tensor Forward(Tensor x)
        {
            var w1x = this.w1.Forward(x);
            var w3x = this.w3.Forward(x);
            var silu = w1x.Silu();
            var elementWise = silu * w3x;
            var output = this.w2.Forward(elementWise);
            return output;
        }
    }

    public class TransformerBlock : Module
    {
        private Attention attention;
        private FeedForward feed_forward;
        private RMSNorm attention_norm;
        private RMSNorm ffn_norm;

        public TransformerBlock(int layer_id, ModelArgs args)
        {
            this.attention = new Attention(args);
            this.feed_forward = new FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier);
            this.attention_norm = new RMSNorm(args.dim, args.norm_eps);
            this.ffn_norm = new RMSNorm(args.dim, args.norm_eps);
        }

        public Tensor Forward(Tensor x, int start_pos, Tensor freqs_cis, Tensor mask = null)
        {
            var h = x + this.attention.Forward(this.attention_norm.Forward(x), start_pos, freqs_cis, mask);
            var output = h + this.feed_forward.Forward(this.ffn_norm.Forward(h));
            return output;
        }

        public override Tensor Forward(Tensor x)
        {
            throw new NotImplementedException();
        }
    }

    public class Transformer : Module
    {
        private ModelArgs args;
        private Embedding tok_embeddings;
        private List<TransformerBlock> layers;
        private RMSNorm norm;
        private Linear output;
        private Tensor freqs_cis;

        public Transformer(ModelArgs args)
        {
            this.args = args;
            this.tok_embeddings = new Embedding(args.vocab_size, args.dim);
            this.layers = new List<TransformerBlock>();
            for (int i = 0; i < args.n_layers; i++)
            {
                this.layers.Add(new TransformerBlock(i, args));
            }
            this.norm = new RMSNorm(args.dim, args.norm_eps);
            this.output = new Linear(args.dim, args.vocab_size, bias: false);

            this.freqs_cis = Functions.PrecomputeFreqsCis(args.dim / args.n_heads, args.max_seq_len * 2);
        }

        public Tensor Forward(Tensor tokens, int start_pos)
        {
            int bsz = tokens.Shape[0];
            int seqlen = tokens.Shape[1];
            var h = this.tok_embeddings.Forward(tokens);
            var freqs_cis = this.freqs_cis.Slice(start_pos, start_pos + seqlen);

            Tensor mask = null;
            if (seqlen > 1)
            {
                mask = Tensor.Full(new[] { 1, 1, seqlen, seqlen }, double.NegativeInfinity);
                mask = mask.Triu(start_pos + 1).TypeAs(h);
            }

            foreach (var layer in this.layers)
            {
                h = layer.Forward(h, start_pos, freqs_cis, mask);
            }

            h = this.norm.Forward(h);
            var output = this.output.Forward(h).Float();
            return output;
        }

        public override Tensor Forward(Tensor x)
        {
            throw new NotImplementedException();
        }
    }

    // Implementations for Tensor, Linear, Embedding, and other necessary classes.

    public class Tensor
    {
        public double[] Data;
        public int[] Shape { get; private set; }
        public int Dimensions => Shape.Length;

        public Tensor(double[] data, int[] shape)
        {
            this.Data = data;
            this.Shape = shape;
        }

        public static Tensor Ones(int dim)
        {
            var data = Enumerable.Repeat(1.0, dim).ToArray();
            return new Tensor(data, new[] { dim });
        }

        public static Tensor OnesLike(Tensor x)
        {
            var data = Enumerable.Repeat(1.0, x.Data.Length).ToArray();
            return new Tensor(data, x.Shape);
        }

        public static Tensor Zeros(params int[] shape)
        {
            int size = shape.Aggregate(1, (a, b) => a * b);
            var data = new double[size];
            return new Tensor(data, shape);
        }

        public static Tensor Full(int[] shape, double value)
        {
            int size = shape.Aggregate(1, (a, b) => a * b);
            var data = Enumerable.Repeat(value, size).ToArray();
            return new Tensor(data, shape);
        }

        public static Tensor Arange(int start, int end, int step = 1)
        {
            var data = Enumerable.Range(start, (end - start) / step).Select(i => (double)(start + i * step)).ToArray();
            return new Tensor(data, new[] { data.Length });
        }

        public static Tensor Arange(int end)
        {
            return Arange(0, end, 1);
        }

        public static Tensor Pow(double a, Tensor x)
        {
            var data = x.Data.Select(v => Math.Pow(a, v)).ToArray();
            return new Tensor(data, x.Shape);
        }

        public Tensor Pow(double exponent)
        {
            var data = Data.Select(v => Math.Pow(v, exponent)).ToArray();
            return new Tensor(data, Shape);
        }

        public Tensor Mean(int dim, bool keepdim = false)
        {
            var newShape = Shape.ToArray();
            newShape[dim] = 1;
            int stride = Shape.Skip(dim + 1).Aggregate(1, (a, b) => a * b);
            int count = Shape[dim];
            var data = new double[Data.Length / count];
            for (int i = 0; i < data.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < count; j++)
                {
                    sum += Data[i * count + j];
                }
                data[i] = sum / count;
            }
            return new Tensor(data, keepdim ? newShape : newShape.Where((v, idx) => idx != dim).ToArray());
        }

        public Tensor Rsqrt()
        {
            var data = Data.Select(v => 1.0 / Math.Sqrt(v)).ToArray();
            return new Tensor(data, Shape);
        }

        public Tensor Float()
        {
            return this;
        }

        public Tensor TypeAs(Tensor x)
        {
            return this;
        }

        public Tensor Mul(Tensor other)
        {
            var data = Data.Zip(other.Data, (a, b) => a * b).ToArray();
            return new Tensor(data, Shape);
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            return a.Mul(b);
        }

        public static Tensor operator *(Tensor a, double b)
        {
            var data = a.Data.Select(x => x * b).ToArray();
            return new Tensor(data, a.Shape);
        }

        public static Tensor operator *(double a, Tensor b)
        {
            return b * a;
        }

        public Tensor Add(Tensor other)
        {
            var data = Data.Zip(other.Data, (a, b) => a + b).ToArray();
            return new Tensor(data, Shape);
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            return a.Add(b);
        }

        public static Tensor operator +(Tensor a, double b)
        {
            var data = a.Data.Select(x => x + b).ToArray();
            return new Tensor(data, a.Shape);
        }

        public static Tensor operator +(double a, Tensor b)
        {
            return b + a;
        }

        public Tensor Sub(Tensor other)
        {
            var data = Data.Zip(other.Data, (a, b) => a - b).ToArray();
            return new Tensor(data, Shape);
        }

        public static Tensor operator -(Tensor a, Tensor b)
        {
            return a.Sub(b);
        }

        public static Tensor operator -(Tensor a, double b)
        {
            var data = a.Data.Select(x => x - b).ToArray();
            return new Tensor(data, a.Shape);
        }

        public static Tensor operator -(double a, Tensor b)
        {
            var data = b.Data.Select(x => a - x).ToArray();
            return new Tensor(data, b.Shape);
        }

        public Tensor Div(double value)
        {
            var data = Data.Select(v => v / value).ToArray();
            return new Tensor(data, Shape);
        }

        public static Tensor operator /(Tensor a, double b)
        {
            return a.Div(b);
        }

        public static Tensor operator /(double a, Tensor b)
        {
            var data = b.Data.Select(v => a / v).ToArray();
            return new Tensor(data, b.Shape);
        }

        public Tensor Silu()
        {
            var data = Data.Select(v => v / (1 + Math.Exp(-v))).ToArray();
            return new Tensor(data, Shape);
        }

        public Tensor MatMul(Tensor other)
        {
            // Simplified implementation for 2D matrices
            if (Dimensions != 2 || other.Dimensions != 2)
                throw new NotImplementedException("MatMul only implemented for 2D tensors.");

            int m = Shape[0];
            int k = Shape[1];
            int n = other.Shape[1];
            if (k != other.Shape[0])
                throw new InvalidOperationException("Matrix dimensions do not match for multiplication.");

            var resultData = new double[m * n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int p = 0; p < k; p++)
                    {
                        sum += Data[i * k + p] * other.Data[p * n + j];
                    }
                    resultData[i * n + j] = sum;
                }
            }
            return new Tensor(resultData, new[] { m, n });
        }

        public Tensor Softmax(int dim)
        {
            // Simplified softmax along the specified dimension
            int[] newShape = Shape.ToArray();
            int stride = Shape.Skip(dim + 1).Aggregate(1, (a, b) => a * b);
            int count = Shape[dim];
            var data = new double[Data.Length];
            for (int i = 0; i < Data.Length / count; i++)
            {
                double maxVal = Data.Skip(i * count).Take(count).Max();
                double sumExp = 0;
                for (int j = 0; j < count; j++)
                {
                    double expVal = Math.Exp(Data[i * count + j] - maxVal);
                    data[i * count + j] = expVal;
                    sumExp += expVal;
                }
                for (int j = 0; j < count; j++)
                {
                    data[i * count + j] /= sumExp;
                }
            }
            return new Tensor(data, Shape);
        }

        public Tensor Transpose(int dim1, int dim2)
        {
            int[] newShape = Shape.ToArray();
            int temp = newShape[dim1];
            newShape[dim1] = newShape[dim2];
            newShape[dim2] = temp;

            // Simplified transpose, only swaps the two dimensions
            var newData = new double[Data.Length];
            var indices = new int[Dimensions];
            var strides = ComputeStrides(Shape);
            var newStrides = ComputeStrides(newShape);

            for (int i = 0; i < Data.Length; i++)
            {
                GetIndices(i, strides, indices);
                int tempIdx = indices[dim1];
                indices[dim1] = indices[dim2];
                indices[dim2] = tempIdx;
                int newIdx = GetFlatIndex(indices, newStrides);
                newData[newIdx] = Data[i];
            }

            return new Tensor(newData, newShape);
        }

        public Tensor Contiguous()
        {
            return this;
        }

        public Tensor Reshape(params int[] newShape)
        {
            int newSize = newShape.Aggregate(1, (a, b) => a * b);
            if (newSize != Data.Length)
                throw new InvalidOperationException("Cannot reshape array of size " + Data.Length + " into shape " + string.Join(", ", newShape));
            return new Tensor(Data, newShape);
        }

        public Tensor View(params int[] newShape)
        {
            return Reshape(newShape);
        }

        public Tensor Flatten(int startDim, int endDim)
        {
            var preShape = Shape.Take(startDim).ToList();
            var midShape = Shape.Skip(startDim).Take(endDim - startDim + 1).Aggregate(1, (a, b) => a * b);
            var postShape = Shape.Skip(endDim + 1).ToList();
            var newShape = preShape.Concat(new[] { midShape }).Concat(postShape).ToArray();
            return Reshape(newShape);
        }

        public Tensor Unsqueeze(int dim)
        {
            var newShape = Shape.ToList();
            newShape.Insert(dim, 1);
            return Reshape(newShape.ToArray());
        }

        public Tensor Expand(int[] newShape)
        {
            // Simplified expand implementation
            if (newShape.Length != Shape.Length)
                throw new InvalidOperationException("Shape mismatch in expand operation.");
            for (int i = 0; i < Shape.Length; i++)
            {
                if (Shape[i] != 1 && Shape[i] != newShape[i])
                    throw new InvalidOperationException("Cannot expand dimension " + i);
            }
            int newSize = newShape.Aggregate(1, (a, b) => a * b);
            var newData = new double[newSize];
            var strides = ComputeStrides(Shape);
            var newStrides = ComputeStrides(newShape);

            for (int i = 0; i < newSize; i++)
            {
                var indices = new int[newShape.Length];
                GetIndices(i, newStrides, indices);
                var oldIndices = indices.Select((idx, dim) => Shape[dim] == 1 ? 0 : idx).ToArray();
                int oldIdx = GetFlatIndex(oldIndices, strides);
                newData[i] = Data[oldIdx];
            }

            return new Tensor(newData, newShape);
        }

        private int[] ComputeStrides(int[] shape)
        {
            var strides = new int[shape.Length];
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }

        private void GetIndices(int flatIndex, int[] strides, int[] indices)
        {
            for (int i = 0; i < strides.Length; i++)
            {
                indices[i] = flatIndex / strides[i];
                flatIndex %= strides[i];
            }
        }

        private int GetFlatIndex(int[] indices, int[] strides)
        {
            int flatIndex = 0;
            for (int i = 0; i < strides.Length; i++)
            {
                flatIndex += indices[i] * strides[i];
            }
            return flatIndex;
        }

        public Tensor SliceAssign(Tensor source, int bsz, int start_pos, int seqlen)
        {
            // Simplified slice assignment
            int size = source.Data.Length;
            Array.Copy(source.Data, 0, Data, bsz * Shape[1] * Shape[2] * Shape[3] + start_pos * Shape[2] * Shape[3], size);
            return this;
        }

        public Tensor Slice(int bsz, int start, int end)
        {
            int length = (end - start) * Shape[2] * Shape[3];
            var data = new double[length];
            Array.Copy(Data, bsz * Shape[1] * Shape[2] * Shape[3] + start * Shape[2] * Shape[3], data, 0, length);
            var newShape = new[] { bsz, end - start, Shape[2], Shape[3] };
            return new Tensor(data, newShape);
        }

        public static Tensor Polar(Tensor abs, Tensor angle)
        {
            var data = abs.Data.Zip(angle.Data, (r, theta) => new Complex(r * Math.Cos(theta), r * Math.Sin(theta))).ToArray();
            return new ComplexTensor(data, abs.Shape);
        }

        public Tensor ViewAsComplex()
        {
            // Simplified conversion to complex numbers
            var data = new Complex[Data.Length / 2];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = new Complex(Data[2 * i], Data[2 * i + 1]);
            }
            return new ComplexTensor(data, Shape.Take(Shape.Length - 1).Concat(new[] { Shape[Shape.Length - 1] / 2 }).ToArray());
        }

        public Tensor ViewAsReal()
        {
            throw new NotImplementedException();
        }

        public Tensor Triu(int diagonal)
        {
            var data = Data.ToArray();
            int n = Shape[Shape.Length - 1];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < Math.Min(i + diagonal, n); j++)
                {
                    data[i * n + j] = 0;
                }
            }
            return new Tensor(data, Shape);
        }

        public Tensor Slice(int start, int end)
        {
            int length = (end - start) * Shape.Skip(1).Aggregate(1, (a, b) => a * b);
            var data = new double[length];
            Array.Copy(Data, start * Shape.Skip(1).Aggregate(1, (a, b) => a * b), data, 0, length);
            var newShape = new[] { end - start }.Concat(Shape.Skip(1)).ToArray();
            return new Tensor(data, newShape);
        }

        public static Tensor Outer(Tensor a, Tensor b)
        {
            var data = new double[a.Data.Length * b.Data.Length];
            int idx = 0;
            foreach (var av in a.Data)
            {
                foreach (var bv in b.Data)
                {
                    data[idx++] = av * bv;
                }
            }
            return new Tensor(data, new[] { a.Data.Length, b.Data.Length });
        }
    }

    public class ComplexTensor : Tensor
    {
        public Complex[] ComplexData;

        public ComplexTensor(Complex[] data, int[] shape) : base(null, shape)
        {
            this.ComplexData = data;
        }

        public static ComplexTensor operator *(ComplexTensor a, ComplexTensor b)
        {
            var data = a.ComplexData.Zip(b.ComplexData, (x, y) => x * y).ToArray();
            return new ComplexTensor(data, a.Shape);
        }

        public Tensor ViewAsReal()
        {
            var data = new double[ComplexData.Length * 2];
            for (int i = 0; i < ComplexData.Length; i++)
            {
                data[2 * i] = ComplexData[i].Real;
                data[2 * i + 1] = ComplexData[i].Imaginary;
            }
            var shape = Shape.Take(Shape.Length - 1).Concat(new[] { Shape[Shape.Length - 1] * 2 }).ToArray();
            return new Tensor(data, shape);
        }
    }

    public class Linear : Module
    {
        private int input_dim;
        private int output_dim;
        private bool bias;
        private Tensor weight;
        private Tensor biasTensor;

        public Linear(int input_dim, int output_dim, bool bias = true)
        {
            this.input_dim = input_dim;
            this.output_dim = output_dim;
            this.bias = bias;
            this.weight = Tensor.Zeros(input_dim, output_dim); // Initialize weights
            if (bias)
                this.biasTensor = Tensor.Zeros(output_dim); // Initialize bias
        }

        public override Tensor Forward(Tensor x)
        {
            var output = x.MatMul(this.weight);
            if (this.bias)
                output = output + this.biasTensor;
            return output;
        }
    }

    public class Embedding : Module
    {
        private int num_embeddings;
        private int embedding_dim;
        private Tensor weight;

        public Embedding(int num_embeddings, int embedding_dim)
        {
            this.num_embeddings = num_embeddings;
            this.embedding_dim = embedding_dim;
            this.weight = Tensor.Zeros(num_embeddings, embedding_dim); // Initialize embeddings
        }

        public override Tensor Forward(Tensor x)
        {
            var batchSize = x.Shape[0];
            var seqLen = x.Shape[1];
            var outputData = new double[batchSize * seqLen * embedding_dim];

            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    int idx = (int)x.Data[b * seqLen + s];
                    Array.Copy(weight.Data, idx * embedding_dim, outputData, (b * seqLen + s) * embedding_dim, embedding_dim);
                }
            }

            return new Tensor(outputData, new[] { batchSize, seqLen, embedding_dim });
        }
    }
}
