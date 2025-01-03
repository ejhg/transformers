using llama.cs;
using llama.torchsharp;

namespace llama.c;

public class Transformer
{
    public config config; // Hyperparameters
    public TransformerWeights weights; // Model weights
    public RunState state; // Run state buffers
}

static class TransformerModel
{
    static void MemoryMapWeights (TransformerWeights w, config p, float[] data, int shared_weights) {
        var head_size = p.dim / p.n_heads;
        long n_layers = p.n_layers;
        long index = 0;

        w.token_embedding_table = new float[p.vocab_size * p.dim];
        Array.Copy (data, index, w.token_embedding_table, 0, w.token_embedding_table.Length);
        index += w.token_embedding_table.Length;

        w.rms_att_weight = new float[n_layers][];
        for (var l = 0; l < n_layers; l++) {
            w.rms_att_weight[l] = new float[p.dim];
            Array.Copy (data, index, w.rms_att_weight[l], 0, p.dim);
            index += p.dim;
        }

        float[][] readLayers (int layer_size) {
            var ret = new float[n_layers][];
            for (var l = 0; l < n_layers; l++) {
                ret[l] = new float [layer_size];
                Array.Copy (data, index, ret[l], 0, ret[l].Length);
                index += ret[l].Length;
            }

            return ret;
        }

        float[][,] readLayers2D (int dim1, int dim2) {
            var ret = new float[n_layers][,];
            for (var l = 0; l < n_layers; l++) {
                ret[l] = new float [dim1, dim2];

                for (var i = 0; i < dim1; i++) {
                    for (var j = 0; j < dim2; j++) {
                        ret[l][i, j] = data[index + i * dim2 + j];
                    }
                }

                index += dim1 * dim2;
            }

            return ret;
        }

        w.wq = readLayers2D (p.n_heads * head_size, p.dim);
        w.wk = readLayers2D (p.n_kv_heads * head_size, p.dim);
        w.wv = readLayers2D (p.n_kv_heads * head_size, p.dim);
        w.wo = readLayers2D (p.dim, p.n_heads * head_size);

        w.rms_ffn_weight = new float[n_layers][];
        for (var l = 0; l < n_layers; l++) {
            w.rms_ffn_weight[l] = new float[p.dim];
            Array.Copy (data, index, w.rms_ffn_weight[l], 0, p.dim);
            index += p.dim;
        }

        w.w1 = readLayers (p.hidden_dim * p.dim);
        w.w2 = readLayers (p.dim * p.hidden_dim);
        w.w3 = readLayers (p.hidden_dim * p.dim);

        w.rms_final_weight = new float[p.dim];
        Array.Copy (data, index, w.rms_final_weight, 0, w.rms_final_weight.Length);
        index += w.rms_final_weight.Length;

        // Skip RoPE frequencies (not used in this implementation)
        index += p.seq_len * head_size / 2; // freq_cis_real
        index += p.seq_len * head_size / 2; // freq_cis_imag

        if (shared_weights != 0) {
            w.wcls = w.token_embedding_table;
        } else {
            w.wcls = new float[p.vocab_size * p.dim];
            Array.Copy (data, index, w.wcls, 0, w.wcls.Length);
        }
    }

    static void ReadCheckpoint (string checkpoint, config config, TransformerWeights weights) {
        using var fs = new FileStream (checkpoint, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader (fs);

        // Read config
        config.dim = br.ReadInt32 ();
        config.hidden_dim = br.ReadInt32 ();
        config.n_layers = br.ReadInt32 ();
        config.n_heads = br.ReadInt32 ();
        config.n_kv_heads = br.ReadInt32 ();
        config.vocab_size = br.ReadInt32 ();
        config.seq_len = br.ReadInt32 ();

        var shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = Math.Abs (config.vocab_size);

        var file_size = fs.Length;

        // Read data
        var dataSize = (int)((file_size - 7 * sizeof(int)) / sizeof(float)); // 7 ints in Config
        var data = new float[dataSize];
        for (var i = 0; i < dataSize; i++) {
            data[i] = br.ReadSingle ();
        }

        // Map weights
        MemoryMapWeights (weights, config, data, shared_weights);
    }

    public static void BuildTransformer (Transformer t, string checkpoint_path) {
        t.config = new config ();
        t.weights = new TransformerWeights ();
        t.state = new RunState ();

        ReadCheckpoint (checkpoint_path, t.config, t.weights);
        t.state.x = new float[t.config.dim];
        t.state.xb = new float[t.config.dim];
        t.state.xb2 = new float[t.config.dim];
        t.state.hb = new float[t.config.hidden_dim];
        t.state.hb2 = new float[t.config.hidden_dim];
        t.state.q = new float[t.config.n_heads, t.config.dim / t.config.n_heads];
        t.state.k = new float[t.config.n_heads, t.config.dim / t.config.n_heads];
        t.state.v = new float[t.config.n_heads, t.config.dim / t.config.n_heads];
        t.state.logits = new float[t.config.vocab_size];

        t.state.kv_cache = Enumerable
            .Range (0, t.config.n_layers)
            .Select (_ => new RunState.LayerCache {
                key_cache = new float[t.config.seq_len, t.config.n_kv_heads, t.config.dim / t.config.n_heads],
                value_cache = new float[t.config.seq_len, t.config.n_kv_heads, t.config.dim / t.config.n_heads]
            }).ToArray ();
    }

    static void MatMul (float[] xout, float[] x, float[] w, int wOffset, int n, int d) {
        if (x.Length > 1000) {
            Parallel.For (0, d, i => {
                var val = 0.0f;
                var ptr = wOffset + i * n;

                for (var j = 0; j < n; j++) {
                    val += w[ptr + j] * x[j];
                }

                xout[i] = val;
            });
        } else {
            for (var i = 0; i < d; i++) {
                var val = 0.0f;
                var ptr = wOffset + i * n;

                for (var j = 0; j < n; j++) {
                    val += w[ptr + j] * x[j];
                }

                xout[i] = val;
            }
        }
    }

    static void MatMul (float[] xout, float[] x, float[,] w, int wOffset, int n, int d) {
        for (var i = 0; i < d; i++) {
            var val = 0.0f;

            for (var j = 0; j < n; j++) {
                val += w[i, j] * x[j];
            }

            xout[i] = val;
        }
    }

    static void MatMul (float[,] xout, float[] x, float[,] w, int wOffset, int n, int d) {
        var size = xout.GetLength (0);

        for (var i = 0; i < d; i++) {
            var val = 0.0f;

            for (var j = 0; j < n; j++) {
                val += w[i, j] * x[j];
            }

            xout[i / size, i % size] = val;
        }
    }

    public static float[] Forward (Transformer transformer, int token, int pos) {
        var p = transformer.config;
        var w = transformer.weights;
        var s = transformer.state;

        var dim = p.dim;
        var head_size = p.dim / p.n_heads;
        var kv_dim = head_size * p.n_kv_heads;

        // Copy token embedding into x
        Array.Copy (w.token_embedding_table, token * dim, s.x, 0, dim);

        // Forward pass through layers
        for (var l = 0; l < p.n_layers; l++) {
            // Attention rmsnorm
            math.RmsNorm (s.xb, s.x, w.rms_att_weight[l]);

            // Compute q, k, v
            MatMul (s.q, s.xb, w.wq[l], 0, dim, dim);
            MatMul (s.k, s.xb, w.wk[l], 0, dim, kv_dim);
            MatMul (s.v, s.xb, w.wv[l], 0, dim, kv_dim);

            // RoPE positional encoding
            for (var i = 0; i < dim; i += 2) {
                var head_dim = i % (dim / p.n_heads);
                var freq = 1.0f / MathF.Pow (10000.0f, head_dim / (float)(dim / p.n_heads));
                var val = pos * freq;
                var fcr = MathF.Cos (val);
                var fci = MathF.Sin (val);
                var rotn = i < kv_dim ? 2 : 1; // Rotate q and k
                for (var v = 0; v < rotn; v++) {
                    var dst = v == 0
                        ? s.q
                        : s.k;
                    var v0 = dst[i / head_size, i % head_size];
                    var v1 = dst[(i + 1) / head_size, (i + 1) % head_size];
                    dst[i / head_size, i % head_size] = v0 * fcr - v1 * fci;
                    dst[(i + 1) / head_size, (i + 1) % head_size] = v0 * fci + v1 * fcr;
                }
            }

            // Store k and v in cache
            for (var j = 0; j < head_size; j++) {
                for (var h = 0; h < p.n_kv_heads; h++) {
                    s.kv_cache[l].key_cache[pos, h, j] = s.k[h, j];
                    s.kv_cache[l].value_cache[pos, h, j] = s.v[h, j];
                }
            }

            // Multihead attention
            for (var h = 0; h < p.n_heads; h++) {
                // Initialize attention scores
                var att = new float[pos + 1];

                // Iterate over all timesteps
                for (var t = 0; t <= pos; t++) {
                    // Dot product between q and k
                    var score = 0.0f;
                    for (var i = 0; i < dim / p.n_heads; i++) {
                        score += s.q[h, i] * s.kv_cache[l].key_cache[t, h * p.n_kv_heads / p.n_heads, i];
                    }

                    score /= MathF.Sqrt (dim / p.n_heads);
                    att[t] = score;
                }

                // Softmax the attention scores
                int size = pos + 1;
                math.Softmax (att, size);

                // Zero out s.xb for this head
                Array.Fill (s.xb, 0, h * (dim / p.n_heads), dim / p.n_heads);

                for (var t = 0; t <= pos; t++) {
                    var a = att[t];
                    for (var i = 0; i < dim / p.n_heads; i++) {
                        s.xb[h * (dim / p.n_heads) + i] += a * s.kv_cache[l].value_cache[t, h * p.n_kv_heads / p.n_heads, i];
                    }
                }
            }

            // Final matmul
            MatMul (s.xb2, s.xb, w.wo[l], 0, dim, dim);

            // Residual connection
            for (var i = 0; i < dim; i++) {
                s.x[i] += s.xb2[i];
            }

            // FFN rmsnorm
            math.RmsNorm (s.xb, s.x, w.rms_ffn_weight[l]);

            // FFN computation
            MatMul (s.hb, s.xb, w.w1[l], 0, dim, p.hidden_dim);
            MatMul (s.hb2, s.xb, w.w3[l], 0, dim, p.hidden_dim);

            // SwiGLU activation
            for (var i = 0; i < p.hidden_dim; i++) {
                var val = s.hb[i];
                val *= 1.0f / (1.0f + MathF.Exp (-val)); // SiLU activation
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // Final FFN matmul
            MatMul (s.xb, s.hb, w.w2[l], 0, p.hidden_dim, dim);

            // Residual connection
            for (var i = 0; i < dim; i++) {
                s.x[i] += s.xb[i];
            }
        }

        // Final rmsnorm
        math.RmsNorm (s.x, s.x, w.rms_final_weight);

        // Classifier into logits
        MatMul (s.logits, s.x, w.wcls, 0, p.dim, p.vocab_size);
        return s.logits;
    }
}

public class Generator
{
    public static void Generate (Transformer transformer, ITokenizer tokenizer, Sampler sampler, string prompt, int steps) {
        if (prompt == null) {
            prompt = "";
        }

        var tokens = tokenizer.Encode (prompt, true, false);

        long start = 0;
        var next = 0;
        var pos = 0;

        while (pos < steps) {
            var logits = TransformerModel.Forward (transformer, tokens[pos], pos);

            if (pos < tokens.Length - 1) {
                next = tokens[pos + 1];
            } else {
                next = sampler.Sample (logits);
                tokens = tokens.Concat ([next]).ToArray ();
            }

            pos++;

            if (next == 1) break;

            Console.WriteLine (tokenizer.Decode (tokens));

            if (start == 0) start = DateTimeOffset.Now.ToUnixTimeMilliseconds ();
        }

        Console.WriteLine ();

        if (pos > 1) {
            var end = DateTimeOffset.Now.ToUnixTimeMilliseconds ();
            Console.Error.WriteLine ($"Achieved tok/s: {(pos - 1) / ((end - start) / 1000.0)}");
        }
    }
}

class Program
{
    static void ErrorUsage () {
        Console.WriteLine ("Usage:   run <checkpoint> [options]");
        Console.WriteLine ("Example: run model.bin -n 256 -i \"Once upon a time\"");
        Console.WriteLine ("Options:");
        Console.WriteLine ("  -t <float>  temperature in [0,inf], default 1.0");
        Console.WriteLine ("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9");
        Console.WriteLine ("  -s <int>    random seed, default time(NULL)");
        Console.WriteLine ("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len");
        Console.WriteLine ("  -i <string> input prompt");
        Console.WriteLine ("  -z <string> optional path to custom tokenizer");
        Console.WriteLine ("  -m <string> mode: generate|chat, default: generate");
        Console.WriteLine ("  -y <string> (optional) system prompt in chat mode");
        Environment.Exit (1);
    }

    public static void main (params string[] args) {
        // Default parameters
        string checkpoint_path = null;
        var tokenizer_path = "resources/tokenizer.bin";
        var temperature = 0.0f;
        var topp = 0.9f;
        var steps = 256;
        string prompt = null;
        var rng_seed = 0;
        var mode = "generate";
        string system_prompt = null;

        if (args.Length >= 1) {
            checkpoint_path = args[0];
        } else {
            ErrorUsage ();
        }

        // Argument parsing
        for (var i = 1; i < args.Length; i += 2) {
            if (i + 1 >= args.Length) ErrorUsage ();
            if (args[i][0] != '-') ErrorUsage ();
            if (args[i].Length != 2) ErrorUsage ();

            switch (args[i][1]) {
                case 't':
                    temperature = float.Parse (args[i + 1]);
                    break;
                case 'p':
                    topp = float.Parse (args[i + 1]);
                    break;
                case 's':
                    rng_seed = int.Parse (args[i + 1]);
                    break;
                case 'n':
                    steps = int.Parse (args[i + 1]);
                    break;
                case 'i':
                    prompt = args[i + 1];
                    break;
                case 'z':
                    tokenizer_path = args[i + 1];
                    break;
                case 'm':
                    mode = args[i + 1];
                    break;
                case 'y':
                    system_prompt = args[i + 1];
                    break;
                default:
                    ErrorUsage ();
                    break;
            }
        }

        if (rng_seed <= 0) rng_seed = (int)DateTime.Now.Ticks;
        if (temperature < 0.0) temperature = 0.0f;
        if (topp < 0.0 || topp > 1.0) topp = 0.9f;
        if (steps < 0) steps = 0;

        // Build the Transformer via the model .bin file
        var transformer = new Transformer ();
        TransformerModel.BuildTransformer (transformer, checkpoint_path);
        if (steps == 0 || steps > transformer.config.seq_len) {
            steps = transformer.config.seq_len;
        }

        // Build the Tokenizer via the tokenizer .bin file
        var tokenizer = Tokenizer.fromBinary (tokenizer_path, transformer.config.vocab_size);

        // Build the Sampler
        var sampler = new Sampler (transformer.config.vocab_size, temperature, topp, rng_seed);

        // Run!
        if (mode == "generate") {
            Generator.Generate (transformer, tokenizer, sampler, prompt, steps);
        } else {
            Console.Error.WriteLine ($"Unknown mode: {mode}");
            ErrorUsage ();
        }
    }
}
