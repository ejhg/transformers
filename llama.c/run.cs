using llama.cs;
using llama.torchsharp;

namespace llama.c;

public class TransformerWeights
{
    // Token embedding table
    public float[] token_embedding_table; // (vocab_size * dim)

    // Weights for RMSNorms
    public float[] rms_att_weight; // (n_layers * dim)
    public float[] rms_ffn_weight; // (n_layers * dim)

    // Weights for matmuls. note dim == n_heads * head_size
    public float[] wq; // (n_layers * dim * n_heads * head_size)
    public float[] wk; // (n_layers * dim * n_kv_heads * head_size)
    public float[] wv; // (n_layers * dim * n_kv_heads * head_size)
    public float[] wo; // (n_layers * n_heads * head_size * dim)

    // Weights for FFN
    public float[] w1; // (n_layers * hidden_dim * dim)
    public float[] w2; // (n_layers * dim * hidden_dim)
    public float[] w3; // (n_layers * hidden_dim * dim)

    // Final RMSNorm
    public float[] rms_final_weight; // (dim)

    // Classifier weights for the logits
    public float[] wcls;
}

public class RunState
{
    // Current wave of activations
    public float[] x; // (dim)
    public float[] xb; // (dim)
    public float[] xb2; // (dim)
    public float[] hb; // (hidden_dim)
    public float[] hb2; // (hidden_dim)
    public float[] q; // (dim)
    public float[] k; // (dim)
    public float[] v; // (dim)
    public float[] logits; // Output logits

    // KV cache
    public float[] key_cache; // (n_layers * seq_len * kv_dim)
    public float[] value_cache; // (n_layers * seq_len * kv_dim)
}

public class Transformer
{
    public config config; // Hyperparameters
    public TransformerWeights weights; // Model weights
    public RunState state; // Run state buffers
}

static class TransformerModel
{
    static void MallocRunState (RunState s, config p) {
        var kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        s.x = new float[p.dim];
        s.xb = new float[p.dim];
        s.xb2 = new float[p.dim];
        s.hb = new float[p.hidden_dim];
        s.hb2 = new float[p.hidden_dim];
        s.q = new float[p.dim];
        s.k = new float[p.dim];
        s.v = new float[p.dim];
        s.logits = new float[p.vocab_size];
        s.key_cache = new float[p.n_layers * p.seq_len * kv_dim];
        s.value_cache = new float[p.n_layers * p.seq_len * kv_dim];
    }

    static void MemoryMapWeights (TransformerWeights w, config p, float[] data, int shared_weights) {
        var head_size = p.dim / p.n_heads;
        long n_layers = p.n_layers;
        long index = 0;

        w.token_embedding_table = new float[p.vocab_size * p.dim];
        Array.Copy (data, index, w.token_embedding_table, 0, w.token_embedding_table.Length);
        index += w.token_embedding_table.Length;

        w.rms_att_weight = new float[n_layers * p.dim];
        Array.Copy (data, index, w.rms_att_weight, 0, w.rms_att_weight.Length);
        index += w.rms_att_weight.Length;

        w.wq = new float[n_layers * p.dim * p.n_heads * head_size];
        Array.Copy (data, index, w.wq, 0, w.wq.Length);
        index += w.wq.Length;

        w.wk = new float[n_layers * p.dim * p.n_kv_heads * head_size];
        Array.Copy (data, index, w.wk, 0, w.wk.Length);
        index += w.wk.Length;

        w.wv = new float[n_layers * p.dim * p.n_kv_heads * head_size];
        Array.Copy (data, index, w.wv, 0, w.wv.Length);
        index += w.wv.Length;

        w.wo = new float[n_layers * p.n_heads * head_size * p.dim];
        Array.Copy (data, index, w.wo, 0, w.wo.Length);
        index += w.wo.Length;

        w.rms_ffn_weight = new float[n_layers * p.dim];
        Array.Copy (data, index, w.rms_ffn_weight, 0, w.rms_ffn_weight.Length);
        index += w.rms_ffn_weight.Length;

        w.w1 = new float[n_layers * p.hidden_dim * p.dim];
        Array.Copy (data, index, w.w1, 0, w.w1.Length);
        index += w.w1.Length;

        w.w2 = new float[n_layers * p.dim * p.hidden_dim];
        Array.Copy (data, index, w.w2, 0, w.w2.Length);
        index += w.w2.Length;

        w.w3 = new float[n_layers * p.hidden_dim * p.dim];
        Array.Copy (data, index, w.w3, 0, w.w3.Length);
        index += w.w3.Length;

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
            index += w.wcls.Length;
        }
    }

    static void ReadCheckpoint (string checkpoint, config config, TransformerWeights weights, out float[] data, out long file_size) {
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

        file_size = fs.Length;

        // Read data
        var dataSize = (int)((file_size - 7 * sizeof(int)) / sizeof(float)); // 7 ints in Config
        data = new float[dataSize];
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

        ReadCheckpoint (checkpoint_path, t.config, t.weights, out _, out _);
        MallocRunState (t.state, t.config);
    }

    static void RmsNorm (float[] o, float[] x, float[] weight, int weightOffset, int size) {
        // Calculate sum of squares
        var ss = 0.0f;
        for (var j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }

        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / MathF.Sqrt (ss);

        // Normalize and scale
        for (var j = 0; j < size; j++) {
            o[j] = weight[weightOffset + j] * (ss * x[j]);
        }
    }

    public static void Softmax (float[] x, int size) {
        // Find max value
        var max_val = x[0];
        for (var i = 1; i < size; i++) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }

        // Exponentiate and sum
        var sum = 0.0f;
        for (var i = 0; i < size; i++) {
            x[i] = MathF.Exp (x[i] - max_val);
            sum += x[i];
        }

        // Normalize
        for (var i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }

    static void MatMul (float[] xout, float[] x, float[] w, int wOffset, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)

        Parallel.For (0, d, i => {
            var val = 0.0f;
            var ptr = wOffset + i * n;

            for (var j = 0; j < n; j++) {
                val += w[ptr + j] * x[j];
            }

            xout[i] = val;
        });
    }

    public static float[] Forward (Transformer transformer, int token, int pos) {
        // Convenience variables
        var p = transformer.config;
        var w = transformer.weights;
        var s = transformer.state;
        var x = s.x;
        var dim = p.dim;
        var kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        var kv_mul = p.n_heads / p.n_kv_heads;
        var head_size = dim / p.n_heads;

        // Copy token embedding into x
        Array.Copy (w.token_embedding_table, token * dim, x, 0, dim);

        // Forward pass through layers
        for (var l = 0; l < p.n_layers; l++) {
            // Attention rmsnorm
            RmsNorm (s.xb, x, w.rms_att_weight, l * dim, dim);

            // Key and value cache offsets
            var loff = l * p.seq_len * kv_dim;
            var kOffset = loff + pos * kv_dim;
            var vOffset = loff + pos * kv_dim;

            // Compute q, k, v
            MatMul (s.q, s.xb, w.wq, l * dim * p.n_heads * head_size, dim, p.n_heads * head_size);
            MatMul (s.k, s.xb, w.wk, l * dim * p.n_kv_heads * head_size, dim, p.n_kv_heads * head_size);
            MatMul (s.v, s.xb, w.wv, l * dim * p.n_kv_heads * head_size, dim, p.n_kv_heads * head_size);

            // RoPE positional encoding
            for (var i = 0; i < p.n_heads * head_size; i += 2) {
                var head_dim = i % head_size;
                var freq = 1.0f / MathF.Pow (10000.0f, head_dim / (float)head_size);
                var val = pos * freq;
                var fcr = MathF.Cos (val);
                var fci = MathF.Sin (val);
                var rotn = i < p.n_kv_heads * head_size ? 2 : 1; // Rotate q and k
                for (var v = 0; v < rotn; v++) {
                    var dst = v == 0
                        ? s.q
                        : s.k;
                    var v0 = dst[i];
                    var v1 = dst[i + 1];
                    dst[i] = v0 * fcr - v1 * fci;
                    dst[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // Store k and v in cache
            Array.Copy (s.k, 0, s.key_cache, kOffset, kv_dim);
            Array.Copy (s.v, 0, s.value_cache, vOffset, kv_dim);

            // Multihead attention
            for (var h = 0; h < p.n_heads; h++) {
                var headOffset = h * head_size;

                // Indices for q
                var qStart = headOffset;

                // Initialize attention scores
                var att = new float[pos + 1];

                // Iterate over all timesteps
                for (var t = 0; t <= pos; t++) {
                    var kHeadOffset = loff + t * kv_dim + (h / kv_mul) * head_size;

                    // Dot product between q and k
                    var score = 0.0f;
                    for (var i = 0; i < head_size; i++) {
                        score += s.q[qStart + i] * s.key_cache[kHeadOffset + i];
                    }

                    score /= MathF.Sqrt (head_size);
                    att[t] = score;
                }

                // Softmax the attention scores
                Softmax (att, pos + 1);

                // Zero out s.xb for this head
                Array.Fill (s.xb, 0, headOffset, head_size);

                for (var t = 0; t <= pos; t++) {
                    var vHeadOffset = loff + t * kv_dim + (h / kv_mul) * head_size;

                    var a = att[t];
                    for (var i = 0; i < head_size; i++) {
                        s.xb[headOffset + i] += a * s.value_cache[vHeadOffset + i];
                    }
                }
            }

            // Final matmul
            MatMul (s.xb2, s.xb, w.wo, l * p.n_heads * head_size * dim, p.n_heads * head_size, dim);

            // Residual connection
            for (var i = 0; i < dim; i++) {
                x[i] += s.xb2[i];
            }

            // FFN rmsnorm
            RmsNorm (s.xb, x, w.rms_ffn_weight, l * dim, dim);

            // FFN computation
            MatMul (s.hb, s.xb, w.w1, l * p.hidden_dim * dim, dim, p.hidden_dim);
            MatMul (s.hb2, s.xb, w.w3, l * p.hidden_dim * dim, dim, p.hidden_dim);

            // SwiGLU activation
            for (var i = 0; i < p.hidden_dim; i++) {
                var val = s.hb[i];
                val *= 1.0f / (1.0f + MathF.Exp (-val)); // SiLU activation
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // Final FFN matmul
            MatMul (s.xb, s.hb, w.w2, l * dim * p.hidden_dim, p.hidden_dim, dim);

            // Residual connection
            for (var i = 0; i < dim; i++) {
                x[i] += s.xb[i];
            }
        }

        // Final rmsnorm
        RmsNorm (x, x, w.rms_final_weight, 0, dim);

        // Classifier into logits
        MatMul (s.logits, x, w.wcls, 0, p.dim, p.vocab_size);
        return s.logits;
    }
}

public class TokenIndex : IComparable<TokenIndex>
{
    public string str;
    public int id;

    public int CompareTo (TokenIndex other) {
        return string.Compare (str, other.str, StringComparison.Ordinal);
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

            Console.WriteLine(tokenizer.Decode (tokens));

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
        var temperature = 1.0f;
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
