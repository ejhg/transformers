using System.Text;

namespace llama.cs;

public class Config
{
    public int dim; // Transformer dimension
    public int hidden_dim; // For FFN layers
    public int n_layers; // Number of layers
    public int n_heads; // Number of query heads
    public int n_kv_heads; // Number of key/value heads
    public int vocab_size; // Vocabulary size, usually 256 (byte-level)
    public int seq_len; // Max sequence length
}

public class TransformerWeights
{
    // Token embedding table
    public float[][] token_embedding_table; // [vocab_size, dim]

    public LayerWeights[] layers;

    public class LayerWeights
    {
        // Weights for RMSNorms
        public float[] rms_att_weight; // [dim]
        public float[] rms_ffn_weight; // [dim]

        // Weights for matmuls. note dim == n_heads * head_size
        public float[,] wq; // [n_heads * head_size][dim]
        public float[,] wk; // [n_kv_heads * head_size][dim]
        public float[,] wv; // [n_kv_heads * head_size][dim]
        public float[,] wo; // [dim][n_heads * head_size]

        // Weights for FFN
        public float[,] w1; // [hidden_dim][dim]
        public float[,] w2; // [dim][hidden_dim]
        public float[,] w3; // [hidden_dim][dim]
    }

    // Final RMSNorm
    public float[] rms_final_weight; // [dim]

    // Classifier weights for the logits
    public float[][] wcls; // [vocab_size, dim]
}

public class RunState
{
    // Current wave of activations
    public float[] x; // [dim]
    public float[] xb; // [dim]
    public float[] xb2; // [dim]
    public float[] hb; // [hidden_dim]
    public float[] hb2; // [hidden_dim]
    public float[] q; // [n_heads * head_size]
    public float[] k; // [n_kv_heads * head_size]
    public float[] v; // [n_kv_heads * head_size]
    public float[] logits; // [vocab_size]

    // KV cache
    public LayerCache[] kv_cache;

    public struct LayerCache
    {
        public float[][] key_cache; // [seq_len][n_kv_heads * head_size]
        public float[][] value_cache; // [seq_len][n_kv_heads * head_size]
    }
}

public class Transformer
{
    public Config config; // Hyperparameters
    public TransformerWeights weights; // Model weights
    public RunState state; // Run state buffers
}

static class TransformerModel
{
    static RunState createRunState (Config p) {
        int head_size = p.dim / p.n_heads;
        int kv_head_size = head_size; // Assuming kv_head_size == head_size

        var kv_cache = Enumerable
            .Range (0, p.n_layers)
            .Select (_ => {
                var _cache = new RunState.LayerCache {
                    key_cache = new float[p.seq_len][],
                    value_cache = new float[p.seq_len][]
                };

                for (int t = 0; t < p.seq_len; t++) {
                    _cache.key_cache[t] = new float[p.n_kv_heads * kv_head_size];
                    _cache.value_cache[t] = new float[p.n_kv_heads * kv_head_size];
                }

                return _cache;
            }).ToArray ();

        return new RunState {
            x = new float[p.dim],
            xb = new float[p.dim],
            xb2 = new float[p.dim],
            hb = new float[p.hidden_dim],
            hb2 = new float[p.hidden_dim],
            q = new float[p.n_heads * head_size],
            k = new float[p.n_kv_heads * kv_head_size],
            v = new float[p.n_kv_heads * kv_head_size],
            logits = new float[p.vocab_size],
            kv_cache = kv_cache,
        };
    }

    static TransformerWeights MemoryMapWeights (Config p, float[] data, int shared_weights) {
        int head_size = p.dim / p.n_heads;
        int kv_head_size = head_size; // Assuming kv_head_size == head_size
        int n_layers = p.n_layers;
        long index = 0;

        TransformerWeights w = new TransformerWeights ();

        // Token embedding table
        w.token_embedding_table = new float[p.vocab_size][];
        for (int i = 0; i < p.vocab_size; i++) {
            w.token_embedding_table[i] = new float[p.dim];
            for (int j = 0; j < p.dim; j++) {
                w.token_embedding_table[i][j] = data[index++];
            }
        }

        w.layers = new TransformerWeights.LayerWeights[n_layers];

        for (int l = 0; l < n_layers; l++) {
            w.layers[l] = new TransformerWeights.LayerWeights {
                rms_att_weight = new float[p.dim],
                wq = new float[p.n_heads * head_size, p.dim],
                wk = new float[p.n_kv_heads * kv_head_size, p.dim],
                wv = new float[p.n_kv_heads * kv_head_size, p.dim],
                wo = new float[p.dim, p.n_heads * head_size],
                rms_ffn_weight = new float[p.dim],
                w1 = new float[p.hidden_dim, p.dim],
                w2 = new float[p.dim, p.hidden_dim],
                w3 = new float[p.hidden_dim, p.dim]
            };
        }

        // RMSNorm weights
        for (int l = 0; l < n_layers; l++) {
            for (int j = 0; j < p.dim; j++)
                w.layers[l].rms_att_weight[j] = data[index++];
        }

        // wq weights
        for (int l = 0; l < n_layers; l++) {
            for (int i = 0; i < p.n_heads * head_size; i++) {
                for (int j = 0; j < p.dim; j++)
                    w.layers[l].wq[i, j] = data[index++];
            }
        }

        // wk weights
        for (int l = 0; l < n_layers; l++) {
            for (int i = 0; i < p.n_kv_heads * kv_head_size; i++) {
                for (int j = 0; j < p.dim; j++)
                    w.layers[l].wk[i, j] = data[index++];
            }
        }

        // wv weights
        for (int l = 0; l < n_layers; l++) {
            for (int i = 0; i < p.n_kv_heads * kv_head_size; i++) {
                for (int j = 0; j < p.dim; j++)
                    w.layers[l].wv[i, j] = data[index++];
            }
        }

        // wo weights
        for (int l = 0; l < n_layers; l++) {
            for (int i = 0; i < p.dim; i++) {
                for (int j = 0; j < p.n_heads * head_size; j++)
                    w.layers[l].wo[i, j] = data[index++];
            }
        }

        // RMSNorm FFN weights
        for (int l = 0; l < n_layers; l++) {
            for (int j = 0; j < p.dim; j++)
                w.layers[l].rms_ffn_weight[j] = data[index++];
        }

        // w1 weights
        for (int l = 0; l < n_layers; l++) {
            for (int i = 0; i < p.hidden_dim; i++) {
                for (int j = 0; j < p.dim; j++)
                    w.layers[l].w1[i, j] = data[index++];
            }
        }

        // w2 weights
        for (int l = 0; l < n_layers; l++) {
            for (int i = 0; i < p.dim; i++) {
                for (int j = 0; j < p.hidden_dim; j++)
                    w.layers[l].w2[i, j] = data[index++];
            }
        }

        // w3 weights
        for (int l = 0; l < n_layers; l++) {
            for (int i = 0; i < p.hidden_dim; i++) {
                for (int j = 0; j < p.dim; j++)
                    w.layers[l].w3[i, j] = data[index++];
            }
        }

        // Final RMSNorm weights
        w.rms_final_weight = new float[p.dim];
        for (int i = 0; i < p.dim; i++)
            w.rms_final_weight[i] = data[index++];

        // Skip RoPE frequencies (not used in this implementation)
        index += p.seq_len * head_size / 2; // freq_cis_real
        index += p.seq_len * head_size / 2; // freq_cis_imag

        if (shared_weights != 0) {
            w.wcls = w.token_embedding_table;
        } else {
            w.wcls = new float[p.vocab_size][];
            for (int i = 0; i < p.vocab_size; i++) {
                w.wcls[i] = new float[p.dim];
                for (int j = 0; j < p.dim; j++)
                    w.wcls[i][j] = data[index++];
            }
        }

        return w;
    }

    static (Config config, TransformerWeights weights) ReadCheckpoint (string checkpoint) {
        using FileStream fs = new FileStream (checkpoint, FileMode.Open, FileAccess.Read);
        using BinaryReader br = new BinaryReader (fs);

        var config = new Config {
            dim = br.ReadInt32 (),
            hidden_dim = br.ReadInt32 (),
            n_layers = br.ReadInt32 (),
            n_heads = br.ReadInt32 (),
            n_kv_heads = br.ReadInt32 (),
            vocab_size = br.ReadInt32 (),
            seq_len = br.ReadInt32 ()
        };

        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = Math.Abs (config.vocab_size);

        var file_size = fs.Length;

        // Read data
        int dataSize = (int)((file_size - 7 * sizeof(int)) / sizeof(float)); // 7 ints in Config
        var data = new float[dataSize];
        for (int i = 0; i < dataSize; i++) {
            data[i] = br.ReadSingle ();
        }

        // Map weights
        var weights = MemoryMapWeights (config, data, shared_weights);

        return (config, weights);
    }

    public static Transformer load (string checkpoint_path) {
        var t = new Transformer ();
        (t.config, t.weights) = ReadCheckpoint (checkpoint_path);
        t.state = createRunState (t.config);
        return t;
    }

    static void RmsNorm (float[] o, float[] x, float[] weight) {
        var size = x.Length;

        // Calculate sum of squares
        float sumOfSquaresOfX = 0.0f;
        for (int j = 0; j < size; j++) {
            sumOfSquaresOfX += x[j] * x[j];
        }

        var scaleX = 1.0f / MathF.Sqrt (sumOfSquaresOfX / size + 1e-5f);

        // Normalize and scale
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (scaleX * x[j]);
        }
    }

    public static void Softmax (float[] x, int size) {
        // Find max value
        float max_val = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }

        // Exponentiate and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i] = MathF.Exp (x[i] - max_val);
            sum += x[i];
        }

        // Normalize
        for (int i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }

    /**
     * W (m,n) @ x (n,) -> xout (m,)
     */
    static void MatMul (float[] xout, float[] x, float[][] W) {
        Parallel.For (0, xout.Length, i => {
            float val = 0.0f;
            int n = x.Length;
            var row = W[i];

            for (int j = 0; j < n; j++) {
                val += row[j] * x[j];
            }

            xout[i] = val;
        });
    }

    /**
     * W (m,n) @ x (n,) -> xout (m,)
     */
    static unsafe void MatMul (float[] xout, float[] x, float[,] W) {
        Parallel.For (0, xout.Length, i => {
            int n = x.Length;

            fixed (float* pW = W) {
                float val = 0.0f;
                float* pRowW = pW + i * n;

                for (int j = 0; j < n; j++) {
                    val += pRowW[j] * x[j];
                }

                xout[i] = val;
            }
        });
    }

    public static float[] Forward (Transformer transformer, int token, int pos) {
        // Convenience variables
        Config p = transformer.config;
        TransformerWeights w = transformer.weights;
        RunState s = transformer.state;
        int head_size = p.dim / p.n_heads;
        int kv_head_size = head_size; // Assuming kv_head_size == head_size

        // Copy token embedding into x
        Array.Copy (w.token_embedding_table[token], 0, s.x, 0, p.dim);

        // Forward pass through layers
        for (int l = 0; l < p.n_layers; l++) {
            var layer = w.layers[l];
            var _cache = s.kv_cache[l];

            // Attention rmsnorm
            RmsNorm (s.xb, s.x, layer.rms_att_weight);

            // Compute q, k, v
            MatMul (s.q, s.xb, layer.wq);
            MatMul (s.k, s.xb, layer.wk);
            MatMul (s.v, s.xb, layer.wv);

            // RoPE positional encoding
            for (int i = 0; i < s.q.Length; i += 2) {
                int head_dim = i % head_size;
                float freq = 1.0f / MathF.Pow (10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = MathF.Cos (val);
                float fci = MathF.Sin (val);
                int rotn = i < p.n_kv_heads * head_size ? 2 : 1; // Rotate q and k
                for (int v = 0; v < rotn; v++) {
                    var dst = v == 0
                        ? s.q
                        : s.k;
                    float v0 = dst[i];
                    float v1 = dst[i + 1];
                    dst[i] = v0 * fcr - v1 * fci;
                    dst[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // Store k and v in cache
            Array.Copy (s.k, _cache.key_cache[pos], s.k.Length);
            Array.Copy (s.v, _cache.value_cache[pos], s.v.Length);

            // Multihead attention
            for (int h = 0; h < p.n_heads; h++) {
                int q_offset = h * head_size;
                int kv_h = h % p.n_kv_heads;
                int k_offset = kv_h * kv_head_size;

                // Initialize attention scores
                float[] att = new float[pos + 1];

                // Iterate over all timesteps
                for (int t = 0; t <= pos; t++) {
                    var _key = _cache.key_cache[t];

                    // Dot product between q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += s.q[q_offset + i] * _key[k_offset + i];
                    }

                    score /= MathF.Sqrt (head_size);
                    att[t] = score;
                }

                // Softmax the attention scores
                Softmax (att, pos + 1);

                // Zero out s.xb for this head
                Array.Fill (s.xb, 0, q_offset, head_size);

                for (int t = 0; t <= pos; t++) {
                    var _value = _cache.value_cache[t];

                    float a = att[t];
                    for (int i = 0; i < head_size; i++) {
                        s.xb[q_offset + i] += a * _value[k_offset + i];
                    }
                }
            }

            // Final matmul
            MatMul (s.xb2, s.xb, layer.wo);

            // Residual connection
            for (int i = 0; i < p.dim; i++) {
                s.x[i] += s.xb2[i];
            }

            // FFN rmsnorm
            RmsNorm (s.xb, s.x, layer.rms_ffn_weight);

            // FFN computation
            MatMul (s.hb, s.xb, layer.w1);
            MatMul (s.hb2, s.xb, layer.w3);

            // SwiGLU activation
            for (int i = 0; i < p.hidden_dim; i++) {
                float val = s.hb[i];
                val *= 1.0f / (1.0f + MathF.Exp (-val)); // SiLU activation
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // Final FFN matmul
            MatMul (s.xb, s.hb, layer.w2);

            // Residual connection
            for (int i = 0; i < p.dim; i++) {
                s.x[i] += s.xb[i];
            }
        }

        // Final rmsnorm
        RmsNorm (s.x, s.x, w.rms_final_weight);

        // Classifier into logits
        MatMul (s.logits, s.x, w.wcls);

        return s.logits;
    }
}

/**
 * The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
 */
public class Tokenizer
{
    public string[] vocab;
    public float[] vocab_scores;
    public Dictionary<string, int> vocab_lookup;
    public string[] byte_pieces = new string[512];

    public void BuildTokenizer (string tokenizer_path, int vocab_size) {
        vocab = new string[vocab_size];
        vocab_scores = new float[vocab_size];
        vocab_lookup = null;

        // Initialize byte pieces
        for (int i = 0; i < 256; i++) {
            byte_pieces[i * 2] = ((char)i).ToString ();
            byte_pieces[i * 2 + 1] = '\0'.ToString ();
        }

        using FileStream fs = new FileStream (tokenizer_path, FileMode.Open, FileAccess.Read);
        using BinaryReader br = new BinaryReader (fs);

        // variable not used, but must be read from structure
        var max_token_length = br.ReadUInt32 ();

        int len;
        for (int i = 0; i < vocab_size; i++) {
            vocab_scores[i] = br.ReadSingle ();
            len = br.ReadInt32 ();
            byte[] strBytes = br.ReadBytes (len);
            vocab[i] = Encoding.UTF8.GetString (strBytes);
        }
    }

    public string Decode (int prev_token, int token) {
        string piece = vocab[token];
        if (prev_token == 1 && piece.StartsWith (" ")) {
            piece = piece.Substring (1);
        }

        if (piece.StartsWith ("<0x") && piece.EndsWith (">")) {
            string hex = piece.Substring (3, piece.Length - 4);
            if (byte.TryParse (hex, System.Globalization.NumberStyles.HexNumber, null, out byte byte_val)) {
                piece = ((char)byte_val).ToString ();
            }
        }

        return piece;
    }

    public void SafePrint (string piece) {
        if (string.IsNullOrEmpty (piece))
            return;

        if (piece.Length == 1) {
            char c = piece[0];
            if (!char.IsControl (c) || char.IsWhiteSpace (c)) {
                Console.Write (piece);
            }
        } else {
            Console.Write (piece);
        }
    }

    int StrLookup (string str) {
        if (vocab_lookup == null) {
            vocab_lookup = vocab
                .Select ((str, index) => (str, index))
                .ToDictionary ();
        }

        return vocab_lookup.GetValueOrDefault (str, -1);
    }

    public void Encode (string text, bool bos, bool eos, List<int> tokens) {
        if (text == null) {
            Console.Error.WriteLine ("Cannot encode null text");
            Environment.Exit (1);
        }

        // Start encoding
        if (bos)
            tokens.Add (1); // BOS token

        if (text.Length > 0) {
            int dummy_prefix = StrLookup (" ");
            tokens.Add (dummy_prefix);
        }

        int str_len = 0;
        StringBuilder str_buffer = new StringBuilder ();

        for (int i = 0; i < text.Length; i++) {
            char c = text[i];

            if ((c & 0xC0) != 0x80) {
                str_len = 0;
            }

            str_buffer.Append (c);
            str_len++;

            if (i + 1 < text.Length && (text[i + 1] & 0xC0) == 0x80 && str_len < 4) {
                continue;
            }

            string str = str_buffer.ToString ();
            int id = StrLookup (str);

            if (id != -1) {
                tokens.Add (id);
            } else {
                foreach (char ch in str) {
                    tokens.Add ((byte)ch + 3);
                }
            }

            str_buffer.Clear ();
            str_len = 0;
        }

        // Merge pairs
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < tokens.Count - 1; i++) {
                string merged = vocab[tokens[i]] + vocab[tokens[i + 1]];
                int id = StrLookup (merged);
                if (id != -1 && vocab_scores[id] > best_score) {
                    best_score = vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1)
                break;

            tokens[best_idx] = best_id;
            tokens.RemoveAt (best_idx + 1);
        }

        if (eos)
            tokens.Add (2); // EOS token
    }
}

public class ProbIndex : IComparable<ProbIndex>
{
    public float prob;
    public int index;

    public int CompareTo (ProbIndex other) {
        return other.prob.CompareTo (prob);
    }
}

/**
 * The Sampler, which takes logits and returns a sampled token
 */
public class Sampler
{
    public ProbIndex[] probindex;
    public float temperature;
    public float topp;
    public Random rng;

    public Sampler (int vocab_size, float temperature, float topp, int rng_seed) {
        this.temperature = temperature;
        this.topp = topp;
        rng = new Random (rng_seed);
        probindex = new ProbIndex[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            probindex[i] = new ProbIndex ();
        }
    }

    public int SampleArgMax (float[] probabilities) {
        int max_i = 0;
        float max_p = probabilities[0];
        for (int i = 1; i < probabilities.Length; i++) {
            if (probabilities[i] > max_p) {
                max_p = probabilities[i];
                max_i = i;
            }
        }

        return max_i;
    }

    public int SampleMult (float[] probabilities) {
        float coin = (float)rng.NextDouble ();
        float cdf = 0.0f;
        for (int i = 0; i < probabilities.Length; i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }

        return probabilities.Length - 1;
    }

    public int SampleTopP (float[] probabilities) {
        float coin = (float)rng.NextDouble ();
        int n0 = 0;
        float cutoff = (1.0f - topp) / (probabilities.Length - 1);

        for (int i = 0; i < probabilities.Length; i++) {
            if (probabilities[i] >= cutoff) {
                probindex[n0].index = i;
                probindex[n0].prob = probabilities[i];
                n0++;
            }
        }

        Array.Sort (probindex, 0, n0);

        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1;
        for (int i = 0; i < n0; i++) {
            cumulative_prob += probindex[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break;
            }
        }

        float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += probindex[i].prob;
            if (r < cdf) {
                return probindex[i].index;
            }
        }

        return probindex[last_idx].index;
    }

    public int Sample (float[] logits) {
        if (temperature == 0.0f) {
            return SampleArgMax (logits);
        } else {
            for (int i = 0; i < logits.Length; i++) {
                logits[i] /= temperature;
            }

            TransformerModel.Softmax (logits, logits.Length);

            if (topp <= 0 || topp >= 1) {
                return SampleMult (logits);
            } else {
                return SampleTopP (logits);
            }
        }
    }
}

public class Generator
{
    public static void Generate (Transformer transformer, Tokenizer tokenizer, Sampler sampler, string prompt, int steps) {
        if (prompt == null) {
            prompt = "";
        }

        List<int> tokens = new List<int> ();
        tokenizer.Encode (prompt, true, false, tokens);

        if (tokens.Count < 1) {
            Console.Error.WriteLine ("Expected at least 1 prompt token");
            Environment.Exit (1);
        }

        long start = 0;
        int next = 0;
        int token = tokens[0];
        int pos = 0;

        while (pos < steps) {
            float[] logits = TransformerModel.Forward (transformer, token, pos);

            if (pos < tokens.Count - 1) {
                next = tokens[pos + 1];
            } else {
                next = sampler.Sample (logits);
            }

            pos++;

            if (next == 1) break;

            string piece = tokenizer.Decode (token, next);
            tokenizer.SafePrint (piece);
            token = next;

            if (start == 0) start = TimeInMs ();
        }

        Console.WriteLine ();

        if (pos > 1) {
            long end = TimeInMs ();
            Console.Error.WriteLine ($"Achieved tok/s: {(pos - 1) / ((end - start) / 1000.0)}");
        }
    }

    static long TimeInMs () {
        return DateTimeOffset.Now.ToUnixTimeMilliseconds ();
    }

    public static void Chat (Transformer transformer, Tokenizer tokenizer, Sampler sampler, string cli_user_prompt, string cli_system_prompt,
        int steps) {
        // Buffers for prompts
        string system_prompt = "";
        string user_prompt = "";
        string rendered_prompt = "";
        List<int> prompt_tokens = new List<int> ();
        int user_idx = 0;

        bool user_turn = true; // User starts
        int next = 0; // Next token
        int token = 0; // Current token
        int pos = 0; // Position in the sequence

        while (pos < steps) {
            // User's turn
            if (user_turn) {
                // Get system prompt at position 0
                if (pos == 0) {
                    if (cli_system_prompt == null) {
                        // System prompt not provided, read from stdin
                        ReadStdin ("Enter system prompt (optional): ", out system_prompt);
                    } else {
                        system_prompt = cli_system_prompt;
                    }
                }

                // Get user prompt
                if (pos == 0 && cli_user_prompt != null) {
                    user_prompt = cli_user_prompt;
                } else {
                    ReadStdin ("User: ", out user_prompt);
                }

                // Render prompts into the Llama 2 Chat schema
                if (pos == 0 && !string.IsNullOrEmpty (system_prompt)) {
                    rendered_prompt = $"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]";
                } else {
                    rendered_prompt = $"[INST] {user_prompt} [/INST]";
                }

                // Encode the rendered prompt into tokens
                prompt_tokens.Clear ();
                tokenizer.Encode (rendered_prompt, true, false, prompt_tokens);
                user_idx = 0; // Reset user index
                user_turn = false;
                Console.Write ("Assistant: ");
            }

            // Determine the token to pass into the transformer next
            if (user_idx < prompt_tokens.Count) {
                token = prompt_tokens[user_idx++];
            } else {
                token = next;
            }

            // EOS token ends the Assistant turn
            if (token == 2) {
                user_turn = true;
            }

            // Forward the transformer to get logits
            float[] logits = TransformerModel.Forward (transformer, token, pos);
            next = sampler.Sample (logits);
            pos++;

            if (user_idx >= prompt_tokens.Count && next != 2) {
                // Assistant is responding
                string piece = tokenizer.Decode (token, next);
                tokenizer.SafePrint (piece);
            }

            if (next == 2) {
                Console.WriteLine ();
            }
        }

        Console.WriteLine ();
    }

    static void ReadStdin (string guide, out string buffer) {
        Console.Write (guide);
        buffer = Console.ReadLine ();
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
        string tokenizer_path = "resources/tokenizer.bin";
        float temperature = 1.0f;
        float topp = 0.9f;
        int steps = 256;
        string prompt = null;
        int rng_seed = 0;
        string mode = "generate";
        string system_prompt = null;

        if (args.Length >= 1) {
            checkpoint_path = args[0];
        } else {
            ErrorUsage ();
        }

        // Argument parsing
        for (int i = 1; i < args.Length; i += 2) {
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
        var transformer = TransformerModel.load (checkpoint_path);

        if (steps == 0 || steps > transformer.config.seq_len) {
            steps = transformer.config.seq_len;
        }

        // Build the Tokenizer via the tokenizer .bin file
        Tokenizer tokenizer = new Tokenizer ();
        tokenizer.BuildTokenizer (tokenizer_path, transformer.config.vocab_size);

        // Build the Sampler
        Sampler sampler = new Sampler (transformer.config.vocab_size, temperature, topp, rng_seed);

        // Run!
        if (mode == "generate") {
            Generator.Generate (transformer, tokenizer, sampler, prompt, steps);
        } else if (mode == "chat") {
            Generator.Chat (transformer, tokenizer, sampler, prompt, system_prompt, steps);
        } else {
            Console.Error.WriteLine ($"Unknown mode: {mode}");
            ErrorUsage ();
        }
    }
}
