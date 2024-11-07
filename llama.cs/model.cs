using System.Text;

namespace llama.cs
{
    // ----------------------------------------------------------------------------
    // Transformer model

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
        public Config config; // Hyperparameters
        public TransformerWeights weights; // Model weights
        public RunState state; // Run state buffers
    }

    public static class TransformerModel
    {
        public static void MallocRunState (RunState s, Config p) {
            int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
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

        public static void MemoryMapWeights (TransformerWeights w, Config p, float[] data, int shared_weights) {
            int head_size = p.dim / p.n_heads;
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

        public static void ReadCheckpoint (string checkpoint, Config config, TransformerWeights weights, out float[] data, out long file_size) {
            using (FileStream fs = new FileStream (checkpoint, FileMode.Open, FileAccess.Read)) {
                using (BinaryReader br = new BinaryReader (fs)) {
                    // Read config
                    config.dim = br.ReadInt32 ();
                    config.hidden_dim = br.ReadInt32 ();
                    config.n_layers = br.ReadInt32 ();
                    config.n_heads = br.ReadInt32 ();
                    config.n_kv_heads = br.ReadInt32 ();
                    config.vocab_size = br.ReadInt32 ();
                    config.seq_len = br.ReadInt32 ();

                    int shared_weights = config.vocab_size > 0 ? 1 : 0;
                    config.vocab_size = Math.Abs (config.vocab_size);

                    file_size = fs.Length;

                    // Read data
                    int dataSize = (int)((file_size - 7 * sizeof(int)) / sizeof(float)); // 7 ints in Config
                    data = new float[dataSize];
                    for (int i = 0; i < dataSize; i++) {
                        data[i] = br.ReadSingle ();
                    }

                    // Map weights
                    MemoryMapWeights (weights, config, data, shared_weights);
                }
            }
        }

        public static void BuildTransformer (Transformer t, string checkpoint_path) {
            t.config = new Config ();
            t.weights = new TransformerWeights ();
            t.state = new RunState ();

            ReadCheckpoint (checkpoint_path, t.config, t.weights, out _, out _);
            MallocRunState (t.state, t.config);
        }

        public static void RmsNorm (float[] o, float[] x, float[] weight, int weightOffset, int size) {
            // Calculate sum of squares
            float ss = 0.0f;
            for (int j = 0; j < size; j++) {
                ss += x[j] * x[j];
            }

            ss /= size;
            ss += 1e-5f;
            ss = 1.0f / (float)Math.Sqrt (ss);

            // Normalize and scale
            for (int j = 0; j < size; j++) {
                o[j] = weight[weightOffset + j] * (ss * x[j]);
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
                x[i] = (float)Math.Exp (x[i] - max_val);
                sum += x[i];
            }

            // Normalize
            for (int i = 0; i < size; i++) {
                x[i] /= sum;
            }
        }

        public static void MatMul (float[] xout, float[] x, float[] w, int wOffset, int n, int d) {
            // W (d,n) @ x (n,) -> xout (d,)
            Parallel.For (0, d, i => {
                float val = 0.0f;
                for (int j = 0; j < n; j++) {
                    val += w[wOffset + i * n + j] * x[j];
                }

                xout[i] = val;
            });
        }

        public static float[] Forward (Transformer transformer, int token, int pos) {
            // Convenience variables
            Config p = transformer.config;
            TransformerWeights w = transformer.weights;
            RunState s = transformer.state;
            float[] x = s.x;
            int dim = p.dim;
            int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
            int kv_mul = p.n_heads / p.n_kv_heads;
            int hidden_dim = p.hidden_dim;
            int head_size = dim / p.n_heads;

            // Copy token embedding into x
            Array.Copy (w.token_embedding_table, token * dim, x, 0, dim);

            // Forward pass through layers
            for (int l = 0; l < p.n_layers; l++) {
                // Attention rmsnorm
                RmsNorm (s.xb, x, w.rms_att_weight, l * dim, dim);

                // Key and value cache offsets
                int loff = l * p.seq_len * kv_dim;
                int kOffset = loff + pos * kv_dim;
                int vOffset = loff + pos * kv_dim;

                // Compute q, k, v
                MatMul (s.q, s.xb, w.wq, l * dim * p.n_heads * head_size, dim, p.n_heads * head_size);
                MatMul (s.k, s.xb, w.wk, l * dim * p.n_kv_heads * head_size, dim, p.n_kv_heads * head_size);
                MatMul (s.v, s.xb, w.wv, l * dim * p.n_kv_heads * head_size, dim, p.n_kv_heads * head_size);

                // RoPE positional encoding
                for (int i = 0; i < p.n_heads * head_size; i += 2) {
                    int head_dim = i % head_size;
                    float freq = 1.0f / (float)Math.Pow (10000.0, head_dim / (float)head_size);
                    float val = pos * freq;
                    float fcr = (float)Math.Cos (val);
                    float fci = (float)Math.Sin (val);
                    int rotn = i < p.n_kv_heads * head_size ? 2 : 1; // Rotate q and k
                    for (int v = 0; v < rotn; v++) {
                        float[] vec = v == 0 ? s.q : s.k;
                        float v0 = vec[i];
                        float v1 = vec[i + 1];
                        vec[i] = v0 * fcr - v1 * fci;
                        vec[i + 1] = v0 * fci + v1 * fcr;
                    }
                }

                // Store k and v in cache
                Array.Copy (s.k, 0, s.key_cache, kOffset, kv_dim);
                Array.Copy (s.v, 0, s.value_cache, vOffset, kv_dim);

                // Multihead attention
                for (int h = 0; h < p.n_heads; h++) {
                    int headOffset = h * head_size;

                    // Indices for q
                    int qStart = headOffset;

                    // Initialize attention scores
                    float[] att = new float[pos + 1];

                    // Iterate over all timesteps
                    for (int t = 0; t <= pos; t++) {
                        int kHeadOffset = loff + t * kv_dim + (h / kv_mul) * head_size;

                        // Dot product between q and k
                        float score = 0.0f;
                        for (int i = 0; i < head_size; i++) {
                            score += s.q[qStart + i] * s.key_cache[kHeadOffset + i];
                        }

                        score /= (float)Math.Sqrt (head_size);
                        att[t] = score;
                    }

                    // Softmax the attention scores
                    Softmax (att, pos + 1);

                    // Zero out s.xb for this head
                    for (int i = 0; i < head_size; i++) {
                        s.xb[headOffset + i] = 0.0f;
                    }

                    for (int t = 0; t <= pos; t++) {
                        int vHeadOffset = loff + t * kv_dim + (h / kv_mul) * head_size;

                        float a = att[t];
                        for (int i = 0; i < head_size; i++) {
                            s.xb[headOffset + i] += a * s.value_cache[vHeadOffset + i];
                        }
                    }
                }

                // Final matmul
                MatMul (s.xb2, s.xb, w.wo, l * p.n_heads * head_size * dim, p.n_heads * head_size, dim);

                // Residual connection
                for (int i = 0; i < dim; i++) {
                    x[i] += s.xb2[i];
                }

                // FFN rmsnorm
                RmsNorm (s.xb, x, w.rms_ffn_weight, l * dim, dim);

                // FFN computation
                MatMul (s.hb, s.xb, w.w1, l * p.hidden_dim * dim, dim, p.hidden_dim);
                MatMul (s.hb2, s.xb, w.w3, l * p.hidden_dim * dim, dim, p.hidden_dim);

                // SwiGLU activation
                for (int i = 0; i < p.hidden_dim; i++) {
                    float val = s.hb[i];
                    val *= 1.0f / (1.0f + (float)Math.Exp (-val)); // SiLU activation
                    val *= s.hb2[i];
                    s.hb[i] = val;
                }

                // Final FFN matmul
                MatMul (s.xb, s.hb, w.w2, l * dim * p.hidden_dim, p.hidden_dim, dim);

                // Residual connection
                for (int i = 0; i < dim; i++) {
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

    // ----------------------------------------------------------------------------
    // The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

    public class TokenIndex : IComparable<TokenIndex>
    {
        public string str;
        public int id;

        public int CompareTo (TokenIndex other) {
            return string.Compare (str, other.str, StringComparison.Ordinal);
        }
    }

    public class Tokenizer
    {
        public string[] vocab;
        public float[] vocab_scores;
        public TokenIndex[] sorted_vocab;
        public int vocab_size;
        public uint max_token_length;
        public string[] byte_pieces = new string[512];

        public void BuildTokenizer (string tokenizer_path, int vocab_size) {
            this.vocab_size = vocab_size;
            vocab = new string[vocab_size];
            vocab_scores = new float[vocab_size];
            sorted_vocab = null;

            // Initialize byte pieces
            for (int i = 0; i < 256; i++) {
                byte_pieces[i * 2] = ((char)i).ToString ();
                byte_pieces[i * 2 + 1] = '\0'.ToString ();
            }

            using (FileStream fs = new FileStream (tokenizer_path, FileMode.Open, FileAccess.Read))
            using (BinaryReader br = new BinaryReader (fs)) {
                max_token_length = br.ReadUInt32 ();
                int len;
                for (int i = 0; i < vocab_size; i++) {
                    vocab_scores[i] = br.ReadSingle ();
                    len = br.ReadInt32 ();
                    byte[] strBytes = br.ReadBytes (len);
                    vocab[i] = Encoding.UTF8.GetString (strBytes);
                }
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
                if (!char.IsControl (c)) {
                    Console.Write (piece);
                }
            } else {
                Console.Write (piece);
            }
        }

        public int StrLookup (string str) {
            if (sorted_vocab == null) {
                sorted_vocab = new TokenIndex[vocab_size];
                for (int i = 0; i < vocab_size; i++) {
                    sorted_vocab[i] = new TokenIndex {
                        str = vocab[i],
                        id = i
                    };
                }

                Array.Sort (sorted_vocab);
            }

            int index = Array.BinarySearch (sorted_vocab, new TokenIndex { str = str });
            if (index >= 0) {
                return sorted_vocab[index].id;
            }

            return -1;
        }

        public void Encode (string text, bool bos, bool eos, List<int> tokens) {
            if (text == null) {
                Console.Error.WriteLine ("Cannot encode null text");
                Environment.Exit (1);
            }

            if (sorted_vocab == null) {
                sorted_vocab = new TokenIndex[vocab_size];
                for (int i = 0; i < vocab_size; i++) {
                    sorted_vocab[i] = new TokenIndex {
                        str = vocab[i],
                        id = i
                    };
                }

                Array.Sort (sorted_vocab);
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

    // ----------------------------------------------------------------------------
    // The Sampler, which takes logits and returns a sampled token

    public class ProbIndex : IComparable<ProbIndex>
    {
        public float prob;
        public int index;

        public int CompareTo (ProbIndex other) {
            return other.prob.CompareTo (prob);
        }
    }

    public class Sampler
    {
        public int vocab_size;
        public ProbIndex[] probindex;
        public float temperature;
        public float topp;
        public Random rng;

        public Sampler (int vocab_size, float temperature, float topp, int rng_seed) {
            this.vocab_size = vocab_size;
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

    // ----------------------------------------------------------------------------
    // Utilities: time measurement

    public static class Utils
    {
        public static long TimeInMs () {
            return DateTimeOffset.Now.ToUnixTimeMilliseconds ();
        }

        public static void ReadStdin (string guide, out string buffer) {
            Console.Write (guide);
            buffer = Console.ReadLine ();
        }
    }

    // ----------------------------------------------------------------------------
    // Generation loop

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

                if (start == 0) start = Utils.TimeInMs ();
            }

            Console.WriteLine ();

            if (pos > 1) {
                long end = Utils.TimeInMs ();
                Console.Error.WriteLine ($"Achieved tok/s: {(pos - 1) / ((end - start) / 1000.0)}");
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Main CLI program

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
            Transformer transformer = new Transformer ();
            TransformerModel.BuildTransformer (transformer, checkpoint_path);
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
            } else {
                Console.Error.WriteLine ($"Unknown mode: {mode}");
                ErrorUsage ();
            }
        }
    }
}
