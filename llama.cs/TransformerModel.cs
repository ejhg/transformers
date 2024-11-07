namespace llama.cs;

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
