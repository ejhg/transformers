namespace llama.cs;

public class ModelLoader
{
    public static model load (string checkpoint_path) {
        var t = new model ();
        (t.config, t.weights) = ReadCheckpoint (checkpoint_path);
        t.state = createRunState (t.config);
        return t;
    }

    static run_state createRunState (config p) {
        int head_size = p.dim / p.n_heads;
        int kv_head_size = head_size; // Assuming kv_head_size == head_size

        var kv_cache = Enumerable
            .Range (0, p.n_layers)
            .Select (_ => {
                var _cache = new run_state.LayerCache {
                    key_cache = new float[p.seq_len][],
                    value_cache = new float[p.seq_len][]
                };

                for (int t = 0; t < p.seq_len; t++) {
                    _cache.key_cache[t] = new float[p.n_kv_heads * kv_head_size];
                    _cache.value_cache[t] = new float[p.n_kv_heads * kv_head_size];
                }

                return _cache;
            }).ToArray ();

        return new run_state {
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

    static weights MemoryMapWeights (config p, float[] data, int shared_weights) {
        int head_size = p.dim / p.n_heads;
        int kv_head_size = head_size; // Assuming kv_head_size == head_size
        int n_layers = p.n_layers;
        long index = 0;

        weights w = new weights ();

        // Token embedding table
        w.token_embedding_table = new float[p.vocab_size][];
        for (int i = 0; i < p.vocab_size; i++) {
            w.token_embedding_table[i] = new float[p.dim];
            for (int j = 0; j < p.dim; j++) {
                w.token_embedding_table[i][j] = data[index++];
            }
        }

        w.layers = new weights.LayerWeights[n_layers];

        for (int l = 0; l < n_layers; l++) {
            w.layers[l] = new weights.LayerWeights {
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

    static (config config, weights weights) ReadCheckpoint (string checkpoint) {
        using FileStream fs = new FileStream (checkpoint, FileMode.Open, FileAccess.Read);
        using BinaryReader br = new BinaryReader (fs);

        var config = new config {
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
}
