namespace llama.cs;

static class ForwardPass
{
    public static float[] Forward (model model, int token, int pos) {
        // Convenience variables
        config p = model.config;
        weights w = model.weights;
        run_state s = model.state;
        int head_size = p.dim / p.n_heads;
        int kv_head_size = head_size; // Assuming kv_head_size == head_size

        // Copy token embedding into x
        Array.Copy (w.token_embedding_table[token], 0, s.x, 0, p.dim);

        // Forward pass through layers
        for (int l = 0; l < p.n_layers; l++) {
            var layer = w.layers[l];
            var _cache = s.kv_cache[l];

            // Attention rmsnorm
            math.RmsNorm (s.xb, s.x, layer.rms_att_weight);

            // Compute q, k, v
            math.MatMul (s.q, s.xb, layer.wq);
            math.MatMul (s.k, s.xb, layer.wk);
            math.MatMul (s.v, s.xb, layer.wv);

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
                math.Softmax (att, pos + 1);

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
            math.MatMul (s.xb2, s.xb, layer.wo);

            // Residual connection
            for (int i = 0; i < p.dim; i++) {
                s.x[i] += s.xb2[i];
            }

            // FFN rmsnorm
            math.RmsNorm (s.xb, s.x, layer.rms_ffn_weight);

            // FFN computation
            math.MatMul (s.hb, s.xb, layer.w1);
            math.MatMul (s.hb2, s.xb, layer.w3);

            // SwiGLU activation
            for (int i = 0; i < p.hidden_dim; i++) {
                float val = s.hb[i];
                val *= 1.0f / (1.0f + MathF.Exp (-val)); // SiLU activation
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // Final FFN matmul
            math.MatMul (s.xb, s.hb, layer.w2);

            // Residual connection
            for (int i = 0; i < p.dim; i++) {
                s.x[i] += s.xb[i];
            }
        }

        // Final rmsnorm
        math.RmsNorm (s.x, s.x, w.rms_final_weight);

        // Classifier into logits
        math.MatMul (s.logits, s.x, w.wcls);

        return s.logits;
    }
}
