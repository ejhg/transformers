namespace llama.cs;

static class ForwardPass
{
    public static float[] Forward (model model, int token, int pos) {
        var p = model.config;
        var w = model.weights;
        var s = model.state;

        var dim = p.dim;
        var head_size = p.dim / p.n_heads;
        var kv_dim = head_size * p.n_kv_heads;

        // Copy token embedding into x
        Array.Copy (w.token_embedding_table[token], 0, s.x, 0, dim);

        // Forward pass through layers
        for (var l = 0; l < p.n_layers; l++) {
            var layer = w.layers[l];
            var _cache = s.kv_cache[l];

            // Attention rmsnorm
            math.RmsNorm (s.xb, s.x, layer.rms_att_weight);

            // Compute q, k, v
            math.MatMul (s.q, s.xb, layer.wq);
            math.MatMul (s.k, s.xb, layer.wk);
            math.MatMul (s.v, s.xb, layer.wv);

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

                    score /= MathF.Sqrt (head_size);
                    att[t] = score;
                }

                // Softmax the attention scores
                math.Softmax (att, pos + 1);

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
            math.MatMul (s.xb2, s.xb, layer.wo);

            // Residual connection
            for (var i = 0; i < dim; i++) {
                s.x[i] += s.xb2[i];
            }

            // FFN rmsnorm
            math.RmsNorm (s.xb, s.x, layer.rms_ffn_weight);

            // FFN computation
            math.MatMul (s.hb, s.xb, layer.w1);
            math.MatMul (s.hb2, s.xb, layer.w3);

            // SwiGLU activation
            for (var i = 0; i < p.hidden_dim; i++) {
                var val = s.hb[i];
                val *= 1.0f / (1.0f + MathF.Exp (-val)); // SiLU activation
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // Final FFN matmul
            math.MatMul (s.xb, s.hb, layer.w2);

            // Residual connection
            for (var i = 0; i < dim; i++) {
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
