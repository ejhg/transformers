namespace llama.cs;

static class RunState
{
    public static run_state createRunState (config p) {
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
}
