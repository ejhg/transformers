namespace llama.cs;

static class RunState
{
    public static run_state createRunState (config p) {
        int head_size = p.dim / p.n_heads;

        var kv_cache = Enumerable
            .Range (0, p.n_layers)
            .Select (_ => new run_state.LayerCache {
                key_cache = new float[p.seq_len, p.n_kv_heads, p.dim / p.n_heads],
                value_cache = new float[p.seq_len, p.n_kv_heads, p.dim / p.n_heads]
            }).ToArray ();

        return new run_state {
            x = new float[p.dim],
            xb = new float[p.dim],
            xb2 = new float[p.dim],
            hb = new float[p.hidden_dim],
            hb2 = new float[p.hidden_dim],
            q = new float[p.n_heads, head_size],
            k = new float[p.n_heads, head_size],
            v = new float[p.n_heads, head_size],
            logits = new float[p.vocab_size],
            kv_cache = kv_cache,
        };
    }
}
