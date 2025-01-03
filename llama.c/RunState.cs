namespace llama.c;

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
