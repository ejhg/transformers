namespace llama.cs;

public class run_state
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