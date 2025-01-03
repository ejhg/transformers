namespace llama.c;

public class RunState
{
    // Current wave of activations
    public float[] x; // (dim)
    public float[] xb; // (dim)
    public float[] xb2; // (dim)
    public float[] hb; // (hidden_dim)
    public float[] hb2; // (hidden_dim)
    public float[,] q; // [heads][head_size]
    public float[,] k; // [heads][head_size]
    public float[,] v; // [heads][head_size]
    public float[] logits; // Output logits

    // KV cache
    public LayerCache[] kv_cache;

    public struct LayerCache
    {
        // seq_len | n_kv_head | head_size
        public float[,,] key_cache; // [seq_len][n_kv_heads][head_size]
        public float[,,] value_cache; // [seq_len][n_kv_heads][head_size]
    }
}
