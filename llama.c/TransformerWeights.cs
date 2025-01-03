namespace llama.c;

public class TransformerWeights
{
    // Token embedding table
    public float[] token_embedding_table; // (vocab_size * dim)

    // Weights for RMSNorms
    public float[][] rms_att_weight; // (n_layers * dim)
    public float[][] rms_ffn_weight; // (n_layers * dim)

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
