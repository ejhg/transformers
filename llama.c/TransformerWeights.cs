namespace llama.c;

public class TransformerWeights
{
    // Token embedding table
    public float[] token_embedding_table; // [vocab_size * dim]

    // Weights for RMSNorms
    public float[][] rms_att_weight; // [n_layers * dim]
    public float[][] rms_ffn_weight; // [n_layers * dim]

    // Weights for matmuls
    public float[][,] wq; // [n_layers] [n_heads * head_size][dim]
    public float[][,] wk; // [n_layers] [n_kv_heads * head_size][dim]
    public float[][,] wv; // [n_layers] [n_kv_heads * head_size][dim]
    public float[][,] wo; // [n_layers] [dim][n_heads * head_size]

    // Weights for FFN
    public float[][,] w1; // [n_layers, hidden_dim * dim]
    public float[][,] w2; // [n_layers, dim * hidden_dim]
    public float[][,] w3; // [n_layers, hidden_dim * dim]

    // Final RMSNorm
    public float[] rms_final_weight; // [dim]

    // Classifier weights for the logits
    public float[] wcls; // [vocab_size, dim]
}
