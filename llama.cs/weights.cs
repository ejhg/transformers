namespace llama.cs;

public class weights
{
    // Token embedding table
    public float[][] token_embedding_table; // [vocab_size, dim]

    public LayerWeights[] layers;

    public class LayerWeights
    {
        // Weights for RMSNorms
        public float[] rms_att_weight; // [dim]
        public float[] rms_ffn_weight; // [dim]

        // Weights for matmuls. note dim == n_heads * head_size
        public float[,] wq; // [n_heads * head_size][dim]
        public float[,] wk; // [n_kv_heads * head_size][dim]
        public float[,] wv; // [n_kv_heads * head_size][dim]
        public float[,] wo; // [dim][n_heads * head_size]

        // Weights for FFN
        public float[,] w1; // [hidden_dim][dim]
        public float[,] w2; // [dim][hidden_dim]
        public float[,] w3; // [hidden_dim][dim]
    }

    // Final RMSNorm
    public float[] rms_final_weight; // [dim]

    // Classifier weights for the logits
    public float[][] wcls; // [vocab_size, dim]
}
