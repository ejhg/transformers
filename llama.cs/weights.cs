namespace llama.cs.svd;

public class weights
{
    // Token embedding table
    public float[][] token_embedding_table; // [vocab_size, dim]

    public LayerWeights[] layers;

    public class SVDMatrix
    {
        public bool use_svd = false; // Whether this matrix is SVD decomposed
        public int rank = 0; // Rank of the decomposition
        public float[,]? U = null; // Left singular vectors [rows, rank]
        public float[]? S = null; // Singular values [rank]
        public float[,]? Vt = null; // Right singular vectors (transposed) [rank, cols]
        public float[,]? original = null; // Original matrix - could be set to null after decomposition to save memory

        // Statistics for performance tuning
        public float error_ratio = 0.0f; // Error ratio compared to original matrix
    }

    public class LayerWeights
    {
        // Weights for RMSNorms
        public float[] rms_att_weight; // [dim]
        public float[] rms_ffn_weight; // [dim]

        // Weights for matmuls. note dim == n_heads * head_size
        public float[,] wq; // [n_heads * head_size][dim]
        public SVDMatrix wq_svd;

        public float[,] wk; // [n_kv_heads * head_size][dim]
        public SVDMatrix wk_svd;

        public float[,] wv; // [n_kv_heads * head_size][dim]
        public SVDMatrix wv_svd;

        public float[,] wo; // [dim][n_heads * head_size]
        public SVDMatrix wo_svd;

        // Weights for FFN
        public float[,] w1; // [hidden_dim][dim]
        public SVDMatrix w1_svd;

        public float[,] w2; // [dim][hidden_dim]
        public SVDMatrix w2_svd;

        public float[,] w3; // [hidden_dim][dim]
        public SVDMatrix w3_svd;
    }

    // Final RMSNorm
    public float[] rms_final_weight; // [dim]

    // Classifier weights for the logits
    public float[][] wcls; // [vocab_size, dim]
    // Note: We don't apply SVD to token embedding and classifier weights since they're jagged arrays
    // If needed, we can convert them to 2D arrays for SVD processing
}
