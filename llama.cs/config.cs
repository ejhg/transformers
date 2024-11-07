namespace llama.cs;

public class config
{
    public int dim; // Transformer dimension
    public int hidden_dim; // For FFN layers
    public int n_layers; // Number of layers
    public int n_heads; // Number of query heads
    public int n_kv_heads; // Number of key/value heads
    public int vocab_size; // Vocabulary size, usually 256 (byte-level)
    public int seq_len; // Max sequence length
}
