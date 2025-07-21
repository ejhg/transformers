using TorchSharp;

namespace mingpt.torchsharp;

public class Configuration
{
    public int vocab_size { get; set; }
    public int block_size { get; set; }
    public int n_layer { get; set; }
    public int n_head { get; set; }
    public int n_embd { get; set; }
    public float embd_pdrop { get; set; } = 0.1f;
    public float resid_pdrop { get; set; } = 0.1f;
    public float attn_pdrop { get; set; } = 0.1f;
    public torch.ScalarType dtype { get; set; } = torch.ScalarType.Float32;
}
