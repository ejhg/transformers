using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace mingpt.torchsharp.model;

public class CausalSelfAttention : nn.Module<Tensor, Tensor>
{
    private readonly Configuration config;
    private readonly Linear c_attn;
    private readonly Linear c_proj;
    private readonly Dropout attn_dropout;
    private readonly Dropout resid_dropout;
    private readonly Tensor bias;
    private readonly int n_head;
    private readonly int n_embd;

    public CausalSelfAttention(Configuration config) : base(nameof(CausalSelfAttention))
    {
        this.config = config;

        if (config.n_embd % config.n_head != 0)
            throw new ArgumentException("n_embd must be divisible by n_head");

        this.n_head = config.n_head;
        this.n_embd = config.n_embd;

        // key, query, value projections for all heads, but in a batch
        this.c_attn = Linear(config.n_embd, 3 * config.n_embd, dtype: config.dtype);

        // output projection
        this.c_proj = Linear(config.n_embd, config.n_embd, dtype: config.dtype);

        // regularization
        this.attn_dropout = Dropout(config.attn_pdrop);
        this.resid_dropout = Dropout(config.resid_pdrop);

        // causal mask to ensure that attention is only applied to the left in the input sequence
        var mask = tril(ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size);
        register_buffer("bias", mask);
        this.bias = mask;

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();

        var size = x.shape;
        var B = size[0]; // batch size
        var T = size[1]; // sequence length
        var C = size[2]; // embedding dimensionality (n_embd)

        // calculate query, key, values for all heads in batch and move head forward to be the batch dim
        var qkv = this.c_attn.forward(x);
        var chunks = qkv.split(this.n_embd, dim: 2);
        var q = chunks[0];
        var k = chunks[1];
        var v = chunks[2];

        var head_dim = C / this.n_head;

        k = k.view(B, T, this.n_head, head_dim).transpose(1, 2); // (B, nh, T, hs)
        q = q.view(B, T, this.n_head, head_dim).transpose(1, 2); // (B, nh, T, hs)
        v = v.view(B, T, this.n_head, head_dim).transpose(1, 2); // (B, nh, T, hs)

        // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        var att = matmul(q, k.transpose(-2, -1)) * (1.0 / Math.Sqrt(k.size(-1)));
        var causal_mask = this.bias.slice(2, 0, T, 1).slice(3, 0, T, 1);
        att = att.masked_fill(causal_mask == 0, float.NegativeInfinity);
        att = functional.softmax(att, dim: -1);
        att = this.attn_dropout.forward(att);

        var y = matmul(att, v); // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C); // re-assemble all head outputs side by side

        // output projection
        y = this.resid_dropout.forward(this.c_proj.forward(y));

        return scope.MoveToOuter(y);
    }
}
