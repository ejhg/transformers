using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace mingpt.torchsharp.model;

public class TransformerBlock : nn.Module<Tensor, Tensor>
{
    private readonly Configuration config;
    private readonly LayerNorm ln_1;
    private readonly CausalSelfAttention attn;
    private readonly LayerNorm ln_2;
    private readonly Sequential mlp;

    public TransformerBlock (Configuration config) : base (nameof(TransformerBlock)) {
        this.config = config;

        this.ln_1 = LayerNorm (config.n_embd);
        this.attn = new CausalSelfAttention (config);
        this.ln_2 = LayerNorm (config.n_embd);

        // MLP
        this.mlp = Sequential (
            ("c_fc", Linear (config.n_embd, 4 * config.n_embd, dtype: config.dtype)),
            ("gelu", new NewGELU ()),
            ("c_proj", Linear (4 * config.n_embd, config.n_embd, dtype: config.dtype)),
            ("dropout", Dropout (config.resid_pdrop))
        );

        RegisterComponents ();
    }

    public override Tensor forward (Tensor x) {
        using var scope = NewDisposeScope ();

        // Pre-norm architecture: norm -> attention -> residual
        var attn_out = this.attn.forward (this.ln_1.forward (x));
        x = x + attn_out;

        // Pre-norm architecture: norm -> mlp -> residual
        var mlp_out = this.mlp.forward (this.ln_2.forward (x));
        x = x + mlp_out;

        return scope.MoveToOuter (x);
    }
}
