using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using mingpt.torchsharp.model;

namespace mingpt.torchsharp;

public class GPT : nn.Module<Tensor, Tensor?, (Tensor logits, Tensor? loss)>
{
    private readonly Configuration config;
    private readonly ModuleDict<nn.Module> transformer;
    private readonly Linear lm_head;
    private readonly int block_size;

    public GPT (Configuration config) : base (nameof(GPT)) {
        if (config.vocab_size == 0) throw new ArgumentException ("vocab_size must be set");
        if (config.block_size == 0) throw new ArgumentException ("block_size must be set");
        if (config.n_layer == 0) throw new ArgumentException ("n_layer must be set");
        if (config.n_head == 0) throw new ArgumentException ("n_head must be set");
        if (config.n_embd == 0) throw new ArgumentException ("n_embd must be set");

        this.config = config;
        this.block_size = config.block_size;

        var layers = new List<nn.Module<Tensor, Tensor>> ();
        for (int i = 0; i < config.n_layer; i++) {
            layers.Add (new TransformerBlock (config));
        }

        this.transformer = new ModuleDict<nn.Module> ();
        this.transformer.Add ("wte", Embedding (config.vocab_size, config.n_embd, dtype: config.dtype));
        this.transformer.Add ("wpe", Embedding (config.block_size, config.n_embd, dtype: config.dtype));
        this.transformer.Add ("drop", Dropout (config.embd_pdrop));
        this.transformer.Add ("h", ModuleList (layers.ToArray ()));
        this.transformer.Add ("ln_f", LayerNorm (config.n_embd));

        this.lm_head = Linear (config.n_embd, config.vocab_size, hasBias: false, dtype: config.dtype);

        // Initialize weights
        InitWeights ();

        RegisterComponents ();
    }

    private void InitWeights () {
        apply (module => {
            if (module is Linear linear) {
                init.normal_ (linear.weight, mean: 0.0, std: 0.02);
                if (linear.bias is not null) {
                    init.zeros_ (linear.bias);
                }
            } else if (module is Embedding embedding) {
                init.normal_ (embedding.weight, mean: 0.0, std: 0.02);
            } else if (module is LayerNorm layerNorm) {
                init.zeros_ (layerNorm.bias);
                init.ones_ (layerNorm.weight);
            }
        });

        // Apply special scaled init to the residual projections, per GPT-2 paper
        foreach (var (name, param) in named_parameters ()) {
            if (name.EndsWith ("c_proj.weight")) {
                var std = 0.02 / Math.Sqrt (2 * config.n_layer);
                init.normal_ (param, mean: 0.0, std: std);
            }
        }
    }

    public optim.Optimizer ConfigureOptimizers (TrainerConfig trainConfig) {
        // Separate out all parameters to those that will and won't experience regularizing weight decay
        var decay = new HashSet<string> ();
        var no_decay = new HashSet<string> ();

        foreach (var (module_name, module) in named_modules ()) {
            foreach (var (param_name, param) in module.named_parameters ()) {
                var fpn = string.IsNullOrEmpty (module_name) ? param_name : $"{module_name}.{param_name}";

                if (param_name.EndsWith ("bias")) {
                    no_decay.Add (fpn);
                } else if (param_name.EndsWith ("weight") && (module is Linear)) {
                    decay.Add (fpn);
                } else if (param_name.EndsWith ("weight") && (module is LayerNorm || module is Embedding)) {
                    no_decay.Add (fpn);
                }
            }
        }

        var param_dict = named_parameters ().ToDictionary (kv => kv.name, kv => kv.parameter);

        var decay_params = decay.Select (pn => param_dict[pn]).ToList ();
        var no_decay_params = no_decay.Select (pn => param_dict[pn]).ToList ();

        return optim.AdamW (
            new[] {
                decay_params,
                no_decay_params
            }.SelectMany (x => x),
            lr: trainConfig.learning_rate,
            weight_decay: trainConfig.weight_decay);
    }

    public override (Tensor logits, Tensor? loss) forward (Tensor idx, Tensor? targets = null) {
        using var scope = NewDisposeScope ();

        var device = idx.device;
        var size = idx.shape;
        var b = size[0];
        var t = size[1];

        if (t > this.block_size)
            throw new ArgumentException ($"Cannot forward sequence of length {t}, block size is only {this.block_size}");

        var pos = arange (0, t, dtype: ScalarType.Int64, device: device).unsqueeze (0); // shape (1, t)

        // Forward the GPT model itself
        var tok_emb = ((Embedding)this.transformer["wte"]).forward (idx); // token embeddings of shape (b, t, n_embd)
        var pos_emb = ((Embedding)this.transformer["wpe"]).forward (pos); // position embeddings of shape (1, t, n_embd)
        var x = ((Dropout)this.transformer["drop"]).forward (tok_emb + pos_emb);

        var moduleList = (ModuleList<nn.Module<Tensor, Tensor>>)this.transformer["h"];
        foreach (var block in moduleList) {
            x = block.forward (x);
        }

        x = ((LayerNorm)this.transformer["ln_f"]).forward (x);
        var logits = this.lm_head.forward (x);

        // If we are given some desired targets also calculate the loss
        Tensor? loss = null;
        if (targets is not null) {
            loss = functional.cross_entropy (logits.view (-1, logits.size (-1)), targets.view (-1), ignore_index: -1);
            loss = scope.MoveToOuter (loss);
        }

        return (scope.MoveToOuter (logits), loss);
    }

    public Tensor Generate (Tensor idx, int max_new_tokens, double temperature = 1.0, bool do_sample = false, int? top_k = null) {
        for (int i = 0; i < max_new_tokens; i++) {
            // If the sequence context is growing too long we must crop it at block_size
            var idx_cond = idx.size (1) <= this.block_size ? idx : idx.slice (1, -this.block_size, idx.size (1), 1);

            // Forward the model to get the logits for the index in the sequence
            var (logits, _) = this.forward (idx_cond);

            // Pluck the logits at the final step and scale by desired temperature
            // Python: logits[:, -1, :] becomes logits.slice(1, -1, logits.size(1), 1).squeeze(1)
            logits = logits.slice (1, -1, logits.size (1), 1).squeeze(1) / temperature;

            // Optionally crop the logits to only the top k options
            if (top_k.HasValue) {
                var (v, _) = topk (logits, top_k.Value);
                // Python: logits[logits < v[:, [-1]]] = -float('Inf')
                logits = logits.masked_fill (logits < v.slice (-1, -1, v.size (-1), 1), double.NegativeInfinity);
            }

            // Apply softmax to convert logits to (normalized) probabilities
            var probs = functional.softmax (logits, dim: -1);

            // Either sample from the distribution or take the most likely element
            Tensor idx_next;
            if (do_sample) {
                idx_next = multinomial (probs, num_samples: 1);
            } else {
                var (_, indices) = topk (probs, k: 1, dim: -1);
                idx_next = indices;
            }

            // Append sampled index to the running sequence and continue
            idx = cat (new[] { idx, idx_next }, dim: 1);
        }

        return idx;
    }
}
