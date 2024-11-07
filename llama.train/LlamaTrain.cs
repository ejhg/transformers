using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace llamatrain;

public class ModelArgs
{
    public int Dim { get; set; } = 4096;
    public int NLayers { get; set; } = 32;
    public int NHeads { get; set; } = 32;
    public int? NKvHeads { get; set; } = null;
    public int VocabSize { get; set; } = 32000;
    public int? HiddenDim { get; set; } = null;
    public int MultipleOf { get; set; } = 256;
    public float NormEps { get; set; } = 1e-5f;
    public int MaxSeqLen { get; set; } = 2048;
    public float Dropout { get; set; } = 0.0f;
}

public class RMSNorm : Module<Tensor, Tensor>
{
    private readonly float eps;
    private readonly Parameter weight;

    public RMSNorm (int dim, float eps) : base ("RMSNorm") {
        this.eps = eps;
        this.weight = nn.Parameter (ones (dim));
        RegisterComponents ();
    }

    public override Tensor forward (Tensor x) {
        Tensor _norm (Tensor y) => y * rsqrt (y.pow (2).mean ([-1], keepdim: true) + eps);

        var output = _norm (x.to_type (ScalarType.Float32)).to_type (x.dtype);
        return output * weight;
    }
}

public static class Utilities
{
    public static (Tensor, Tensor) PrecomputeFreqsCis (int dim, int end, float theta = 10000.0f) {
        var arange = torch.arange (0, dim, 2);
        var freqs = 1.0 / pow (theta, arange[..(dim / 2)].to_type (ScalarType.Float32) / dim);
        var t = torch.arange (end, device: freqs.device);
        freqs = outer (t, freqs).to_type (ScalarType.Float32);
        var freqs_cos = cos (freqs);
        var freqs_sin = sin (freqs);
        return (freqs_cos, freqs_sin);
    }

    static Tensor ReshapeForBroadcast (Tensor freqs_cis, Tensor x) {
        var ndim = x.dim ();
        if (ndim <= 1)
            throw new InvalidOperationException ("Invalid tensor dimensions.");
        var expectedShape = new long[] {
            x.shape[1],
            x.shape[^1]
        };
        if (!freqs_cis.shape.SequenceEqual (expectedShape))
            throw new InvalidOperationException ("Shape mismatch.");
        var shape = x.shape.Select ((d, i) => (i == 1 || i == ndim - 1) ? d : 1).ToArray ();
        return freqs_cis.view (shape);
    }

    public static (Tensor, Tensor) ApplyRotaryEmb (Tensor xq, Tensor xk, Tensor freqs_cos, Tensor freqs_sin) {
        var xqShape = xq.shape;
        var xkShape = xk.shape;

        var xqFloat = xq.to_type (ScalarType.Float32);
        var xkFloat = xk.to_type (ScalarType.Float32);

        var xqReshaped = xqFloat.reshape (xqShape[..^1].Concat (new long[] {
            -1,
            2
        }).ToArray ());
        var xkReshaped = xkFloat.reshape (xkShape[..^1].Concat (new long[] {
            -1,
            2
        }).ToArray ());

        var xqUnbind = xqReshaped.unbind (-1);
        var xkUnbind = xkReshaped.unbind (-1);

        var xq_r = xqUnbind[0];
        var xq_i = xqUnbind[1];
        var xk_r = xkUnbind[0];
        var xk_i = xkUnbind[1];

        freqs_cos = ReshapeForBroadcast (freqs_cos, xq_r);
        freqs_sin = ReshapeForBroadcast (freqs_sin, xq_r);

        var xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin;
        var xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos;
        var xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin;
        var xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos;

        var xq_out = stack (new Tensor[] {
            xq_out_r,
            xq_out_i
        }, -1).flatten (3);
        var xk_out = stack (new Tensor[] {
            xk_out_r,
            xk_out_i
        }, -1).flatten (3);

        return (xq_out.to_type (xq.dtype), xk_out.to_type (xk.dtype));
    }

    public static Tensor RepeatKv (Tensor x, int nRep) {
        var shape = x.shape;
        var bs = shape[0];
        var slen = shape[1];
        var nKvHeads = shape[2];
        var headDim = shape[3];

        if (nRep == 1)
            return x;

        var expanded = x.unsqueeze (3).expand (bs, slen, nKvHeads, nRep, headDim);
        var reshaped = expanded.reshape (bs, slen, nKvHeads * nRep, headDim);
        return reshaped;
    }
}

public class Attention : Module
{
    private readonly int nKvHeads;
    private readonly int nLocalHeads;
    private readonly int nLocalKvHeads;
    private readonly int nRep;
    private readonly int headDim;
    private readonly Linear wq, wk, wv, wo;
    private readonly Dropout attnDropout, residDropout;
    private readonly float dropout;
    private readonly bool flash;
    private readonly Tensor mask;

    public Attention (ModelArgs args) : base ("Attention") {
        nKvHeads = args.NKvHeads.HasValue ? args.NKvHeads.Value : args.NHeads;
        if (args.NHeads % nKvHeads != 0)
            throw new InvalidOperationException ("n_heads must be divisible by n_kv_heads.");

        int modelParallelSize = 1;
        nLocalHeads = args.NHeads / modelParallelSize;
        nLocalKvHeads = nKvHeads / modelParallelSize;
        nRep = nLocalHeads / nLocalKvHeads;
        headDim = args.Dim / args.NHeads;

        wq = nn.Linear (args.Dim, args.NHeads * headDim, hasBias: false);
        wk = nn.Linear (args.Dim, nKvHeads * headDim, hasBias: false);
        wv = nn.Linear (args.Dim, nKvHeads * headDim, hasBias: false);
        wo = nn.Linear (args.NHeads * headDim, args.Dim, hasBias: false);

        attnDropout = nn.Dropout (args.Dropout);
        residDropout = nn.Dropout (args.Dropout);
        dropout = args.Dropout;

        // Flash attention not available in TorchSharp as of now
        flash = false;
        if (!flash) {
            Console.WriteLine ("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0");
            mask = full (new long[] {
                1,
                1,
                args.MaxSeqLen,
                args.MaxSeqLen
            }, float.NegativeInfinity, device: wq.weight.device);
            mask = triu (mask, diagonal: 1);
        }

        RegisterComponents ();
    }

    public Tensor Forward (Tensor x, Tensor freqs_cos, Tensor freqs_sin) {
        var bsz = x.shape[0];
        var seqlen = x.shape[1];

        var xq = wq.forward (x).view (bsz, seqlen, nLocalHeads, headDim);
        var xk = wk.forward (x).view (bsz, seqlen, nLocalKvHeads, headDim);
        var xv = wv.forward (x).view (bsz, seqlen, nLocalKvHeads, headDim);

        (xq, xk) = Utilities.ApplyRotaryEmb (xq, xk, freqs_cos, freqs_sin);

        xk = Utilities.RepeatKv (xk, nRep);
        xv = Utilities.RepeatKv (xv, nRep);

        xq = xq.transpose (1, 2);
        xk = xk.transpose (1, 2);
        xv = xv.transpose (1, 2);

        var scores = matmul (xq, xk.transpose (2, 3)) / Math.Sqrt (headDim);
        var maskSlice = mask.slice (2, 0, seqlen, 1).slice (3, 0, seqlen, 1);
        scores = scores + maskSlice;

        scores = nn.functional.softmax (scores.to_type (ScalarType.Float32), dim: -1).to_type (xq.dtype);
        scores = attnDropout.forward (scores);

        var output = matmul (scores, xv);

        output = output.transpose (1, 2).contiguous ().view (bsz, seqlen, -1);

        output = wo.forward (output);
        output = residDropout.forward (output);

        return output;
    }
}

public class FeedForward : Module<Tensor, Tensor>
{
    private readonly Linear w1, w2, w3;
    private readonly Dropout dropout;

    public FeedForward (int dim, int? hiddenDim, int multipleOf, float dropout) : base ("FeedForward") {
        if (!hiddenDim.HasValue) {
            hiddenDim = 4 * dim;
            hiddenDim = (int)(2 * hiddenDim / 3);
            hiddenDim = multipleOf * ((hiddenDim + multipleOf - 1) / multipleOf);
        }

        w1 = nn.Linear (dim, hiddenDim.Value, hasBias: false);
        w2 = nn.Linear (hiddenDim.Value, dim, hasBias: false);
        w3 = nn.Linear (dim, hiddenDim.Value, hasBias: false);
        this.dropout = nn.Dropout (dropout);

        RegisterComponents ();
    }

    public override Tensor forward (Tensor x) {
        var h = nn.functional.silu (w1.forward (x)) * w3.forward (x);
        return dropout.forward (w2.forward (h));
    }
}

public class TransformerBlock : Module
{
    private readonly Attention attention;
    private readonly FeedForward feedForward;
    private readonly RMSNorm attentionNorm;
    private readonly RMSNorm ffnNorm;

    public TransformerBlock (int layerId, ModelArgs args) : base ("TransformerBlock") {
        attention = new Attention (args);
        feedForward = new FeedForward (args.Dim, args.HiddenDim, args.MultipleOf, args.Dropout);
        attentionNorm = new RMSNorm (args.Dim, args.NormEps);
        ffnNorm = new RMSNorm (args.Dim, args.NormEps);

        RegisterComponents ();
    }

    public Tensor Forward (Tensor x, Tensor freqs_cos, Tensor freqs_sin) {
        var h = x + attention.Forward (attentionNorm.forward (x), freqs_cos, freqs_sin);
        var output = h + feedForward.forward (ffnNorm.forward (h));
        return output;
    }
}

public class Transformer : Module
{
    private readonly ModelArgs args;
    private readonly Embedding tokEmbeddings;
    private readonly Dropout dropout;
    private readonly ModuleList<TransformerBlock> layers;
    private readonly RMSNorm norm;
    private readonly Linear output;

    private readonly Tensor freqs_cos;
    private readonly Tensor freqs_sin;

    public Tensor LastLoss { get; private set; }

    public Transformer (ModelArgs args) : base ("Transformer") {
        this.args = args;
        tokEmbeddings = nn.Embedding (args.VocabSize, args.Dim);
        dropout = nn.Dropout (args.Dropout);

        layers = new ModuleList<TransformerBlock> ();
        for (int i = 0; i < args.NLayers; i++)
            layers.append (new TransformerBlock (i, args));

        norm = new RMSNorm (args.Dim, args.NormEps);
        output = nn.Linear (args.Dim, args.VocabSize, hasBias: false);

        // Weight tying
        output.weight = tokEmbeddings.weight;

        // Precompute frequencies
        var freqs = Utilities.PrecomputeFreqsCis (args.Dim / args.NHeads, args.MaxSeqLen);
        freqs_cos = freqs.Item1;
        freqs_sin = freqs.Item2;

        // Initialize weights
        apply (_InitWeights);

        // Special scaled init
        foreach (var (name, param) in named_parameters ()) {
            if (name.EndsWith ("w3.weight") || name.EndsWith ("wo.weight")) {
                nn.init.normal_ (param, mean: 0.0, std: 0.02 / Math.Sqrt (2 * args.NLayers));
            }
        }

        RegisterComponents ();
    }

    private void _InitWeights (Module module) {
        if (module is Linear linear) {
            nn.init.normal_ (linear.weight, mean: 0.0, std: 0.02);
            if (linear.bias is not null) {
                nn.init.zeros_ (linear.bias);
            }
        } else if (module is Embedding embedding) {
            nn.init.normal_ (embedding.weight, mean: 0.0, std: 0.02);
        }
    }

    public Tensor Forward (Tensor tokens, Tensor targets = null) {
        var bsz = tokens.shape[0];
        var seqlen = tokens.shape[1];

        var h = tokEmbeddings.forward (tokens);
        h = dropout.forward (h);

        var freqs_cos_slice = freqs_cos.index_select (0, arange (0, seqlen, device: tokens.device));
        var freqs_sin_slice = freqs_sin.index_select (0, arange (0, seqlen, device: tokens.device));

        foreach (var layer in layers)
            h = layer.Forward (h, freqs_cos_slice, freqs_sin_slice);

        h = norm.forward (h);

        Tensor logits;
        if (targets is not null) {
            logits = output.forward (h);
            LastLoss = nn.functional.cross_entropy (logits.view (-1, logits.shape[^1]), targets.view (-1), ignore_index: -1);
        } else {
            logits = output.forward (h.index_select (1, tensor (new long[] { seqlen - 1 }, device: tokens.device)));
            LastLoss = null;
        }

        return logits;
    }

    public Optimizer ConfigureOptimizers (double weightDecay, double learningRate, (double, double) betas, string deviceType) {
        var paramDict = named_parameters ().ToDictionary (kv => kv.name, kv => kv.parameter);
        paramDict = paramDict.Where (kv => kv.Value.requires_grad).ToDictionary (kv => kv.Key, kv => kv.Value);

        var decayParams = paramDict.Where (kv => kv.Value.dim () >= 2).Select (kv => kv.Value).ToList ();
        var nodecayParams = paramDict.Where (kv => kv.Value.dim () < 2).Select (kv => kv.Value).ToList ();

        // Logging
        var numDecayParams = decayParams.Sum (p => p.numel ());
        var numNodecayParams = nodecayParams.Sum (p => p.numel ());
        Console.WriteLine ($"num decayed parameter tensors: {decayParams.Count}, with {numDecayParams:N0} parameters");
        Console.WriteLine ($"num non-decayed parameter tensors: {nodecayParams.Count}, with {numNodecayParams:N0} parameters");

        // Create optimizer parameter groups
        var optimGroups = new List<AdamW.ParamGroup> {
            new(decayParams, weight_decay: weightDecay),
            new(nodecayParams, weight_decay: 0),
        };

        // TorchSharp does not support fused optimizers yet
        return AdamW (optimGroups, learningRate, betas.Item1, betas.Item2);
    }

    public double EstimateMfu (int fwdbwdPerIter, double dt) {
        var N = named_parameters ().Select (kv => kv.parameter.numel ()).Sum ();
        var cfg = args;
        var L = cfg.NLayers;
        var H = cfg.NHeads;
        var Q = cfg.Dim / cfg.NHeads;
        var T = cfg.MaxSeqLen;
        var flopsPerToken = 6 * N + 12 * L * H * Q * T;
        var flopsPerFwdbwd = flopsPerToken * T;
        var flopsPerIter = flopsPerFwdbwd * fwdbwdPerIter;
        var flopsAchieved = flopsPerIter * (1.0 / dt);
        var flopsPromised = 312e12; // A100 GPU bfloat16 peak flops is 312 TFLOPS
        var mfu = flopsAchieved / flopsPromised;
        return mfu;
    }

    public Tensor Generate (Tensor idx, int maxNewTokens, double temperature = 1.0, int? topK = null) {
        this.eval (); // Ensure the model is in evaluation mode

        for (int i = 0; i < maxNewTokens; i++) {
            // If the sequence context is too long, crop it at max_seq_len
            Tensor idx_cond;
            if (idx.shape[1] <= args.MaxSeqLen) {
                idx_cond = idx;
            } else {
                idx_cond = idx.index (new TensorIndex[] {
                    TensorIndex.Ellipsis,
                    TensorIndex.Slice (idx.shape[1] - args.MaxSeqLen, idx.shape[1])
                });
            }

            // Forward pass
            Tensor logits = this.Forward (idx_cond);
            logits = logits.index_select (1, tensor (idx_cond.shape[1] - 1, device: idx.device)); // Get logits at last time step
            logits = logits.squeeze (1); // Remove time dimension if necessary

            Tensor idx_next;
            if (temperature == 0.0) {
                // Sample the single most likely index
                var topk = logits.topk (1, dim: -1);
                idx_next = topk.indexes;
            } else {
                logits = logits / temperature;
                if (topK.HasValue) {
                    var topk = logits.topk (Math.Min (topK.Value, (int)logits.shape[^1]), dim: -1);
                    var v = topk.values;
                    var threshold = v.index_select (1, tensor (topk.values.shape[1] - 1, device: logits.device));
                    logits = torch.where (logits < threshold, full_like (logits, double.NegativeInfinity), logits);
                }

                // Apply softmax to convert logits to probabilities
                var probs = nn.functional.softmax (logits, dim: -1);
                // Sample from the distribution
                idx_next = probs.multinomial (1);
            }

            // Append sampled index to the running sequence
            idx = torch.cat (new Tensor[] {
                idx,
                idx_next
            }, dim: 1);
        }

        return idx;
    }
}

class Program
{
    static void Main (string[] args) {
        // Hyperparameters and configurations
        string outDir = "out";
        int evalInterval = 2000;
        int logInterval = 1;
        int evalIters = 100;
        bool evalOnly = false;
        bool alwaysSaveCheckpoint = false;
        string initFrom = "scratch"; // "scratch" or "resume"

        int batchSize = 128;
        int maxSeqLen = 256;
        string vocabSource = "llama2"; // "llama2" or "custom"
        int vocabSize = 32000;

        int dim = 288;
        int nLayers = 6;
        int nHeads = 6;
        int nKvHeads = 6;
        int multipleOf = 32;
        float dropout = 0.0f;

        int gradientAccumulationSteps = 4;
        double learningRate = 5e-4;
        int maxIters = 100000;
        double weightDecay = 1e-1;
        double beta1 = 0.9;
        double beta2 = 0.95;
        double gradClip = 1.0;

        bool decayLr = true;
        int warmupIters = 1000;
        int lrDecayIters = maxIters;
        double minLr = 0.0;

        string device = cuda.is_available () ? "cuda" : "cpu";
        string dtype = "float32"; // TorchSharp supports float32 and float64
        bool compile = false; // TorchSharp does not support model compilation

        // Set device
        Device deviceType = device == "cuda" ? CUDA : CPU;

        // Set random seed
        manual_seed (1337);

        // Model initialization
        var modelArgs = new ModelArgs {
            Dim = dim,
            NLayers = nLayers,
            NHeads = nHeads,
            NKvHeads = nKvHeads,
            VocabSize = vocabSize,
            MultipleOf = multipleOf,
            MaxSeqLen = maxSeqLen,
            Dropout = dropout
        };

        Transformer model;

        if (initFrom == "scratch") {
            Console.WriteLine ("Initializing a new model from scratch");
            model = new Transformer (modelArgs);
        } else if (initFrom == "resume") {
            Console.WriteLine ($"Resuming training from {outDir}");
            // Load model from checkpoint (to be implemented)
            throw new NotImplementedException ("Resuming from checkpoint is not implemented.");
        } else {
            throw new ArgumentException ("Invalid init_from option.");
        }

        model.to (deviceType);

        // Optimizer
        var optimizer = model.ConfigureOptimizers (weightDecay, learningRate, (beta1, beta2), device);

        // Training loop
        int iterNum = 0;
        int localIterNum = 0;
        double bestValLoss = double.MaxValue;
        double runningMfu = -1.0;

        var trainBatchIter = GenerateSyntheticData (100000, batchSize, maxSeqLen, vocabSize, deviceType).GetEnumerator ();
        trainBatchIter.MoveNext ();
        var (X, Y) = trainBatchIter.Current;

        // For timing
        var t0 = Stopwatch.GetTimestamp ();
        var stopwatch = Stopwatch.StartNew ();

        Console.WriteLine ("Starting training loop...");
        while (true) {
            // Determine and set the learning rate for this iteration
            double lr = GetLearningRate (iterNum, warmupIters, lrDecayIters, learningRate, minLr, decayLr);

            foreach (var paramGroup in optimizer.ParamGroups) {
                paramGroup.LearningRate = lr;
            }

            // Evaluate the loss on train/val sets and write checkpoints
            if (iterNum % evalInterval == 0) {
                var losses = EstimateLoss (model, evalIters, batchSize, maxSeqLen, vocabSize, deviceType);
                Console.WriteLine ($"step {iterNum}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}");

                if (losses["val"] < bestValLoss || alwaysSaveCheckpoint) {
                    bestValLoss = losses["val"];
                    if (iterNum > 0) {
                        Console.WriteLine ($"saving checkpoint to {outDir}");
                        // Implement model export if needed
                    }
                }

                if (iterNum == 0 && evalOnly) {
                    break;
                }
            }

            optimizer.zero_grad ();

            for (int microStep = 0; microStep < gradientAccumulationSteps; microStep++) {
                // Pre-fetch next batch
                if (!trainBatchIter.MoveNext ()) {
                    trainBatchIter = GenerateSyntheticData (100000, batchSize, maxSeqLen, vocabSize, deviceType).GetEnumerator ();
                    trainBatchIter.MoveNext ();
                }

                (X, Y) = trainBatchIter.Current;
            }

            // Clip the gradient
            if (gradClip > 0) {
                nn.utils.clip_grad_norm_ (model.parameters (), gradClip);
            }

            // Zero gradients
            optimizer.zero_grad ();

            // Timing and logging
            var t1 = Stopwatch.GetTimestamp ();
            var dt = (t1 - t0) / (double)Stopwatch.Frequency;
            t0 = t1;

            if (iterNum % logInterval == 0) {
                // Get loss as float, scale up due to the divide above.
                var lossf = model.LastLoss.item<double> () * gradientAccumulationSteps;

                if (localIterNum >= 5) {
                    var mfu = model.EstimateMfu (batchSize * gradientAccumulationSteps, dt);
                    runningMfu = runningMfu == -1.0 ? mfu : 0.9 * runningMfu + 0.1 * mfu;
                } else {
                    runningMfu = -1.0;
                }

                Console.WriteLine ($"{iterNum} | loss {lossf:F4} | lr {lr:E} | {dt * 1000:F2}ms | mfu {runningMfu * 100:F2}%");
            }

            iterNum++;
            localIterNum++;

            // Termination conditions
            if (iterNum > maxIters)
                break;
        }

        stopwatch.Stop ();
        Console.WriteLine ($"Training completed in {stopwatch.Elapsed.TotalSeconds} seconds");
    }

    static double GetLearningRate (int iterNum, int warmupIters, int lrDecayIters, double learningRate, double minLr, bool decayLr) {
        if (!decayLr)
            return learningRate;

        if (iterNum < warmupIters) {
            return learningRate * iterNum / warmupIters;
        } else if (iterNum > lrDecayIters) {
            return minLr;
        } else {
            double decayRatio = (double)(iterNum - warmupIters) / (lrDecayIters - warmupIters);
            double coeff = 0.5 * (1.0 + Math.Cos (Math.PI * decayRatio));
            return minLr + coeff * (learningRate - minLr);
        }
    }

    static Dictionary<string, double> EstimateLoss (Transformer model, int evalIters, int batchSize, int maxSeqLen, int vocabSize,
        Device deviceType) {
        var outDict = new Dictionary<string, double> ();
        model.eval ();

        foreach (var split in new[] {
                     "train",
                     "val"
                 }) {
            var batchIter = GenerateSyntheticData (10000, batchSize, maxSeqLen, vocabSize, deviceType).GetEnumerator ();
            var losses = new List<double> ();

            for (int k = 0; k < evalIters; k++) {
                if (!batchIter.MoveNext ()) {
                    batchIter = GenerateSyntheticData (10000, batchSize, maxSeqLen, vocabSize, deviceType).GetEnumerator ();
                    batchIter.MoveNext ();
                }

                var (X, Y) = batchIter.Current;
                X = X.to (deviceType);
                Y = Y.to (deviceType);

                using (no_grad ()) {
                    var logits = model.Forward (X, Y);
                    var loss = model.LastLoss;
                    losses.Add (loss.item<double> ());
                }
            }

            outDict[split] = losses.Average ();
        }

        model.train ();
        return outDict;
    }

    static IEnumerable<(Tensor, Tensor)> GenerateSyntheticData (int totalSamples, int batchSize, int seqLen, int vocabSize, Device device) {
        int samplesGenerated = 0;
        while (samplesGenerated < totalSamples) {
            int currentBatchSize = Math.Min (batchSize, totalSamples - samplesGenerated);
            var data = randint (0, vocabSize, new long[] {
                currentBatchSize,
                seqLen
            }, device: device);
            var targets = randint (0, vocabSize, new long[] {
                currentBatchSize,
                seqLen
            }, device: device);
            samplesGenerated += currentBatchSize;
            yield return (data, targets);
        }
    }
}
