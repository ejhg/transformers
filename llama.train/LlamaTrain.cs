using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

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

public class RMSNorm : nn.Module<Tensor, Tensor>
{
    private float eps;
    private Parameter weight;

    public RMSNorm (int dim, float eps) : base (nameof(RMSNorm)) {
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

public class Attention : nn.Module
{
    private int nKvHeads;
    private int nLocalHeads;
    private int nLocalKvHeads;
    private int nRep;
    private int headDim;
    private Linear wq, wk, wv, wo;
    private Dropout attnDropout, residDropout;
    private float dropout;
    private bool flash;
    private Tensor mask;

    public Attention (ModelArgs args) : base (nameof(Attention)) {
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

public class FeedForward : nn.Module<Tensor, Tensor>
{
    private Linear w1, w2, w3;
    private Dropout dropout;

    public FeedForward (int dim, int? hiddenDim, int multipleOf, float dropout) : base (nameof(FeedForward)) {
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

public class TransformerBlock : nn.Module
{
    private Attention attention;
    private FeedForward feedForward;
    private RMSNorm attentionNorm;
    private RMSNorm ffnNorm;

    public TransformerBlock (int layerId, ModelArgs args) : base (nameof(TransformerBlock)) {
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

public class Transformer : nn.Module
{
    private ModelArgs args;
    private Embedding tokEmbeddings;
    private Dropout dropout;
    private List<TransformerBlock> layers;
    private RMSNorm norm;
    private Linear output;

    private Tensor freqs_cos;
    private Tensor freqs_sin;

    public Tensor LastLoss { get; private set; }

    public Transformer (ModelArgs args) : base (nameof(Transformer)) {
        this.args = args;
        tokEmbeddings = nn.Embedding (args.VocabSize, args.Dim);
        dropout = nn.Dropout (args.Dropout);

        layers = new List<TransformerBlock> ();
        for (int i = 0; i < args.NLayers; i++)
            layers.Add (new TransformerBlock (i, args));

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

    private void _InitWeights (nn.Module module) {
        if (module is Linear linear) {
            nn.init.normal_ (linear.weight, mean: 0.0, std: 0.02);
            if (!ReferenceEquals (linear.bias, null)) {
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

        var freqs_cos_slice = freqs_cos.index (new TensorIndex[] { TensorIndex.Slice (0, seqlen) });
        var freqs_sin_slice = freqs_sin.index (new TensorIndex[] { TensorIndex.Slice (0, seqlen) });

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

    public optim.Optimizer ConfigureOptimizers (double weightDecay, double learningRate, (double, double) betas, string deviceType) {
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
        var optimGroups = new List<Dictionary<string, object>> {
            new Dictionary<string, object> {
                { "params", decayParams },
                { "weight_decay", weightDecay }
            },
            new Dictionary<string, object> {
                { "params", nodecayParams },
                { "weight_decay", 0.0 }
            }
        };

        // TorchSharp does not support fused optimizers yet
        var optimizer = optim.AdamW (optimGroups, learningRate, beta1: betas.Item1, beta2: betas.Item2);

        Console.WriteLine ($"Using fused AdamW: False");
        return optimizer;
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
                idx_next = topk.indices;
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

        // Data loader (using synthetic data for demonstration)
        var trainData = GenerateSyntheticData (100000, batchSize, maxSeqLen, vocabSize, deviceType).GetEnumerator ();
        var valData = GenerateSyntheticData (10000, batchSize, maxSeqLen, vocabSize, deviceType).GetEnumerator ();

        // For timing
        var t0 = Stopwatch.GetTimestamp ();
        var stopwatch = Stopwatch.StartNew ();

        Console.WriteLine ("Starting training loop...");
        while (iterNum <= maxIters) {
            model.train ();

            optimizer.zero_grad ();

            double runningLoss = 0.0;
            for (int accStep = 0; accStep < gradientAccumulationSteps; accStep++) {
                if (!trainData.MoveNext ()) {
                    trainData = GenerateSyntheticData (100000, batchSize, maxSeqLen, vocabSize, deviceType).GetEnumerator ();
                    trainData.MoveNext ();
                }

                var (X, Y) = trainData.Current;

                X = X.to (deviceType);
                Y = Y.to (deviceType);

                var logits = model.Forward (X, Y);
                var loss = model.LastLoss / gradientAccumulationSteps;
                loss.backward ();

                runningLoss += loss.item<double> ();
            }

            // Gradient clipping
            if (gradClip > 0) {
                nn.utils.clip_grad_norm_ (model.parameters (), gradClip);
            }

            optimizer.step ();

            // Timing and logging
            var t1 = Stopwatch.GetTimestamp ();
            var dt = (t1 - t0) / (double)Stopwatch.Frequency;
            t0 = t1;

            if (iterNum % logInterval == 0) {
                var lossf = runningLoss * gradientAccumulationSteps;
                if (localIterNum >= 5) {
                    var mfu = model.EstimateMfu (batchSize * gradientAccumulationSteps, dt);
                    runningMfu = runningMfu == -1.0 ? mfu : 0.9 * runningMfu + 0.1 * mfu;
                } else {
                    runningMfu = -1.0;
                }

                Console.WriteLine ($"{iterNum} | loss {runningLoss:F4} | lr {learningRate:E} | {dt * 1000:F2}ms | mfu {runningMfu * 100:F2}%");
            }

            localIterNum++;
            iterNum++;

            // Evaluation
            if (iterNum % evalInterval == 0 && iterNum > 0) {
                model.eval ();
                double valLoss = 0.0;
                int valSteps = 0;

                using (torch.no_grad ()) {
                    valData = GenerateSyntheticData (10000, batchSize, maxSeqLen, vocabSize, deviceType).GetEnumerator ();
                    while (valData.MoveNext ()) {
                        var (X_val, Y_val) = valData.Current;
                        X_val = X_val.to (deviceType);
                        Y_val = Y_val.to (deviceType);

                        var logits = model.Forward (X_val, Y_val);
                        var loss = model.LastLoss;
                        valLoss += loss.item<double> ();

                        valSteps++;
                        if (valSteps >= evalIters)
                            break;
                    }
                }

                valLoss /= valSteps;

                Console.WriteLine ($"Validation Loss after {iterNum} iterations: {valLoss}");

                if (valLoss < bestValLoss) {
                    bestValLoss = valLoss;
                    Console.WriteLine ("New best validation loss, saving model...");
                    // Save model (to be implemented)
                }

                model.train ();
            }

            // Learning rate scheduling
            if (decayLr) {
                double lr;
                if (iterNum < warmupIters) {
                    lr = learningRate * iterNum / warmupIters;
                } else if (iterNum > lrDecayIters) {
                    lr = minLr;
                } else {
                    var decayRatio = (double)(iterNum - warmupIters) / (lrDecayIters - warmupIters);
                    var coeff = 0.5 * (1.0 + Math.Cos (Math.PI * decayRatio));
                    lr = minLr + coeff * (learningRate - minLr);
                }

                foreach (var paramGroup in optimizer.ParamGroups) {
                    paramGroup.LearningRate = lr;
                }
            }
        }

        stopwatch.Stop ();
        Console.WriteLine ($"Training completed in {stopwatch.Elapsed.TotalSeconds} seconds");
    }

    // Generate synthetic data for demonstration purposes
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
