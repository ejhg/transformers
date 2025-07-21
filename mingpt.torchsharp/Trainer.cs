using TorchSharp;
using static TorchSharp.torch;

namespace mingpt.torchsharp;

public class Trainer
{
    private readonly TrainerConfig config;
    private readonly GPT model;
    private readonly utils.data.Dataset<Dictionary<string, Tensor>> train_dataset;
    private readonly torch.Device device;
    private optim.Optimizer? optimizer;
    private readonly Dictionary<string, List<Action<Trainer>>> callbacks;

    public int iter_num { get; private set; } = 0;
    public double iter_time { get; private set; } = 0.0;
    public double iter_dt { get; private set; } = 0.0;
    public Tensor? loss { get; private set; }

    public Trainer (TrainerConfig config, GPT model, utils.data.Dataset<Dictionary<string, Tensor>> train_dataset) {
        this.config = config;
        this.model = model;
        this.train_dataset = train_dataset;
        this.callbacks = new Dictionary<string, List<Action<Trainer>>> ();

        // Determine the device we'll train on
        if (config.device == null) {
            this.device = torch.cuda.is_available () ? CUDA : CPU;
        } else {
            this.device = new torch.Device (config.device);
        }

        this.model.to (this.device);
        Console.WriteLine ($"Running on device {this.device}");
    }

    public void AddCallback (string onevent, Action<Trainer> callback) {
        if (!this.callbacks.ContainsKey (onevent))
            this.callbacks[onevent] = new List<Action<Trainer>> ();
        this.callbacks[onevent].Add (callback);
    }

    public void SetCallback (string onevent, Action<Trainer> callback) {
        this.callbacks[onevent] = new List<Action<Trainer>> { callback };
    }

    private void TriggerCallbacks (string onevent) {
        if (this.callbacks.ContainsKey (onevent)) {
            foreach (var callback in this.callbacks[onevent]) {
                callback (this);
            }
        }
    }

    public void Run () {
        this.optimizer = this.model.ConfigureOptimizers (this.config);

        this.model.train ();
        this.iter_num = 0;
        this.iter_time = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds ();

        var random = new Random ();

        while (true) {
            // Simple batch creation - randomly sample from dataset
            var batch_data = new List<Dictionary<string, Tensor>> ();
            for (int i = 0; i < this.config.batch_size; i++) {
                var idx = random.NextInt64 (0, this.train_dataset.Count);
                batch_data.Add (this.train_dataset.GetTensor (idx));
            }

            // Stack the batch tensors
            var input_ids_list = batch_data.Select (d => d["input_ids"]).ToArray ();
            var labels_list = batch_data.Select (d => d["labels"]).ToArray ();

            var batch = new Dictionary<string, Tensor> {
                ["input_ids"] = stack (input_ids_list),
                ["labels"] = stack (labels_list)
            };

            var x = batch["input_ids"].to (this.device);
            var y = batch["labels"].to (this.device);

            // Forward the model
            var (logits, loss) = this.model.forward (x, y);
            this.loss = loss;

            // Backprop and update the parameters
            this.optimizer.zero_grad ();
            if (loss is not null) {
                loss.backward ();
                nn.utils.clip_grad_norm_ (this.model.parameters (), max_norm: this.config.grad_norm_clip);
                this.optimizer.step ();
            }

            this.TriggerCallbacks ("on_batch_end");

            if (this.iter_num % 10 == 0) {
                var loss_val = loss?.ToScalar () ?? 0.0;
                Console.WriteLine ($"iter_dt {this.iter_dt:F2}ms, iter {this.iter_num}, loss {loss_val:F5}");
            }

            this.iter_num++;
            var tnow = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds ();
            this.iter_dt = tnow - this.iter_time;
            this.iter_time = tnow;

            // Termination conditions
            if (this.config.max_iters.HasValue && this.iter_num >= this.config.max_iters.Value) {
                break;
            }
        }
    }
}
