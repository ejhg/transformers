namespace mingpt.torchsharp;

public class TrainerConfig
{
    public string? device { get; set; } = null;
    public int num_workers { get; set; } = 4;
    public int? max_iters { get; set; } = null;
    public int batch_size { get; set; } = 64;
    public double learning_rate { get; set; } = 3e-4;
    public (double, double) betas { get; set; } = (0.9, 0.95);
    public double weight_decay { get; set; } = 0.1; // only applied on matmul weights
    public double grad_norm_clip { get; set; } = 1.0;
}
