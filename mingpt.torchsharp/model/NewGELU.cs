using static TorchSharp.torch;

namespace mingpt.torchsharp.model;

public class NewGELU : nn.Module<Tensor, Tensor>
{
    public NewGELU () : base (nameof(NewGELU)) {
    }

    public override Tensor forward (Tensor x) {
        using var scope = NewDisposeScope ();

        // GELU activation function: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        var inner = x + 0.044715f * pow (x, 3);
        inner = inner * Math.Sqrt (2.0 / Math.PI);
        inner = tanh (inner);
        var result = 0.5f * x * (1.0f + inner);

        return scope.MoveToOuter (result);
    }
}
