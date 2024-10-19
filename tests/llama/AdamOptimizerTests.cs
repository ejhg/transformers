using NUnit.Framework;

namespace llama;

[TestFixture]
public class AdamOptimizerTests
{
    [Test]
    public void TestOptimizerStep()
    {
        int vocabSize = 50;
        int embedSize = 8;
        int hiddenSize = 16;
        int numHeads = 2;
        int numLayers = 2;
        int numQueryGroups = 1;
        Random rand = new Random(0);

        LlamaForCausalLM model = new LlamaForCausalLM(vocabSize, embedSize, hiddenSize, numHeads, numLayers, numQueryGroups, rand);
        AdamOptimizer optimizer = new AdamOptimizer(learningRate: 0.001);

        // Perform a forward and backward pass
        int[] inputTokens = { 1, 2, 3, 4, 5 };
        List<double[]> logitsList = model.Forward(inputTokens);

        // Create dummy gradients for logits
        List<double[]> dLogitsList = logitsList.Select(logits => logits.Select(_ => 0.1).ToArray()).ToList();

        model.Backward(dLogitsList);

        // Copy weights before optimization
        double[,] weightsBefore = (double[,])model.TokenEmbedding.Weights.Clone();

        // Perform optimization step
        optimizer.Step(model);

        // Check that the weights have been updated
        bool weightsUpdated = false;
        for (int i = 0; i < model.TokenEmbedding.Weights.GetLength(0); i++)
        {
            for (int j = 0; j < model.TokenEmbedding.Weights.GetLength(1); j++)
            {
                if (model.TokenEmbedding.Weights[i, j] != weightsBefore[i, j])
                {
                    weightsUpdated = true;
                    break;
                }
            }
            if (weightsUpdated) break;
        }
        Assert.IsTrue(weightsUpdated);
    }
}
