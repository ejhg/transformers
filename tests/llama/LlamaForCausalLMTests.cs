using NUnit.Framework;

namespace llama;

[TestFixture]
public class LlamaForCausalLMTests
{
    [Test]
    public void TestForward()
    {
        int vocabSize = 50;
        int embedSize = 8;
        int hiddenSize = 16;
        int numHeads = 2;
        int numLayers = 2;
        int numQueryGroups = 1;
        Random rand = new Random(0);

        LlamaForCausalLM model = new LlamaForCausalLM(vocabSize, embedSize, hiddenSize, numHeads, numLayers, numQueryGroups, rand);

        int[] inputTokens = { 1, 2, 3, 4, 5 };
        List<double[]> logitsList = model.Forward(inputTokens);
        Assert.AreEqual(inputTokens.Length, logitsList.Count);
        Assert.AreEqual(vocabSize, logitsList[0].Length);
        // Additional checks can be added to verify logits
    }

    [Test]
    public void TestBackward()
    {
        int vocabSize = 50;
        int embedSize = 8;
        int hiddenSize = 16;
        int numHeads = 2;
        int numLayers = 2;
        int numQueryGroups = 1;
        Random rand = new Random(0);

        LlamaForCausalLM model = new LlamaForCausalLM(vocabSize, embedSize, hiddenSize, numHeads, numLayers, numQueryGroups, rand);

        int[] inputTokens = { 1, 2, 3, 4, 5 };
        List<double[]> logitsList = model.Forward(inputTokens);

        // Create dummy gradients for logits
        List<double[]> dLogitsList = logitsList.Select(logits => logits.Select(_ => 0.1).ToArray()).ToList();

        model.Backward(dLogitsList);

        // Verify that gradients have been accumulated in the embedding layer
        for (int i = 0; i < inputTokens.Length; i++)
        {
            int token = inputTokens[i];
            for (int j = 0; j < embedSize; j++)
            {
                Assert.AreNotEqual(0.0, model.TokenEmbedding.Gradients[token, j]);
            }
        }
        // Additional checks can be added to verify gradients in other layers
    }
}
