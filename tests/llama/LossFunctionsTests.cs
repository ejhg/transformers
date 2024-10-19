using NUnit.Framework;

namespace llama;

[TestFixture]
public class LossFunctionsTests
{
    [Test]
    public void TestCrossEntropyLoss()
    {
        double[] logits = { 1.0, 2.0, 3.0 };
        int targetToken = 2;
        double[] dLogits;
        double loss = LossFunctions.CrossEntropyLoss(logits, targetToken, out dLogits);
        Assert.AreEqual(logits.Length, dLogits.Length);
        double[] probabilities = MathOps.Softmax(logits);
        double expectedLoss = -Math.Log(probabilities[targetToken] + 1e-12);
        Assert.AreEqual(expectedLoss, loss, 1e-6);
        // Verify gradient
        for (int i = 0; i < logits.Length; i++)
        {
            double expectedGrad = probabilities[i];
            if (i == targetToken) expectedGrad -= 1.0;
            Assert.AreEqual(expectedGrad, dLogits[i], 1e-6);
        }
    }
}
