using NUnit.Framework;

namespace llama;

[TestFixture]
public class SelfAttentionTests
{
    [Test]
    public void TestForward()
    {
        int embedSize = 8;
        int numHeads = 2;
        int numQueryGroups = 1;
        Random rand = new Random(0);
        SelfAttention sa = new SelfAttention(embedSize, numHeads, numQueryGroups, rand);

        List<double[]> inputs = new List<double[]>()
        {
            new double[] {1,2,3,4,5,6,7,8},
            new double[] {8,7,6,5,4,3,2,1}
        };

        var (outputs, cache) = sa.Forward(inputs, startPosition: 0);
        Assert.AreEqual(inputs.Count, outputs.Count);
        Assert.AreEqual(embedSize, outputs[0].Length);
        // Additional checks can be added to verify outputs
    }

    [Test]
    public void TestBackward()
    {
        int embedSize = 8;
        int numHeads = 2;
        int numQueryGroups = 1;
        Random rand = new Random(0);
        SelfAttention sa = new SelfAttention(embedSize, numHeads, numQueryGroups, rand);

        List<double[]> inputs = new List<double[]>()
        {
            new double[] {1,2,3,4,5,6,7,8},
            new double[] {8,7,6,5,4,3,2,1}
        };

        var (outputs, cache) = sa.Forward(inputs, startPosition: 0);
        List<double[]> gradOutputs = new List<double[]>()
        {
            new double[] {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8},
            new double[] {0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1}
        };

        List<double[]> dInputs = sa.Backward(gradOutputs, cache);
        Assert.AreEqual(inputs.Count, dInputs.Count);
        Assert.AreEqual(embedSize, dInputs[0].Length);
        // Additional checks can be added to verify gradients
    }
}
