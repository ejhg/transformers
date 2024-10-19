using NUnit.Framework;

namespace llama;

[TestFixture]
public class FeedForwardTests
{
    [Test]
    public void TestForward () {
        int embedSize = 4;
        int hiddenSize = 8;
        Random rand = new Random (0);
        FeedForward ff = new FeedForward (embedSize, hiddenSize, rand);
        double[] x = {
            1.0,
            2.0,
            3.0,
            4.0
        };
        var (output, cache) = ff.Forward (x);
        Assert.AreEqual (embedSize, output.Length);
        // Additional checks can be added to verify specific values
    }

    [Test]
    public void TestBackward () {
        int embedSize = 4;
        int hiddenSize = 8;
        Random rand = new Random (0);
        FeedForward ff = new FeedForward (embedSize, hiddenSize, rand);
        double[] x = {
            1.0,
            2.0,
            3.0,
            4.0
        };
        var (output, cache) = ff.Forward (x);
        double[] dOut = {
            0.1,
            0.2,
            0.3,
            0.4
        };
        double[] dx = ff.Backward (dOut, cache);
        Assert.AreEqual (embedSize, dx.Length);
        // Additional checks can be added to verify gradients
    }
}
