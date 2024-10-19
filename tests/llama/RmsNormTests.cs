using NUnit.Framework;

namespace llama;

[TestFixture]
public class RMSNormTests
{
    [Test]
    public void TestForward () {
        int size = 4;
        RMSNorm rmsNorm = new RMSNorm (size);
        double[] x = {
            1.0,
            2.0,
            3.0,
            4.0
        };
        var (output, cache) = rmsNorm.Forward (x);
        double meanSquare = x.Select (val => val * val).Average ();
        double rms = Math.Sqrt (meanSquare + 1e-6);
        double[] expected = new double[size];
        for (int i = 0; i < size; i++) {
            expected[i] = x[i] * (rmsNorm.Gamma[i] / rms);
        }

        Assert.AreEqual (expected, output);
    }

    [Test]
    public void TestBackward () {
        int size = 4;
        RMSNorm rmsNorm = new RMSNorm (size);
        double[] x = {
            1.0,
            2.0,
            3.0,
            4.0
        };
        var (output, cache) = rmsNorm.Forward (x);
        double[] gradOutput = {
            0.1,
            0.2,
            0.3,
            0.4
        };
        double[] dx = rmsNorm.Backward (gradOutput, cache);
        Assert.AreEqual (size, dx.Length);
        Assert.AreEqual (size, rmsNorm.dGamma.Length);
        // Additional checks can be added to verify specific values
    }
}
