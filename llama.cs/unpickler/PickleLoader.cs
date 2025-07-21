namespace llama.unpickler.svd;

static class PickleLoader
{
    public static (int[] shape, float[] floats) readTensor (object[] args, string key) {
        var storageOffset = (int)args[1];

        if (storageOffset != 0) {
            throw new NotSupportedException ("Implementation expects storage offset to be 0.");
        }

        var shape = ((object[])args[2])
            .Select ((i => (int)i))
            .ToArray ();
        var stride = ((object[])args[3])
            .Select ((i => (int)i))
            .ToArray ();

        var tObject = ((DelayedExecutionUnpickler.TensorStream)args[0]);

        Console.WriteLine ($"loading {key} [{string.Join (",", shape)}] {tObject.dtype}");

        var bytes = new byte[tObject.data.Length];
        tObject.data.Read (bytes, 0, (int)tObject.data.Length);

        var floats = tObject.dtype switch {
            "BFloat16Storage" => convertBFloat16 (bytes),
            "FloatStorage" => convertFloat32 (bytes),
        };

        return (shape, floats);
    }

    static float[] convertBFloat16 (byte[] src) {
        if (src.Length % 2 != 0) {
            throw new ArgumentException ("Invalid array length");
        }

        var dst = new float[src.Length / 2];

        var dstPtr = 0;
        for (var srcPtr = 0; srcPtr < src.Length; srcPtr += 2) {
            // Extract the 2 bytes for the BFloat16 value
            var a = src[srcPtr];
            var b = src[srcPtr + 1];

            // Convert BFloat16 to float (pad with 16 LSBs of zeros)
            var floatBits = (short)((b << 8) | a) << 16;

            unsafe {
                // Reinterpret the bits as a float
                dst[dstPtr++] = (*((float*)&floatBits));
            }
        }

        return dst;
    }

    static float[] convertFloat32 (byte[] src) {
        if (src.Length % 4 != 0) {
            throw new ArgumentException ("Invalid array length");
        }

        var dst = new float[src.Length / 4];

        var dstPtr = 0;
        for (var srcPtr = 0; srcPtr < src.Length; srcPtr += 4) {
            var a = src[srcPtr];
            var b = src[srcPtr + 1];
            var c = src[srcPtr + 2];
            var d = src[srcPtr + 3];

            var floatBits = ((d << 24) | (c << 16) | (b << 8) | a);

            unsafe {
                // Reinterpret the bits as a float
                dst[dstPtr++] = (*((float*)&floatBits));
            }
        }

        return dst;
    }
}
