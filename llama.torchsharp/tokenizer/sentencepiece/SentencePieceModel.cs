using System.Text;

/// <summary>
/// Represents the SentencePiece part of the model.
/// </summary>
public class SentencePieceInfo
{
    public string Piece { get; set; }
    public float Score { get; set; }
    public int Type { get; set; } // If you want, you can map this to an enum.
}

/// <summary>
/// A minimal representation of the entire model, containing only the repeated
/// sentencepiece field.
/// </summary>
public class MinimalSentencePieceModel
{
    public List<SentencePieceInfo> SentencePieces { get; } = new List<SentencePieceInfo> ();
}

/// <summary>
/// Main class to parse a SentencePiece model file (protobuf, wire format)
/// without external dependencies.
/// </summary>
public static class SentencePieceModelParser
{
    /// <summary>
    /// Parses the entire model, returning only the repeated SentencePieces.
    /// </summary>
    public static MinimalSentencePieceModel Parse (byte[] data) {
        // We'll keep it simple and only look for ModelProto's field #1 (repeated SentencePiece).
        // Then inside each SentencePiece sub-message, we look for:
        //    piece = 1 (string)
        //    score = 2 (float, wiretype = 5)
        //    type  = 3 (varint)
        //
        // Everything else is ignored.
        var model = new MinimalSentencePieceModel ();

        // Use a ProtoReader to walk through the top-level message.
        var reader = new ProtoReader (data);

        // Read until we run out of data
        while (!reader.Eof) {
            // Read the next field tag
            var (fieldNumber, wireType) = reader.ReadTag ();
            if (fieldNumber == 0 && wireType == 0) {
                // Means we reached the end or encountered an error
                break;
            }

            switch (fieldNumber) {
                // ModelProto's "repeated SentencePiece sentencepiece = 1;"
                case 1 when wireType == WireType.LengthDelimited:
                    // We have a sub-message (SentencePiece)
                    int subMessageLength = reader.ReadVarint (); // length of submessage
                    var subMessageData = reader.ReadBytes (subMessageLength);

                    // Parse one SentencePiece submessage
                    var sp = ParseSentencePiece (subMessageData);
                    model.SentencePieces.Add (sp);
                    break;

                // Skip anything else
                default:
                    reader.SkipField (wireType);
                    break;
            }
        }

        return model;
    }

    /// <summary>
    /// Parses a single SentencePiece submessage.
    /// </summary>
    private static SentencePieceInfo ParseSentencePiece (byte[] data) {
        var sp = new SentencePieceInfo ();
        var reader = new ProtoReader (data);

        while (!reader.Eof) {
            var (fieldNumber, wireType) = reader.ReadTag ();
            if (fieldNumber == 0 && wireType == 0) {
                // End or invalid
                break;
            }

            switch (fieldNumber) {
                // string piece = 1 (wiretype = length-delimited)
                case 1 when wireType == WireType.LengthDelimited:
                    int strLen = reader.ReadVarint ();
                    byte[] strBytes = reader.ReadBytes (strLen);
                    sp.Piece = Encoding.UTF8.GetString (strBytes);
                    break;

                // float score = 2 (wiretype = 5 -> 32-bit)
                case 2 when wireType == WireType.Bit32:
                    sp.Score = reader.ReadFloat ();
                    break;

                // Type type = 3 (wiretype = 0 -> varint)
                case 3 when wireType == WireType.Varint:
                    sp.Type = reader.ReadVarint ();
                    break;

                default:
                    reader.SkipField (wireType);
                    break;
            }
        }

        return sp;
    }
}

/// <summary>
/// Wire types in Protobuf encoding.
/// </summary>
public enum WireType
{
    Varint = 0, // int32, int64, bool, enum
    Bit64 = 1, // fixed64, double
    LengthDelimited = 2, // string, bytes, sub-message
    StartGroup = 3, // groups (deprecated)
    EndGroup = 4, // groups (deprecated)
    Bit32 = 5 // fixed32, float
}

/// <summary>
/// A bare-bones reader for protobuf wire format from a byte array.
/// </summary>
public class ProtoReader
{
    private readonly byte[] _data;
    private int _position;

    public ProtoReader (byte[] data) {
        _data = data ?? throw new ArgumentNullException (nameof(data));
        _position = 0;
    }

    /// <summary>
    /// Read next “tag” which is (field_number << 3 | wire_type).
    /// Returns (0, 0) if we are at EOF or invalid.
    /// </summary>
    public (int fieldNumber, WireType wireType) ReadTag () {
        if (Eof) return (0, 0);

        // A tag is a varint.
        int tag = ReadVarint ();
        if (tag == 0) return (0, 0);

        int fieldNumber = tag >> 3;
        int wireType = tag & 7;

        return (fieldNumber, (WireType)wireType);
    }

    /// <summary>
    /// Reads a varint (up to 32-bit) from the buffer.
    /// Returns 0 if we are at EOF or an error occurs.
    /// </summary>
    public int ReadVarint () {
        int result = 0;
        int shift = 0;

        while (!Eof) {
            byte b = _data[_position++];
            // lower 7 bits
            result |= (b & 0x7F) << shift;
            if ((b & 0x80) == 0) {
                // no continuation
                return result;
            }

            shift += 7;
            if (shift > 31) {
                // Guard against malformed varint
                throw new FormatException ("Malformed varint (too many bytes).");
            }
        }

        return 0; // EOF
    }

    /// <summary>
    /// Reads a single-precision float (wire type = 5) from the buffer.
    /// </summary>
    public float ReadFloat () {
        if (_position + 4 > _data.Length)
            throw new IndexOutOfRangeException ("Not enough bytes to read float.");

        // 4 bytes, little-endian
        float value = BitConverter.ToSingle (_data, _position);
        _position += 4;
        return value;
    }

    /// <summary>
    /// Reads count bytes from the buffer.
    /// </summary>
    public byte[] ReadBytes (int count) {
        if (_position + count > _data.Length)
            throw new IndexOutOfRangeException ("Not enough bytes to read.");

        var segment = new byte[count];
        Buffer.BlockCopy (_data, _position, segment, 0, count);
        _position += count;
        return segment;
    }

    /// <summary>
    /// Skip a field of a given wire type if we don't need it.
    /// </summary>
    public void SkipField (WireType wireType) {
        switch (wireType) {
            case WireType.Varint:
                ReadVarint (); // discard
                break;
            case WireType.Bit64:
                SkipBytes (8);
                break;
            case WireType.LengthDelimited:
                int length = ReadVarint ();
                SkipBytes (length);
                break;
            case WireType.StartGroup:
            case WireType.EndGroup:
                // Groups are deprecated, but we’d have to recursively skip
                // until we find the matching EndGroup. We'll omit that here.
                throw new NotSupportedException ("Groups not supported in this demo.");
            case WireType.Bit32:
                SkipBytes (4);
                break;
        }
    }

    private void SkipBytes (int count) {
        if (_position + count > _data.Length)
            throw new IndexOutOfRangeException ("Skipping past the end of buffer.");
        _position += count;
    }

    /// <summary>
    /// Tells if we’ve reached the end of the data.
    /// </summary>
    public bool Eof => _position >= _data.Length;
}
