// BLAKE3 on Apple GPUs (Metal), compiled from this source by the GPU's
// driver when a process first uses the GPU (src/gpu.rs). One kernel: a
// batch of a stream's tasks, each a whole 64 KiB piece of a long message
// (a subtree of 64 chunks at its chunk counter), one threadgroup a task:
// a thread a chunk, then the group's chaining values merged a level a
// step. The key words and the mode's flags come with each dispatch, so
// every mode runs here. Written after kobizk's in johnservil/BLAKE3#170.
#include <metal_stdlib>
using namespace metal;

namespace b3 {
constant uint iv[4] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a};
constant uchar schedule[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}};
constant uint CHUNK_START = 1, CHUNK_END = 2, PARENT = 4;

inline uint rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }
inline void g(thread uint *s, uint a, uint b, uint c, uint d, uint x, uint y) {
    s[a] += s[b] + x; s[d] = rotr(s[d] ^ s[a], 16); s[c] += s[d]; s[b] = rotr(s[b] ^ s[c], 12);
    s[a] += s[b] + y; s[d] = rotr(s[d] ^ s[a], 8); s[c] += s[d]; s[b] = rotr(s[b] ^ s[c], 7);
}
inline void compress(thread uint *cv, thread const uint *m, ulong counter, uint len, uint flags) {
    uint s[16];
    for (uint i = 0; i < 8; i++) s[i] = cv[i];
    for (uint i = 0; i < 4; i++) s[8 + i] = iv[i];
    s[12] = uint(counter); s[13] = uint(counter >> 32); s[14] = len; s[15] = flags;
    #pragma unroll
    for (uint r = 0; r < 7; r++) {
        g(s, 0, 4, 8, 12, m[schedule[r][0]], m[schedule[r][1]]);
        g(s, 1, 5, 9, 13, m[schedule[r][2]], m[schedule[r][3]]);
        g(s, 2, 6, 10, 14, m[schedule[r][4]], m[schedule[r][5]]);
        g(s, 3, 7, 11, 15, m[schedule[r][6]], m[schedule[r][7]]);
        g(s, 0, 5, 10, 15, m[schedule[r][8]], m[schedule[r][9]]);
        g(s, 1, 6, 11, 12, m[schedule[r][10]], m[schedule[r][11]]);
        g(s, 2, 7, 8, 13, m[schedule[r][12]], m[schedule[r][13]]);
        g(s, 3, 4, 9, 14, m[schedule[r][14]], m[schedule[r][15]]);
    }
    for (uint i = 0; i < 8; i++) cv[i] = s[i] ^ s[i + 8];
}
} // namespace b3

// One dispatch's arguments: the key words and the mode's flags (0,
// KEYED_HASH, or DERIVE_KEY_MATERIAL).
struct Args {
    uint key[8];
    uint flags;
};

// A task: its piece's byte offset in the ring, and its first chunk's index.
struct Piece {
    ulong offset;
    ulong counter;
};

// Task t's 16 words of output: for a piece at chunk 0 (a message's first),
// the chaining values of its two 32 KiB halves, which the stream merges
// into the message's tree; for any other, the piece's chaining value and
// eight words unused. As the CPU's task writes (lanes::Task::run).
kernel void tasks(device const uchar *ring [[buffer(0)]], device uint *output [[buffer(1)]],
                  constant Args &args [[buffer(2)]], constant Piece *pieces [[buffer(3)]],
                  uint t [[threadgroup_position_in_grid]], uint c [[thread_position_in_threadgroup]]) {
    threadgroup uint cvs[64 * 8];
    Piece piece = pieces[t];
    device const uint *words = reinterpret_cast<device const uint *>(ring + piece.offset + c * 1024);
    uint cv[8];
    for (uint i = 0; i < 8; i++) cv[i] = args.key[i];
    for (uint b = 0; b < 16; b++) {
        uint m[16];
        #pragma unroll
        for (uint j = 0; j < 16; j++) m[j] = words[b * 16 + j];
        uint flags = args.flags | (b == 0 ? b3::CHUNK_START : 0u) | (b == 15 ? b3::CHUNK_END : 0u);
        b3::compress(cv, m, piece.counter + c, 64, flags);
    }
    for (uint j = 0; j < 8; j++) cvs[c * 8 + j] = cv[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Down to the two halves' values, in cvs[0..16].
    for (uint width = 32; width >= 2; width >>= 1) {
        uint p[8];
        if (c < width) {
            uint m[16];
            for (uint i = 0; i < 16; i++) m[i] = cvs[c * 16 + i];
            for (uint i = 0; i < 8; i++) p[i] = args.key[i];
            b3::compress(p, m, 0, 64, args.flags | b3::PARENT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (c < width) {
            for (uint j = 0; j < 8; j++) cvs[c * 8 + j] = p[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (c == 0) {
        device uint *out = output + t * 16;
        if (piece.counter == 0) {
            for (uint j = 0; j < 16; j++) out[j] = cvs[j];
        } else {
            uint m[16], p[8];
            for (uint i = 0; i < 16; i++) m[i] = cvs[i];
            for (uint i = 0; i < 8; i++) p[i] = args.key[i];
            b3::compress(p, m, 0, 64, args.flags | b3::PARENT);
            for (uint j = 0; j < 8; j++) out[j] = p[j];
        }
    }
}
