// BLAKE3 on Apple GPUs (Metal), compiled from this source by the GPU's
// driver when a process first uses the GPU (src/gpu.rs). Two kernels: the
// leaves (one thread a chunk) and a level of parents (one thread a pair),
// run level by level down to the root. Pairing each level's nodes in
// order, the last of an odd count carried up, gives BLAKE3's tree: the
// left subtree of every node holds the largest power of two of chunks
// under its count. The key words and the mode's flags come with each
// dispatch, so every mode runs here. Written after kobizk's in
// johnservil/BLAKE3#170.
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
constant uint CHUNK_START = 1, CHUNK_END = 2, PARENT = 4, ROOT = 8;

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

// One dispatch's arguments: the message's length (leaves) or the level's
// node count (parents), the first index this dispatch covers, the key
// words, and the mode's flags (0, KEYED_HASH, or DERIVE_KEY_MATERIAL).
struct Args {
    ulong length;
    ulong first;
    uint key[8];
    uint flags;
};

kernel void leaves(device const uchar *input [[buffer(0)]], device uint *output [[buffer(1)]],
                   constant Args &args [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    ulong index = args.first + tid, offset = index * 1024;
    uint len = uint(min(args.length - min(args.length, offset), ulong(1024)));
    uint blocks = len == 0 ? 1 : (len + 63) / 64;
    uint cv[8];
    for (uint i = 0; i < 8; i++) cv[i] = args.key[i];
    device const uchar *data = input + offset;
    for (uint b = 0; b < blocks; b++) {
        uint m[16] = {};
        uint take = min(len - b * 64, 64u);
        if (take == 64) {
            device const uint *words = reinterpret_cast<device const uint *>(data + b * 64);
            #pragma unroll
            for (uint j = 0; j < 16; j++) m[j] = words[j];
        } else {
            for (uint j = 0; j < take; j++) m[j / 4] |= uint(data[b * 64 + j]) << (8 * (j % 4));
        }
        uint flags = args.flags | (b == 0 ? b3::CHUNK_START : 0u) | (b + 1 == blocks ? b3::CHUNK_END : 0u);
        if (args.length <= 1024 && b + 1 == blocks) flags |= b3::ROOT;
        b3::compress(cv, m, index, take, flags);
    }
    for (uint j = 0; j < 8; j++) output[index * 8 + j] = cv[j];
}

kernel void parents(device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]],
                    constant Args &args [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    ulong count = args.length, index = args.first + tid, left = index * 2;
    uint cv[8];
    if (left + 1 < count) {
        uint m[16];
        for (uint i = 0; i < 16; i++) m[i] = input[left * 8 + i];
        for (uint i = 0; i < 8; i++) cv[i] = args.key[i];
        b3::compress(cv, m, 0, 64, args.flags | b3::PARENT | (count == 2 ? b3::ROOT : 0u));
    } else {
        for (uint j = 0; j < 8; j++) cv[j] = input[left * 8 + j];
    }
    for (uint j = 0; j < 8; j++) output[index * 8 + j] = cv[j];
}
