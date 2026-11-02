#include <metal_stdlib>
using namespace metal;
namespace b3gpu {
constant uint iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                      0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
constant uchar schedule[7][16] = {
 {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
 {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
 {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
 {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
 {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
 {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
 {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}};
inline uint rotr(uint x, uint n) { return (x >> n) | (x << (32-n)); }
inline void g(thread uint *s, uint a, uint b, uint c, uint d, uint x, uint y) {
 s[a] += s[b]+x; s[d]=rotr(s[d]^s[a],16); s[c]+=s[d]; s[b]=rotr(s[b]^s[c],12);
 s[a] += s[b]+y; s[d]=rotr(s[d]^s[a],8); s[c]+=s[d]; s[b]=rotr(s[b]^s[c],7);
}
inline void compress(thread uint *cv, thread const uint *m,
                     ulong counter, uint len, uint flags) {
 uint s[16];
 for(uint i=0;i<8;i++) s[i]=cv[i];
 for(uint i=0;i<4;i++) s[8+i]=iv[i];
 s[12]=uint(counter); s[13]=uint(counter>>32); s[14]=len; s[15]=flags;
 #pragma unroll
 for(uint r=0;r<7;r++) {
  g(s,0,4,8,12,m[schedule[r][0]],m[schedule[r][1]]);
  g(s,1,5,9,13,m[schedule[r][2]],m[schedule[r][3]]);
  g(s,2,6,10,14,m[schedule[r][4]],m[schedule[r][5]]);
  g(s,3,7,11,15,m[schedule[r][6]],m[schedule[r][7]]);
  g(s,0,5,10,15,m[schedule[r][8]],m[schedule[r][9]]);
  g(s,1,6,11,12,m[schedule[r][10]],m[schedule[r][11]]);
  g(s,2,7,8,13,m[schedule[r][12]],m[schedule[r][13]]);
  g(s,3,4,9,14,m[schedule[r][14]],m[schedule[r][15]]);
 }
 for(uint i=0;i<8;i++) cv[i]=s[i]^s[i+8];
}
inline void chunk(device const uchar *data, uint len, ulong counter,
                  bool root, constant uint *key, uint mode, thread uint *cv) {
 for(uint i=0;i<8;i++) cv[i]=key[i];
 uint blocks = len == 0 ? 1 : (len+63)/64;
 for(uint b=0;b<blocks;b++) {
  uint m[16] = {}; uint remaining=len-b*64; uint take=remaining<64?remaining:64;
  if (take == 64) {
   device const uint *words = reinterpret_cast<device const uint *>(data+b*64);
   #pragma unroll
   for(uint j=0;j<16;j++) m[j]=words[j];
  } else {
   for(uint j=0;j<take;j++) m[j/4] |= uint(data[b*64+j]) << (8*(j%4));
  }
  uint flags=mode | (b==0?1u:0u) | (b+1==blocks?2u:0u);
  if(root && b+1==blocks) flags|=8;
  compress(cv,m,counter,take,flags);
 }
}
inline void parent(device const uint *children, bool root, constant uint *key, uint mode, thread uint *cv) {
 uint m[16];
 for(uint i=0;i<16;i++) m[i]=children[i];
 for(uint i=0;i<8;i++) cv[i]=key[i];
 compress(cv,m,0,64,mode|4u|(root?8u:0u));
}
}

// A dispatch's message length, first node, and the mode's key and flags.
struct LeafArgs { ulong length; ulong first; uint key[8]; uint mode; };
kernel void leaves(device const uchar *input [[buffer(0)]],
                   device uint *output [[buffer(1)]],
                   constant LeafArgs &args [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
 ulong index=args.first+tid, offset=index*1024;
 ulong remaining=args.length>offset?args.length-offset:0;
 uint cv[8];
 b3gpu::chunk(input+offset,uint(min(remaining,ulong(1024))),index,args.length<=1024,args.key,args.mode,cv);
 for(uint j=0;j<8;j++) output[index*8+j]=cv[j];
}
kernel void parents(device const uint *input [[buffer(0)]],
                    device uint *output [[buffer(1)]],
                    constant LeafArgs &args [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
 ulong count=args.length, index=args.first+tid, left=index*2;
 uint cv[8];
 if(left+1<count) b3gpu::parent(input+left*8,count==2,args.key,args.mode,cv);
 else for(uint j=0;j<8;j++) cv[j]=input[left*8+j];
 for(uint j=0;j<8;j++) output[index*8+j]=cv[j];
}
