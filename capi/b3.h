/* libb3_servil: the C API of blake3-servil.
 *
 * The C API is the library's convenience API: one message hashed at once,
 * or fed in pieces, on the calling thread, with every feature of BLAKE3
 * (the plain hash, the keyed hash, key derivation, extended output). The
 * batch, multi-core and stream calls are the Rust crate's throughput API.
 * Hashing entry points initialize the library automatically. Initialization
 * runs a startup self-test, which stops the program if this build or CPU
 * computes wrong digests. Later initialization calls check the completed
 * state.
 *
 * Memory: pointers cover live, correctly aligned storage of the stated
 * sizes, readable for inputs and writable for outputs. Output storage may
 * hold anything, uninitialized too: every call writes all of it. Read-only ranges
 * may overlap; a written range is disjoint from every other range a call
 * uses. Inputs stay unchanged, and outputs are not accessed concurrently
 * by the caller. Simple guards reject misaligned handles, unrepresentable
 * sized ranges, and immediate hasher data overlaps before borrowing
 * them. Detected invalid arguments stop the program with a message.
 * These checks do not establish allocation ownership or lifetime. The
 * remaining memory rules are caller obligations; violations may be
 * undefined behavior.
 *
 * Threads: any call may be made from any thread, and every call runs on
 * the calling thread alone. A b3_mode is a value and may be shared and
 * read by any number of threads at once. Calls using the same b3_hasher
 * are serialized by the caller.
 *
 * Link with -lb3_servil (libb3_servil.a, or libb3_servil.so /
 * libb3_servil.dylib). Build and install with `cargo cinstall --release`
 * in capi/; `pkg-config --cflags --libs b3_servil` gives the compiler flags.
 * The shared library has a versioned soname. C11, or C++11.
 * B3_VERSION names the header's version; b3_version() the library's, so a
 * program can check that the two agree.
 */
#define B3_VERSION "0.8.0+b3c5efc7c975d903762dea809066f7c0a0603861"
#ifndef B3_H
#define B3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The library's version, the same text as B3_VERSION in the header it was
 * built with. */
const char *b3_version(void);

/* A digest: 32 bytes. */
typedef struct {
  uint8_t bytes[32];
} b3_digest;

/* A mode: which of BLAKE3's three functions a call computes. Make one with
 * b3_mode_plain, b3_mode_keyed, or b3_mode_derive_key, and pass it to as many calls as
 * you like; its fields are the library's. */
typedef struct {
  uint8_t kind;
  uint8_t key[32];
} b3_mode;

/* The plain hash function. */
b3_mode b3_mode_plain(void);

/* The keyed hash function (a MAC) with this 32-byte key. */
b3_mode b3_mode_keyed(const uint8_t key[32]);

/* Key derivation (a KDF) with this context: a NUL-terminated UTF-8 string,
 * hardcoded, globally unique, and application-specific, such as
 * "example.com 2026-10-16 session keys". The context is hashed here, once;
 * the calls' inputs are then the key material. */
b3_mode b3_mode_derive_key(const char *context);

/* The digest of `len` bytes at `input` (`input` may be NULL when `len` is
 * 0), on the calling thread. */
b3_digest b3_hash(const b3_mode *mode, const void *input, size_t len);

/* One message fed in pieces. A b3_hasher is your memory (18 KiB; on the
 * stack, in a struct, or allocated). It needs no library destructor; the
 * caller frees any heap storage it supplied. b3_hasher_new starts a
 * message, and starts one over; `length_hint` is the message's likely
 * length in bytes where the program knows one (a file's size), else 0 with
 * `has_length_hint` 0: a scheduling hint, never part of the hash, so the
 * digest is the same whatever it says. Heap storage must also be 64-byte aligned;
 * ordinary malloc does not guarantee this over-alignment. Use a
 * platform-appropriate aligned allocator and its matching deallocator.
 * Use the header matching the linked library. */
#define B3_HASHER_SIZE 18432
#ifdef __cplusplus
#define B3_ALIGN64 alignas(64)
#else
#define B3_ALIGN64 _Alignas(64)
#endif
typedef struct {
  B3_ALIGN64 uint8_t opaque[B3_HASHER_SIZE];
} b3_hasher;

void b3_hasher_new(b3_hasher *hasher, const b3_mode *mode, int has_length_hint, uint64_t length_hint);

/* Add the message's next `len` bytes. */
void b3_hasher_update(b3_hasher *hasher, const void *input, size_t len);

/* The digest of the bytes so far; the hasher can take more. */
b3_digest b3_hasher_finalize(const b3_hasher *hasher);

/* Extended output: `len` bytes of the output from byte `offset` on, into
 * `out`. Its first 32 bytes are the digest. */
void b3_hasher_finalize_xof(const b3_hasher *hasher, uint64_t offset, void *out, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* B3_H */
