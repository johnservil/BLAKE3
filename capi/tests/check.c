/* The C API from C: b3.h compiles as C11 with every warning on, and each
 * call's digest of a few inputs is printed in hex, for tests/c.rs to check
 * against the Rust crate's. */
#include <stdio.h>
#include <string.h>

#include "b3.h"

static void print(const char *what, const uint8_t *bytes, size_t len) {
  printf("%s ", what);
  for (size_t i = 0; i < len; i++) printf("%02x", bytes[i]);
  printf("\n");
}

int main(void) {
  if (strcmp(b3_version(), B3_VERSION) != 0) return 2; /* the header and the library disagree */
  printf("hasher %zu %zu\n", sizeof(b3_hasher), _Alignof(b3_hasher));
  static uint8_t input[100000]; /* any placement: the C calls take any bytes */
  for (size_t i = 0; i < sizeof input; i++) input[i] = (uint8_t)(i % 251);
  uint8_t key[32];
  memcpy(key, "whats the Elvish word for friend", 32);
  b3_mode modes[3] = {b3_mode_plain(), b3_mode_keyed(key), b3_mode_derive_key("BLAKE3 2019-12-27 16:29:52 test vectors context")};
  const char *names[3] = {"hash", "keyed_hash", "derive_key"};
  size_t lens[3] = {0, 1025, 100000};
  for (int m = 0; m < 3; m++) {
    for (int l = 0; l < 3; l++) {
      char what[64];
      snprintf(what, sizeof what, "%s/%zu", names[m], lens[l]);
      b3_digest d = b3_hash(&modes[m], input, lens[l]);
      print(what, d.bytes, 32);
      static b3_hasher hasher;
      b3_hasher_new(&hasher, &modes[m], 0, 0);
      b3_hasher_update(&hasher, input, lens[l] / 2);
      b3_hasher_update(&hasher, input + lens[l] / 2, lens[l] - lens[l] / 2);
      d = b3_hasher_finalize(&hasher);
      print(what, d.bytes, 32);
    }
  }
  return 0;
}

