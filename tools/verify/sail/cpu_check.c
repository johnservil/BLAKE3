#include <stdio.h>
#include <stdint.h>
#include <string.h>
/* Each line: op a b k; prints the CPU's result in w0 (zero-extended). */
int main(void) {
  char op[8]; unsigned long long a, b, k;
  while (scanf("%7s %llu %llu %llu", op, &a, &b, &k) == 4) {
    uint64_t r;
    if (!strcmp(op, "add")) __asm__("add %w0, %w1, %w2" : "=r"(r) : "r"(a), "r"(b));
    else if (!strcmp(op, "eor")) __asm__("eor %w0, %w1, %w2" : "=r"(r) : "r"(a), "r"(b));
    else if (!strcmp(op, "eorror")) switch (k) {
#define E(n) case n: __asm__("eor %w0, %w1, %w2, ror #" #n : "=r"(r) : "r"(a), "r"(b)); break;
      E(0) E(1) E(2) E(3) E(4) E(5) E(6) E(7) E(8) E(9) E(10) E(11) E(12) E(13) E(14) E(15)
      E(16) E(17) E(18) E(19) E(20) E(21) E(22) E(23) E(24) E(25) E(26) E(27) E(28) E(29) E(30) E(31)
    }
    else switch (k) {
#define R(n) case n: __asm__("extr %w0, %w1, %w1, #" #n : "=r"(r) : "r"(a)); break;
      R(0) R(1) R(2) R(3) R(4) R(5) R(6) R(7) R(8) R(9) R(10) R(11) R(12) R(13) R(14) R(15)
      R(16) R(17) R(18) R(19) R(20) R(21) R(22) R(23) R(24) R(25) R(26) R(27) R(28) R(29) R(30) R(31)
    }
    printf("%llu\n", (unsigned long long)r);
  }
  return 0;
}
