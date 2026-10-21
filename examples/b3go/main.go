// A C-callable wrapper of 0u.io/b3 for examples/host_lab.rs, which loads it
// with dlopen and times it with the fork's clocks crate. Each function loops
// inside Go, so a timed call crosses from C into Go once per `iters` hashes.
package main

import "C"

import (
	"unsafe"

	"0u.io/b3"
)

func bytes(p *C.uchar, n C.size_t) []byte { return unsafe.Slice((*byte)(unsafe.Pointer(p)), int(n)) }

//export b3go_sum256
func b3go_sum256(p *C.uchar, n C.size_t, out *C.uchar, iters C.size_t) {
	data, o := bytes(p, n), bytes(out, 32)
	for range int(iters) {
		s := b3.Sum256(data)
		copy(o, s[:])
	}
}

//export b3go_parallel
func b3go_parallel(p *C.uchar, n C.size_t, out *C.uchar, iters C.size_t, threads C.int) {
	data, o := bytes(p, n), bytes(out, 32)
	for range int(iters) {
		h := b3.New(b3.WithParallelism(int(threads)))
		h.Write(data)
		h.Sum(o[:0])
	}
}

//export b3go_xof
func b3go_xof(p *C.uchar, n C.size_t, out *C.uchar, outlen C.size_t, iters C.size_t) {
	data, o := bytes(p, n), bytes(out, outlen)
	for range int(iters) {
		h := b3.New()
		h.Write(data)
		h.XOF().Read(o)
	}
}

func main() {}
